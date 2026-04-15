"""
WDM signal generation and propagation setup.

This module handles:
1. RRC filter design for pulse shaping
2. QAM constellation generation
3. WDM field construction with multiple channels
4. GNLSE setup for linear and nonlinear propagation
"""

import numpy as np
import math
from pathlib import Path
import sys

# Import GNLSE (adjust import path as needed)
try:
    import gnlse
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import gnlse

from gnlse.common import c


def rrc_taps(rolloff, span_symbols, sps):
    """
    Root-raised-cosine filter taps, normalized to unit energy.

    Args:
        rolloff: Rolloff factor (0 < rolloff <= 1)
        span_symbols: Duration in symbols
        sps: Samples per symbol

    Returns:
        RRC filter taps (complex or real)
    """
    if rolloff <= 0 or rolloff > 1:
        raise ValueError("rolloff must be in (0, 1]")
    if span_symbols < 1:
        raise ValueError("span_symbols must be >= 1")

    t = np.arange(-span_symbols / 2, span_symbols / 2 + 1 / sps, 1 / sps)
    taps = np.zeros_like(t)

    for i, ti in enumerate(t):
        if np.isclose(ti, 0.0):
            taps[i] = 1.0 + rolloff * (4 / np.pi - 1)
        elif np.isclose(abs(ti), 1 / (4 * rolloff)):
            alpha = rolloff
            taps[i] = (
                alpha / np.sqrt(2)
                * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * alpha))
                   + (1 - 2 / np.pi) * np.cos(np.pi / (4 * alpha)))
            )
        else:
            numerator = (
                np.sin(np.pi * ti * (1 - rolloff))
                + 4 * rolloff * ti * np.cos(np.pi * ti * (1 + rolloff))
            )
            denominator = np.pi * ti * (1 - (4 * rolloff * ti) ** 2)
            taps[i] = numerator / denominator

    taps /= np.sqrt(np.sum(taps**2))
    return taps


def nyquist_sinc_taps(span_symbols, sps):
    """Ideal Nyquist (sinc) pulse taps truncated to finite span."""
    if span_symbols < 1:
        raise ValueError("span_symbols must be >= 1")
    t = np.arange(-span_symbols / 2, span_symbols / 2 + 1 / sps, 1 / sps)
    taps = np.sinc(t)
    taps /= np.sqrt(np.sum(taps**2))
    return taps


def qam_symbols(order, count, rng):
    """
    Generate square QAM constellation with unit average power.

    Args:
        order: QAM order (must be perfect square, e.g., 16, 64, 256)
        count: Number of symbols to generate
        rng: NumPy random generator

    Return:        Array of complex constellation symbols
    """
    m = int(round(math.sqrt(order)))
    if m * m != order:
        raise ValueError("QAM order must be a perfect square")

    levels = np.arange(-(m - 1), m, 2)
    const = (levels[None, :] + 1j * levels[:, None]).reshape(-1)
    const /= np.sqrt(np.mean(np.abs(const)**2))
    return rng.choice(const, size=count)


def qam_axis_levels(order):
    """
    Get normalized I/Q coordinates of ideal square-QAM symbol centers.

    Args:
        order: QAM order

    Returns:
        Array of normalized symbol positions
    """
    m = int(round(math.sqrt(order)))
    if m * m != order:
        raise ValueError("QAM order must be a perfect square")

    levels = np.arange(-(m - 1), m, 2, dtype=np.float64)
    const = (levels[None, :] + 1j * levels[:, None]).reshape(-1)
    scale = np.sqrt(np.mean(np.abs(const)**2))
    return levels / scale


def upsample_and_filter(symbols, sps, taps):
    """
    Upsample symbols and apply pulse shaping filter.

    Args:
        symbols: Array of modulation symbols
        sps: Samples per symbol
        taps: Pulse shaping filter taps

    Returns:
        Pulse-shaped waveform
    """
    up = np.zeros(len(symbols) * sps, dtype=np.complex128)
    up[::sps] = symbols
    shaped = np.convolve(up, taps, mode='same')
    return shaped


def upsample_and_nyquist_rect(symbols, sps, t, symbol_rate_thz):
    """Upsample symbols and apply ideal rectangular-spectrum Nyquist shaping."""
    up = np.zeros(len(symbols) * sps, dtype=np.complex128)
    up[::sps] = symbols

    dt = float(t[1] - t[0])
    freq = np.fft.fftshift(np.fft.fftfreq(len(up), d=dt))
    aw = np.fft.fftshift(np.fft.ifft(up))
    mask = np.abs(freq) <= (symbol_rate_thz / 2.0)
    aw *= mask.astype(np.float64)
    return np.fft.fft(np.fft.ifftshift(aw))


def scale_to_power(x, target_power_w):
    """
    Scale signal to achieve target power.

    Args:
        x: Input signal
        target_power_w: Target power in Watts

    Returns:
        Scaled signal with target power
    """
    power = np.mean(np.abs(x)**2)
    if power <= 0:
        return x
    return x * math.sqrt(target_power_w / power)


def make_wdm_field(t, channel_spacing_thz, n_channels, power_per_channel_w,
                   symbols_per_channel, sps, taps, modulation_order, rng,
                   pulse_shape="rrc", symbol_rate_thz=None,
                   rrc_rolloff=None):
    """
    Generate complex WDM field with multiple modulated channels.

    Creates a WDM signal with n_channels equally spaced channels, each 
    carrying a QAM-modulated signal with RRC pulse shaping.

    Args:
        t: Time vector
        channel_spacing_thz: Spacing between channels (THz)
        n_channels: Number of channels (must be odd to center one at 0 THz)
        power_per_channel_w: Power per channel (W)
        symbols_per_channel: Number of symbols per channel
        sps: Samples per symbol
        taps: Pulse-shaping taps (RRC or sinc)
        modulation_order: QAM order
        rng: NumPy random generator
        pulse_shape: "rrc" or "nyquist_rect"
        symbol_rate_thz: Symbol rate in THz. If None, inferred from t and sps
        rrc_rolloff: RRC rolloff in [0, 1] used for Nyquist guard (rrc mode)

    Returns:
        Tuple of (complex_field, center_channel_symbols)
    """
    if len(t) < 2:
        raise ValueError("time vector t must contain at least 2 samples")
    if sps <= 0:
        raise ValueError("sps must be positive")
    if n_channels % 2 == 0:
        raise ValueError("n_channels must be odd to keep a center channel at 0 THz")
    if pulse_shape not in {"rrc", "nyquist_rect"}:
        raise ValueError("pulse_shape must be 'rrc' or 'nyquist_rect'")

    dt = float(t[1] - t[0])
    if dt <= 0:
        raise ValueError("time vector must be strictly increasing")

    if symbol_rate_thz is None:
        symbol_rate_thz = 1.0 / (dt * float(sps))
    if symbol_rate_thz <= 0:
        raise ValueError("symbol_rate_thz must be positive")

    if pulse_shape == "rrc":
        rolloff = 0.0 if rrc_rolloff is None else float(rrc_rolloff)
        if rolloff < 0 or rolloff > 1:
            raise ValueError("rrc_rolloff must be in [0, 1]")
        channel_bw_thz = symbol_rate_thz * (1.0 + rolloff)
    else:
        channel_bw_thz = symbol_rate_thz

    f_max_thz = 0.5 / dt
    max_offset_thz = channel_spacing_thz * (n_channels - 1) / 2.0
    if max_offset_thz + channel_bw_thz / 2.0 > f_max_thz:
        raise ValueError(
            "Channel plan exceeds Nyquist bandwidth in make_wdm_field: "
            f"max_offset + bw/2 = {max_offset_thz + channel_bw_thz / 2.0:.6g} THz, "
            f"f_max = {f_max_thz:.6g} THz"
        )

    offsets = channel_spacing_thz * (np.arange(n_channels) -
                                     (n_channels - 1) / 2.0)

    field = np.zeros_like(t, dtype=np.complex128)
    center_symbols = None

    for f_thz in offsets:
        # creates a stream of
        symbols = qam_symbols(modulation_order, symbols_per_channel, rng) # notice that depends only on symbols_per_channel
        if pulse_shape == "rrc":
            baseband = upsample_and_filter(symbols, sps, taps) # here is the place where the samples per symbol get in.
        elif pulse_shape == "nyquist_rect":
            baseband = upsample_and_nyquist_rect(
                symbols, sps, t, symbol_rate_thz)
        baseband = scale_to_power(baseband, power_per_channel_w)

        field += baseband * np.exp(1j * (2 * np.pi * f_thz * t)) # here is the mixing of the basebands to the fields.
        if np.isclose(f_thz, 0.0):
            center_symbols = symbols

    if center_symbols is None:
        raise ValueError("No center channel found at 0 THz.")

    return field, center_symbols


def make_gnlse_setup(A_t, *, resolution, time_window, wavelength_nm,
                     fiber_length_m, nonlinearity, betas, loss_db_per_m,
                     z_saves, rtol, atol):
    """
    Create GNLSE setup for signal propagation.

    Args:
        A_t: Initial time-domain field
        resolution: Number of time samples
        time_window: Time window (ps)
        wavelength_nm: Central wavelength (nm)
        fiber_length_m: Fiber length (m)
        nonlinearity: Nonlinear coefficient gamma (1/W/m)
        betas: Dispersion coefficients
        loss_db_per_m: Loss (dB/m)
        z_saves: Number of z-positions to save
        rtol: Relative ODE tolerance
        atol: Absolute ODE tolerance

    Returns:
        Configured GNLSESetup object
    """
    setup = gnlse.GNLSESetup()
    setup.resolution = resolution
    setup.time_window = time_window
    setup.wavelength = wavelength_nm
    setup.fiber_length = fiber_length_m
    setup.z_saves = z_saves
    setup.rtol = rtol
    setup.atol = atol

    setup.nonlinearity = nonlinearity
    setup.pulse_model = A_t
    setup.dispersion_model = gnlse.DispersionFiberFromTaylor(
        loss_db_per_m, betas)
    setup.self_steepening = False
    setup.raman_model = None

    return setup


def propagate_field_constrained(
        A_t, *, resolution, time_window, wavelength_nm,
        fiber_length_m, nonlinearity, betas, loss_db_per_m,
        z_saves, rtol, atol, max_step_m, max_nonlinear_phase_deg,
        propagation_direction=1,
        propagation_backend="custom_ssfm"):
    """
    Propagate a field with guide-constrained step sizing.

    Enforces:
    1) absolute max step length (m)
    2) max nonlinear phase rotation per step (deg), using
       $dz \le \phi_{max}/(\gamma P_{peak})$

    Args:
        A_t: Initial time-domain field
        resolution: Number of time samples
        time_window: Time window (ps)
        wavelength_nm: Central wavelength (nm)
        fiber_length_m: Fiber length (m)
        nonlinearity: Nonlinear coefficient gamma (1/W/m)
        betas: Dispersion coefficients
        loss_db_per_m: Loss (dB/m)
        z_saves: Number of z snapshots for each segment
        rtol: Relative ODE tolerance
        atol: Absolute ODE tolerance
        max_step_m: Maximum segment length (m)
        max_nonlinear_phase_deg: Max nonlinear phase per segment (deg)
        propagation_direction: +1 (forward) or -1 (backward)
        propagation_backend: "custom_ssfm" or "gnlse"

    Notes:
        This implementation uses an in-memory symmetric SSFM loop and avoids
        rebuilding/rerunning a full `gnlse.GNLSE` object for every constrained
        segment. This is substantially faster when many short segments are
        needed by the nonlinear phase constraint.

    Returns:
        Tuple of (final_solution, segment_count)
    """
    if fiber_length_m < 0:
        raise ValueError("fiber_length_m must be non-negative")
    if max_step_m <= 0:
        raise ValueError("max_step_m must be positive")
    if max_nonlinear_phase_deg <= 0:
        raise ValueError("max_nonlinear_phase_deg must be positive")

    if propagation_direction not in (-1, 1):
        raise ValueError("propagation_direction must be +1 or -1")
    if propagation_backend not in {"custom_ssfm", "gnlse"}:
        raise ValueError(
            "propagation_backend must be 'custom_ssfm' or 'gnlse'")

    # Guide requirement: step bounded above by 1000 m.
    max_step_m = min(float(max_step_m), 1000.0)

    direction = float(propagation_direction)
    effective_nonlinearity = direction * float(nonlinearity)
    effective_betas = direction * np.asarray(betas, dtype=np.float64)
    effective_loss_db_per_m = direction * float(loss_db_per_m)

    # Build propagation grids/operators once (avoid per-segment GNLSE rebuild).
    n = int(resolution)
    t = np.linspace(-time_window / 2.0, time_window / 2.0, n)
    dt = t[1] - t[0]
    v = 2.0 * np.pi * np.arange(-n / 2, n / 2) / (n * dt)
    w_0 = (2.0 * np.pi * c) / wavelength_nm
    omega = v + w_0

    dispersion = gnlse.DispersionFiberFromTaylor(
        effective_loss_db_per_m, effective_betas)
    d = np.fft.fftshift(dispersion.D(v))

    field0 = np.asarray(A_t, dtype=np.complex128).copy()

    if fiber_length_m == 0:
        z = np.array([0.0, 0.0], dtype=np.float64)
        at = np.vstack((field0, field0))
        aw0 = np.fft.fftshift(np.fft.ifft(field0)) * n * dt
        aw = np.vstack((aw0, aw0))
        return gnlse.Solution(t, omega, w_0, z, at, aw), 0

    if propagation_backend == "gnlse":
        setup = make_gnlse_setup(
            field0,
            resolution=resolution,
            time_window=time_window,
            wavelength_nm=wavelength_nm,
            fiber_length_m=fiber_length_m,
            nonlinearity=effective_nonlinearity,
            betas=effective_betas,
            loss_db_per_m=effective_loss_db_per_m,
            z_saves=max(2, int(z_saves)),
            rtol=rtol,
            atol=atol,
        )
        return gnlse.GNLSE(setup).run(), 1

    phase_limit_rad = max_nonlinear_phase_deg * np.pi / 180.0
    gamma_abs = abs(float(nonlinearity))

    # Compute nonlinear phase constrained step ONCE from the input field,
    # then use a uniform step for the whole propagation.
    peak_power0 = float(np.max(np.abs(field0) ** 2))
    dz_nl = float("inf")
    if gamma_abs > 0.0 and peak_power0 > 0.0:
        dz_nl = phase_limit_rad / (gamma_abs * peak_power0)

    dz_target = min(float(fiber_length_m), max_step_m, dz_nl)
    if not np.isfinite(dz_target) or dz_target <= 0.0:
        dz_target = min(float(fiber_length_m), max_step_m)

    segment_count = max(1, int(np.ceil(float(fiber_length_m) / dz_target)))
    dz = float(fiber_length_m) / segment_count
    linear_half = np.exp(d * (dz / 2.0))

    field = field0.copy()
    for _ in range(segment_count):
        # Symmetric SSFM step: L(dz/2) -> N(dz) -> L(dz/2)
        aw_step = np.fft.ifft(field)
        aw_step *= linear_half
        field = np.fft.fft(aw_step)

        if effective_nonlinearity != 0.0:
            field *= np.exp(1j * effective_nonlinearity *
                            np.abs(field) ** 2 * dz)

        aw_step = np.fft.ifft(field)
        aw_step *= linear_half
        field = np.fft.fft(aw_step)

    z = np.array([0.0, float(fiber_length_m)], dtype=np.float64)
    at = np.vstack((field0, field))
    aw_in = np.fft.fftshift(np.fft.ifft(field0)) * n * dt
    aw_out = np.fft.fftshift(np.fft.ifft(field)) * n * dt
    aw = np.vstack((aw_in, aw_out))

    return gnlse.Solution(t, omega, w_0, z, at, aw), segment_count


def propagate_wdm_signal_constrained(
    A_t, *, resolution, time_window, wavelength_nm,
    fiber_length_m, nonlinearity, betas, loss_db_per_m,
        z_saves, rtol, atol, max_step_m, max_nonlinear_phase_deg,
        propagation_backend="custom_ssfm"):
    """Forward-compatible wrapper for constrained forward propagation."""
    return propagate_field_constrained(
        A_t,
        resolution=resolution,
        time_window=time_window,
        wavelength_nm=wavelength_nm,
        fiber_length_m=fiber_length_m,
        nonlinearity=nonlinearity,
        betas=betas,
        loss_db_per_m=loss_db_per_m,
        z_saves=z_saves,
        rtol=rtol,
        atol=atol,
        max_step_m=max_step_m,
        max_nonlinear_phase_deg=max_nonlinear_phase_deg,
        propagation_direction=1,
        propagation_backend=propagation_backend,
    )


class GNLSEDispersionModel:
    """Wrapper for GNLSE dispersion model for use with receiver."""

    def __init__(self, loss_db_per_m, betas, fiber_length_m=1.0,
                 wavelength_nm=1550.0):
        self.loss_db_per_m = loss_db_per_m
        self.betas = np.array(betas)
        self.fiber_length_m = fiber_length_m
        self.wavelength_nm = wavelength_nm
        self._model = gnlse.DispersionFiberFromTaylor(
            loss_db_per_m, self.betas)

    def D(self, omega):
        """Get dispersion at angular frequency omega."""
        return self._model.D(omega)
