"""
Matched receiver implementation following Dar et al. 2015.

The matched receiver procedure:
1. Isolate the channel of interest with a matched optical filter
2. Ideally back-propagate to eliminate chromatic dispersion and SPM effects
3. Apply matched filter for symbol detection
4. Extract and equalize constellation symbols

Reference:
    Dar et al. 2015: "Nonlinear interference noise accumulation in long-haul 
    optical systems with coherent detection"
"""

import numpy as np
from dataclasses import dataclass

try:
    from .signal_generation import propagate_field_constrained
except ImportError:
    from signal_generation import propagate_field_constrained


@dataclass
class MatchedFilterConfig:
    """Configuration for the matched receiver."""
    sps: int  # samples per symbol
    symbol_rate_thz: float
    rrc_rolloff: float
    rrc_span_symbols: int
    guard_symbols: int = 12
    equalizer_taps: int = 11
    fiber_nonlinearity: float = 0.0  # 1/W/m
    ssfm_max_nonlinear_phase_deg: float = 0.02
    ssfm_max_step_m: float = 1000.0
    propagation_backend: str = "custom_ssfm"
    receiver_processing: str = "matched"  # matched | rect_fd


@dataclass
class ReceivedSignals:
    """Container for received time-domain signals."""
    At_linear: np.ndarray
    At_nonlinear: np.ndarray
    time_vector: np.ndarray
    frequency_vector: np.ndarray


def extract_center_channel_matched(At, dt, symbol_rate_thz, rolloff):
    """
    Step 1: Isolate the center channel with a matched optical filter.
    
    Applies a smooth raised-cosine demux filter around the center channel
    (0 THz) to extract the signal of interest.
    
    Args:
        At: Time-domain signal
        dt: Time step
        symbol_rate_thz: Symbol rate in THz
        rolloff: RRC rolloff factor (0 < rolloff <= 1)
        
    Returns:
        Filtered signal in time domain
    """
    n = len(At)
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=dt))
    aw = np.fft.fftshift(np.fft.ifft(At))
    abs_freq = np.abs(freq)

    if rolloff <= 0:
        response = (abs_freq <= symbol_rate_thz / 2.0).astype(np.float64)
    else:
        passband = 0.5 * symbol_rate_thz * (1.0 - rolloff)
        stopband = 0.5 * symbol_rate_thz * (1.0 + rolloff)
        response = np.zeros_like(abs_freq)
        response[abs_freq <= passband] = 1.0
        transition = (abs_freq > passband) & (abs_freq < stopband)
        response[transition] = 0.5 * (
            1.0 + np.cos(
                np.pi * (abs_freq[transition] - passband) / (stopband - passband)
            )
        )

    aw_center = aw * response
    return np.fft.fft(np.fft.ifftshift(aw_center))


def backpropagate_channel_ssfm(At, t, fiber_length_m, dispersion_model,
                               nonlinearity, max_nonlinear_phase_deg,
                               max_step_m, propagation_backend="custom_ssfm"):
    """
    Back-propagate a single channel with symmetric SSFM.

    This applies inverse linear and nonlinear operators over distance L to
    approximate ideal digital back-propagation of CD+SPM for the channel
    of interest only.

    Args:
        At: Time-domain channel field at fiber output
        t: Time vector
        fiber_length_m: Fiber length in meters
        dispersion_model: Dispersion model with D(omega) method
        nonlinearity: Fiber nonlinear coefficient gamma [1/W/m]
        max_nonlinear_phase_deg: Max nonlinear phase rotation per step [deg]
        max_step_m: Absolute upper bound on SSFM step [m]

    Returns:
        Tuple of (backpropagated_time_field, n_steps)
    """
    time_window = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    loss_db_per_m = getattr(dispersion_model, "loss_db_per_m", 0.0)
    betas = np.asarray(getattr(dispersion_model, "betas"), dtype=np.float64)
    wavelength_nm = float(getattr(dispersion_model, "wavelength_nm", 1550.0))

    sol_bp, n_steps = propagate_field_constrained(
        np.asarray(At, dtype=np.complex128),
        resolution=len(t),
        time_window=time_window,
        wavelength_nm=wavelength_nm,
        fiber_length_m=fiber_length_m,
        nonlinearity=nonlinearity,
        betas=betas,
        loss_db_per_m=loss_db_per_m,
        z_saves=2,
        rtol=1e-6,
        atol=1e-8,
        max_step_m=max_step_m,
        max_nonlinear_phase_deg=max_nonlinear_phase_deg,
        propagation_direction=-1,
        propagation_backend=propagation_backend,
    )
    return np.asarray(sol_bp.At[-1, :], dtype=np.complex128), n_steps


def matched_filter_detection(At_center, rrc_taps):
    """
    Step 3: Apply matched filter for symbol detection.
    
    Convolve the center channel with the RRC filter taps to prepare
    for symbol sampling.
    
    Args:
        At_center: Filtered center channel signal
        rrc_taps: Root-raised-cosine filter taps
        
    Returns:
        Matched-filtered output ready for symbol sampling
    """
    return np.convolve(At_center, rrc_taps, mode='same')


def find_optimal_sampling_offset(y_mf, sps, guard_symbols, reference_symbols):
    """
    Find the optimal sampling phase by minimizing NMSE against reference.
    
    Args:
        y_mf: Matched filter output
        sps: Samples per symbol
        guard_symbols: Number of guard symbols to skip
        reference_symbols: Reference constellation symbols
        
    Returns:
        Tuple of (sampled_symbols, best_offset, best_nmse)
    """
    best_offset = 0
    best_samples = np.array([], dtype=np.complex128)
    best_nmse = float("inf")

    for offset in range(sps):
        samples = sample_at_phase(y_mf, sps, guard_symbols, offset=offset)
        nmse = compute_constellation_nmse(samples, reference_symbols)
        if nmse < best_nmse:
            best_nmse = nmse
            best_offset = offset
            best_samples = samples

    return best_samples, best_offset, best_nmse


def sample_at_phase(y, sps, guard_symbols, offset=0):
    """
    Sample the signal at a specific phase offset.
    
    Args:
        y: Input signal
        sps: Samples per symbol
        guard_symbols: Number of guard symbols to skip
        offset: Phase offset in samples
        
    Returns:
        Sampled symbols
    """
    start = guard_symbols * sps + offset
    end = len(y) - guard_symbols * sps
    if end <= start:
        return np.array([], dtype=np.complex128)
    return y[start:end:sps]


def train_ls_equalizer(samples, reference_symbols, n_taps):
    """
    Train a least-squares symbol-spaced FIR equalizer.
    FIXME this is exotic
    
    Args:
        samples: Input symbol samples
        reference_symbols: Reference (transmitted) symbols
        n_taps: Number of equalizer taps (must be odd)
        
    Returns:
        Tuple of (equalizer_taps, aligned_reference_symbols) or (None, None)
    """
    if n_taps % 2 == 0:
        raise ValueError("n_taps must be odd.")

    n = min(len(samples), len(reference_symbols))
    if n < n_taps:
        return None, None

    half = n_taps // 2
    x = samples[:n]
    ref = reference_symbols[:n]
    rows = []
    for i in range(half, n - half):
        rows.append(x[i - half:i + half + 1])
    x_mat = np.asarray(rows, dtype=np.complex128)
    ref_aligned = ref[half:n - half]
    taps, _, _, _ = np.linalg.lstsq(x_mat, ref_aligned, rcond=None)
    return taps.astype(np.complex128), ref_aligned


def apply_equalizer(samples, taps):
    """
    Apply equalizer taps to the input samples.
    
    Args:
        samples: Input symbols
        taps: Equalizer taps
        
    Returns:
        Equalized symbols
    """
    n_taps = len(taps)
    half = n_taps // 2
    if len(samples) < n_taps:
        return np.array([], dtype=np.complex128)

    equalized = []
    for i in range(half, len(samples) - half):
        equalized.append(np.dot(samples[i - half:i + half + 1], taps))
    return np.asarray(equalized, dtype=np.complex128)


def estimate_phase_rotation(symbols_eq_linear, symbols_eq_nonlinear):
    """
    Estimate and compensate phase rotation between linear and nonlinear cases.
    
    Args:
        symbols_eq_linear: Equalized symbols from linear propagation
        symbols_eq_nonlinear: Equalized symbols from nonlinear propagation
        
    Returns:
        Phase rotation angle in radians
    """
    if len(symbols_eq_linear) == 0 or len(symbols_eq_nonlinear) == 0:
        return 0.0
    
    return np.angle(
        np.sum(symbols_eq_nonlinear * np.conj(symbols_eq_linear))
    )


def compute_constellation_nmse(samples, reference_symbols):
    """
    Compute normalized mean squared error of constellation.
    
    Args:
        samples: Received constellation symbols
        reference_symbols: Reference (transmitted) symbols
        
    Returns:
        NMSE value (float('inf') if invalid input)
    """
    if len(samples) == 0 or len(reference_symbols) == 0:
        return float("inf")
    
    n = min(len(samples), len(reference_symbols))
    ref = reference_symbols[:n]
    sam = samples[:n]
    
    gain = estimate_channel_gain(sam, ref)
    equalized = sam / gain if not np.isclose(abs(gain), 0.0) else sam
    
    ref_power = float(np.mean(np.abs(ref)**2))
    if np.isclose(ref_power, 0.0):
        return float("inf")
    
    return float(np.mean(np.abs(equalized - ref)**2) / ref_power)


def estimate_channel_gain(samples, reference_symbols):
    """
    Estimate MMSE channel gain.
    
    Args:
        samples: Received symbols
        reference_symbols: Reference symbols
        
    Returns:
        Complex gain estimate
    """
    if len(samples) == 0 or len(reference_symbols) == 0:
        return 1.0 + 0j
    
    n = min(len(samples), len(reference_symbols))
    ref = reference_symbols[:n]
    sam = samples[:n]
    denom = np.vdot(ref, ref)
    
    if np.isclose(denom, 0.0):
        return 1.0 + 0j
    
    gain = np.vdot(ref, sam) / denom
    if np.isclose(abs(gain), 0.0):
        return 1.0 + 0j
    
    return gain


def trim_guard_symbols(symbols, guard_symbols):
    """
    Remove guard symbols from beginning and end of symbol sequence.
    
    Args:
        symbols: Input symbols
        guard_symbols: Number of symbols to remove from each end
        
    Returns:
        Trimmed symbols
    """
    if guard_symbols <= 0:
        return symbols
    if len(symbols) <= 2 * guard_symbols:
        return np.array([], dtype=np.complex128)
    return symbols[guard_symbols:-guard_symbols]


def matched_receiver_procedure(received: ReceivedSignals, 
                              config: MatchedFilterConfig,
                              dispersion_model,
                              rrc_taps,
                              reference_tx_symbols) -> dict:
    """
    Complete matched receiver procedure following Dar et al. 2015.
    
    This is the main entry point implementing:
    1. Channel isolation with matched optical filter
    2. Ideal back-propagation for CD and SPM compensation
    3. Matched filter detection and symbol sampling
    4. LS equalizer training and application
    5. Phase rotation compensation
    
    Args:
        received: ReceivedSignals containing At_linear and At_nonlinear
        config: MatchedFilterConfig with receiver parameters
        dispersion_model: Dispersion model for fiber (has D(omega) method)
        rrc_taps: Root-raised-cosine filter taps
        reference_tx_symbols: Reference transmitted symbols (for training)
        
    Returns:
        Dictionary containing:
            - "symbols_linear_eq": Equalized symbols from linear case
            - "symbols_nonlinear_eq": Equalized symbols from nonlinear case
            - "constellation_gain": Channel gain for constellation rotation
            - "phase_rotation": Phase rotation angle
            - "equalizer_taps": Trained equalizer taps
            - "sampling_offset": Optimal sampling phase
            - "linear_nmse": NMSE of linear case
    """
    t = received.time_vector
    dt = t[1] - t[0]
    guard_symbols = config.guard_symbols
    if config.receiver_processing not in {"matched", "rect_fd"}:
        raise ValueError("receiver_processing must be 'matched' or 'rect_fd'")
    if guard_symbols < 0:
        raise ValueError("guard_symbols must be non-negative")
    
    # Trim guard symbols from transmitted reference
    ref_symbols_trimmed = trim_guard_symbols(reference_tx_symbols, guard_symbols)
    
    # === LINEAR CASE ===
    # Step 1: Extract center channel with matched optical filter
    At_lin_center = extract_center_channel_matched(
        received.At_linear, dt, config.symbol_rate_thz, config.rrc_rolloff
    )
    
    # Step 2: Back-propagate to compensate fiber effects
    At_lin_comp, lin_backprop_steps = backpropagate_channel_ssfm(
        At_lin_center,
        t,
        dispersion_model.fiber_length_m,
        dispersion_model,
        nonlinearity=0.0,
        max_nonlinear_phase_deg=config.ssfm_max_nonlinear_phase_deg,
        max_step_m=config.ssfm_max_step_m,
        propagation_backend=config.propagation_backend,
    )
    
    if config.receiver_processing == "matched":
        # Step 3: Matched filter detection
        y_lin_mf = matched_filter_detection(At_lin_comp, rrc_taps)

        # Step 3b: Find optimal sampling phase
        samples_lin, sampling_offset, linear_nmse = find_optimal_sampling_offset(
            y_lin_mf, config.sps, guard_symbols, ref_symbols_trimmed
        )

        # Step 4: Train LS equalizer on linear case
        eq_taps, eq_ref_symbols = train_ls_equalizer(
            samples_lin, ref_symbols_trimmed, config.equalizer_taps
        )

        if eq_taps is None:
            raise RuntimeError("Failed to train equalizer")

        # Apply equalizer
        symbols_lin_eq = apply_equalizer(samples_lin, eq_taps)
    else:
        # Nyquist rectangular processing: no matched-filter taps, no LS equalizer.
        samples_lin, sampling_offset, linear_nmse = find_optimal_sampling_offset(
            At_lin_comp, config.sps, guard_symbols, ref_symbols_trimmed
        )
        lin_gain = estimate_channel_gain(samples_lin, ref_symbols_trimmed)
        symbols_lin_eq = samples_lin / lin_gain if not np.isclose(abs(lin_gain), 0.0) else samples_lin
        n_lin = min(len(symbols_lin_eq), len(ref_symbols_trimmed))
        symbols_lin_eq = symbols_lin_eq[:n_lin]
        eq_ref_symbols = ref_symbols_trimmed[:n_lin]
        eq_taps = None
    
    # === NONLINEAR CASE ===
    # Step 1: Extract center channel
    At_nl_center = extract_center_channel_matched(
        received.At_nonlinear, dt, config.symbol_rate_thz, config.rrc_rolloff
    )
    
    # Step 2: Back-propagate with same parameters
    At_nl_comp, nl_backprop_steps = backpropagate_channel_ssfm(
        At_nl_center,
        t,
        dispersion_model.fiber_length_m,
        dispersion_model,
        nonlinearity=config.fiber_nonlinearity,
        max_nonlinear_phase_deg=config.ssfm_max_nonlinear_phase_deg,
        max_step_m=config.ssfm_max_step_m,
        propagation_backend=config.propagation_backend,
    )
    
    if config.receiver_processing == "matched":
        # Step 3: Matched filter detection
        y_nl_mf = matched_filter_detection(At_nl_comp, rrc_taps)

        # Step 3b: Sample at same phase as linear case
        samples_nl = sample_at_phase(
            y_nl_mf, config.sps, guard_symbols, offset=sampling_offset
        )

        # Step 4: Apply same equalizer
        symbols_nl_eq = apply_equalizer(samples_nl, eq_taps)
    else:
        samples_nl = sample_at_phase(
            At_nl_comp, config.sps, guard_symbols, offset=sampling_offset
        )
        lin_gain = estimate_channel_gain(samples_lin, ref_symbols_trimmed)
        symbols_nl_eq = samples_nl / lin_gain if not np.isclose(abs(lin_gain), 0.0) else samples_nl
        n_nl = min(len(symbols_nl_eq), len(eq_ref_symbols), len(symbols_lin_eq))
        symbols_nl_eq = symbols_nl_eq[:n_nl]
        symbols_lin_eq = symbols_lin_eq[:n_nl]
        eq_ref_symbols = eq_ref_symbols[:n_nl]
    
    # Step 5: Estimate and compensate phase rotation
    phase_rotation = estimate_phase_rotation(symbols_lin_eq, symbols_nl_eq)
    
    # Compute channel gain for constellation rotation visualization
    channel_gain = estimate_channel_gain(symbols_lin_eq, eq_ref_symbols)
    
    return {
        "symbols_linear_eq": symbols_lin_eq,
        "symbols_nonlinear_eq": symbols_nl_eq,
        "symbols_nonlinear_phase_comp": symbols_nl_eq * np.exp(-1j * phase_rotation),
        "constellation_gain": channel_gain,
        "phase_rotation": phase_rotation,
        "equalizer_taps": eq_taps,
        "sampling_offset": sampling_offset,
        "linear_nmse": linear_nmse,
        "reference_symbols": eq_ref_symbols,
        "linear_backprop_steps": lin_backprop_steps,
        "nonlinear_backprop_steps": nl_backprop_steps,
    }
