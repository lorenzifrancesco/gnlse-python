"""
WDM Monte Carlo example with constellation plots and NLI estimation.

This example builds modulated WDM channels in the time domain, propagates them
through the GNLSE, and estimates nonlinear-interference (NLI) noise by
subtracting a linear (gamma=0) run from the nonlinear run.

Configuration is loaded from input/wdm_nli_config.toml via Pydantic.
"""

import math
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from pydantic import BaseModel

import gnlse

try:
    import tomllib as toml
except ImportError:  # Python < 3.11
    import tomli as toml


def rrc_taps(rolloff, span_symbols, sps):
    """Root-raised-cosine filter taps, normalized to unit energy."""
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
            taps[i] = (rolloff / math.sqrt(2)) * (
                (1 + 2 / np.pi) * math.sin(np.pi / (4 * rolloff)) +
                (1 - 2 / np.pi) * math.cos(np.pi / (4 * rolloff))
            )
        else:
            num = (math.sin(np.pi * ti * (1 - rolloff)) +
                   4 * rolloff * ti * math.cos(np.pi * ti * (1 + rolloff)))
            den = np.pi * ti * (1 - (4 * rolloff * ti)**2)
            taps[i] = num / den

    taps /= np.sqrt(np.sum(taps**2))
    return taps


def qam_symbols(order, count, rng):
    """Square QAM constellation with unit average power."""
    m = int(round(math.sqrt(order)))
    if m * m != order:
        raise ValueError("QAM order must be a perfect square")

    levels = np.arange(-(m - 1), m, 2)
    const = (levels[None, :] + 1j * levels[:, None]).reshape(-1)
    const /= np.sqrt(np.mean(np.abs(const)**2))
    return rng.choice(const, size=count)


def upsample_and_filter(symbols, sps, taps):
    up = np.zeros(len(symbols) * sps, dtype=np.complex128)
    up[::sps] = symbols
    shaped = np.convolve(up, taps, mode='same')
    return shaped


def scale_to_power(x, target_power_w):
    power = np.mean(np.abs(x)**2)
    if power <= 0:
        return x
    return x * math.sqrt(target_power_w / power)


def make_wdm_field(t, channel_spacing_thz, n_channels, power_per_channel_w,
                   symbols_per_channel, sps, taps, modulation_order, rng):
    """Return complex field A(t) with modulated WDM channels."""
    offsets = channel_spacing_thz * (np.arange(n_channels) -
                                     (n_channels - 1) / 2.0)

    field = np.zeros_like(t, dtype=np.complex128)

    for f_thz in offsets:
        symbols = qam_symbols(modulation_order, symbols_per_channel, rng)
        baseband = upsample_and_filter(symbols, sps, taps)
        baseband = scale_to_power(baseband, power_per_channel_w)

        field += baseband * np.exp(1j * (2 * np.pi * f_thz * t))

    return field


def make_setup(A_t, *, resolution, time_window, wavelength_nm, fiber_length_m,
               nonlinearity, betas, loss_db_per_m, z_saves, rtol, atol):
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


def estimate_nli_noise(solution_nl, solution_lin, center_thz, bandwidth_thz):
    """Estimate NLI noise power in a band around center_thz."""
    freq = (solution_nl.W - solution_nl.w_0) / (2 * np.pi)  # THz
    aw_nl = solution_nl.AW[-1, :]
    aw_lin = solution_lin.AW[-1, :]
    aw_nli = aw_nl - aw_lin

    mask = np.abs(freq - center_thz) <= bandwidth_thz / 2.0
    df = freq[1] - freq[0]

    noise_power = np.sum(np.abs(aw_nli[mask])**2) * df
    signal_power = np.sum(np.abs(aw_lin[mask])**2) * df
    rx_power = np.sum(np.abs(aw_nl[mask])**2) * df
    snr = np.inf if noise_power == 0 else signal_power / noise_power
    nli_norm = np.inf if rx_power == 0 else noise_power / rx_power
    return noise_power, signal_power, rx_power, snr, nli_norm


def matched_filter_and_sample(At, sps, taps, guard_symbols):
    y = np.convolve(At, taps, mode='same')
    start = guard_symbols * sps
    end = len(y) - guard_symbols * sps
    if end <= start:
        return np.array([], dtype=np.complex128)
    return y[start:end:sps]


def compensate_dispersion(At, t, fiber_length_m, dispersion_model):
    """Apply linear dispersion compensation in the frequency domain."""
    n = len(t)
    dt = t[1] - t[0]
    v = 2 * np.pi * np.arange(-n / 2, n / 2) / (n * dt)
    d = dispersion_model.D(v)
    d = np.fft.fftshift(d)
    d_disp = 1j * np.imag(d)

    aw = np.fft.ifft(At)
    aw_comp = aw * np.exp(-d_disp * fiber_length_m)
    return np.fft.fft(aw_comp)


def print_table(rows):
    headers = ("Parameter", "Value", "Units", "Notes")
    all_rows = [headers] + rows

    widths = [0, 0, 0, 0]
    for row in all_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(row):
        return "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))

    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def to_db(value):
    if value <= 0:
        return float("-inf")
    return 10.0 * math.log10(value)


class ModulationConfig(BaseModel):
    order: int
    symbol_rate_gbd: float
    samples_per_symbol: int
    rrc_rolloff: float
    rrc_span_symbols: int
    n_symbols: int


class WdmConfig(BaseModel):
    n_channels: int
    channel_spacing_thz: float
    power_per_channel_w: float


class FiberConfig(BaseModel):
    wavelength_nm: float
    fiber_length_m: float
    nonlinearity: float
    loss_db_per_m: float
    betas: list[float]


class MonteCarloConfig(BaseModel):
    n_trials: int
    rng_seed: int


class SolverConfig(BaseModel):
    z_saves: int
    rtol: float
    atol: float


class OutputConfig(BaseModel):
    save_dir: str
    save_name: str
    show_plot: bool


class SimulationConfig(BaseModel):
    modulation: ModulationConfig
    wdm: WdmConfig
    fiber: FiberConfig
    monte_carlo: MonteCarloConfig
    solver: SolverConfig
    output: OutputConfig


def load_config(path: Path) -> SimulationConfig:
    with path.open("rb") as fh:
        data = toml.load(fh)

    if hasattr(SimulationConfig, "model_validate"):
        return SimulationConfig.model_validate(data)
    return SimulationConfig.parse_obj(data)


if __name__ == '__main__':
    config_path = Path(__file__).resolve().parents[1] / "input" / "wdm_nli_config.toml"
    config = load_config(config_path)

    mod = config.modulation
    wdm = config.wdm
    fiber = config.fiber
    mc = config.monte_carlo
    solver = config.solver
    output = config.output

    if mod.samples_per_symbol <= 0:
        raise ValueError("samples_per_symbol must be positive")
    if wdm.n_channels % 2 == 0:
        raise ValueError("n_channels must be odd to keep a center channel at 0")

    resolution = mod.n_symbols * mod.samples_per_symbol

    symbol_rate_thz = mod.symbol_rate_gbd * 1e-3
    dt = 1.0 / (symbol_rate_thz * mod.samples_per_symbol)
    time_window = dt * (resolution - 1)

    df = 1.0 / (resolution * dt)
    f_max = 0.5 / dt

    channel_bw_thz = symbol_rate_thz * (1 + mod.rrc_rolloff)
    max_offset = wdm.channel_spacing_thz * (wdm.n_channels - 1) / 2.0
    if max_offset + channel_bw_thz / 2.0 > f_max:
        raise ValueError("Channel plan exceeds Nyquist bandwidth.")

    rng = np.random.default_rng(mc.rng_seed)
    taps = rrc_taps(mod.rrc_rolloff, mod.rrc_span_symbols, mod.samples_per_symbol)

    rows = [
        ("modulation", f"{mod.order}-QAM", "", ""),
        ("symbol_rate", f"{mod.symbol_rate_gbd:g}", "GBd", ""),
        ("samples_per_symbol", mod.samples_per_symbol, "", ""),
        ("rrc_rolloff", f"{mod.rrc_rolloff:g}", "", ""),
        ("rrc_span_symbols", mod.rrc_span_symbols, "symbols", ""),
        ("n_symbols", mod.n_symbols, "", "per channel"),
        ("resolution", resolution, "samples", ""),
        ("time_window", f"{time_window:.6g}", "ps", ""),
        ("dt", f"{dt:.6g}", "ps", ""),
        ("df", f"{df:.6g}", "THz", ""),
        ("f_max", f"{f_max:.6g}", "THz", "Nyquist"),
        ("n_channels", wdm.n_channels, "", ""),
        ("channel_spacing", f"{wdm.channel_spacing_thz:g}", "THz", ""),
        ("channel_bw", f"{channel_bw_thz:g}", "THz", "RRC null-to-null"),
        ("power_per_channel", f"{wdm.power_per_channel_w:g}", "W", ""),
        ("total_power", f"{wdm.power_per_channel_w * wdm.n_channels:g}", "W", ""),
        ("wavelength", f"{fiber.wavelength_nm:g}", "nm", ""),
        ("fiber_length", f"{fiber.fiber_length_m:g}", "m", ""),
        ("gamma", f"{fiber.nonlinearity:g}", "1/W/m", ""),
        ("loss", f"{fiber.loss_db_per_m:g}", "dB/m", ""),
        ("betas", np.array2string(np.array(fiber.betas), precision=6), "ps^n/m", ""),
        ("n_trials", mc.n_trials, "", ""),
        ("rng_seed", mc.rng_seed, "", ""),
        ("save_dir", output.save_dir, "", ""),
        ("save_name", output.save_name, "", ""),
        ("show_plot", output.show_plot, "", "")
    ]

    print_table(rows)

    center_noise_samples = []
    center_snr_samples = []
    center_rx_power_samples = []
    center_nli_norm_samples = []

    full_noise_samples = []
    full_snr_samples = []
    full_rx_power_samples = []
    full_nli_norm_samples = []

    sample_const_lin = None
    sample_const_nl = None

    for trial in range(mc.n_trials):
        t = np.linspace(-time_window / 2, time_window / 2, resolution)
        A_t = make_wdm_field(
            t,
            wdm.channel_spacing_thz,
            wdm.n_channels,
            wdm.power_per_channel_w,
            mod.n_symbols,
            mod.samples_per_symbol,
            taps,
            mod.order,
            rng)

        setup_nl = make_setup(
            A_t,
            resolution=resolution,
            time_window=time_window,
            wavelength_nm=fiber.wavelength_nm,
            fiber_length_m=fiber.fiber_length_m,
            nonlinearity=fiber.nonlinearity,
            betas=np.array(fiber.betas),
            loss_db_per_m=fiber.loss_db_per_m,
            z_saves=solver.z_saves,
            rtol=solver.rtol,
            atol=solver.atol)
        setup_lin = make_setup(
            A_t,
            resolution=resolution,
            time_window=time_window,
            wavelength_nm=fiber.wavelength_nm,
            fiber_length_m=fiber.fiber_length_m,
            nonlinearity=0.0,
            betas=np.array(fiber.betas),
            loss_db_per_m=fiber.loss_db_per_m,
            z_saves=solver.z_saves,
            rtol=solver.rtol,
            atol=solver.atol)

        sol_nl = gnlse.GNLSE(setup_nl).run()
        sol_lin = gnlse.GNLSE(setup_lin).run()

        total_wdm_bw_thz = wdm.channel_spacing_thz * (wdm.n_channels - 1) + channel_bw_thz

        center_noise, center_signal, center_rx, center_snr, center_nli_norm = estimate_nli_noise(
            sol_nl, sol_lin, 0.0, channel_bw_thz)
        full_noise, full_signal, full_rx, full_snr, full_nli_norm = estimate_nli_noise(
            sol_nl, sol_lin, 0.0, total_wdm_bw_thz)

        center_noise_samples.append(center_noise)
        center_snr_samples.append(center_snr)
        center_rx_power_samples.append(center_rx)
        center_nli_norm_samples.append(center_nli_norm)

        full_noise_samples.append(full_noise)
        full_snr_samples.append(full_snr)
        full_rx_power_samples.append(full_rx)
        full_nli_norm_samples.append(full_nli_norm)

        if trial == 0:
            At_lin = sol_lin.At[-1, :]
            At_nl = sol_nl.At[-1, :]

            guard_symbols = mod.rrc_span_symbols
            disp_model = gnlse.DispersionFiberFromTaylor(
                fiber.loss_db_per_m, np.array(fiber.betas))
            At_lin = compensate_dispersion(
                At_lin, t, fiber.fiber_length_m, disp_model)
            At_nl = compensate_dispersion(
                At_nl, t, fiber.fiber_length_m, disp_model)
            sample_const_lin = matched_filter_and_sample(
                At_lin, mod.samples_per_symbol, taps, guard_symbols)
            sample_const_nl = matched_filter_and_sample(
                At_nl, mod.samples_per_symbol, taps, guard_symbols)

    def summarize(label, noise_samples, snr_samples, rx_samples, nli_norm_samples):
        noise_avg = float(np.mean(noise_samples))
        noise_std = float(np.std(noise_samples))
        snr_avg = float(np.mean(snr_samples))
        rx_power_avg = float(np.mean(rx_samples))
        nli_norm_avg = float(np.mean(nli_norm_samples))

        results_rows = [
            ("nli_noise_avg", f"{noise_avg:.6g}", "arb", f"{to_db(noise_avg):.3f} dB"),
            ("nli_noise_std", f"{noise_std:.6g}", "arb", f"{to_db(noise_std):.3f} dB"),
            ("rx_power_avg", f"{rx_power_avg:.6g}", "arb", f"{to_db(rx_power_avg):.3f} dB"),
            ("nli_norm_avg", f"{nli_norm_avg:.6g}", "", f"{to_db(nli_norm_avg):.3f} dB"),
            ("snr_avg", f"{snr_avg:.6g}", "", f"{to_db(snr_avg):.3f} dB")
        ]
        print(f"\nResults ({label})")
        print_table(results_rows)

    summarize("center channel", center_noise_samples, center_snr_samples,
              center_rx_power_samples, center_nli_norm_samples)
    summarize("all channels", full_noise_samples, full_snr_samples,
              full_rx_power_samples, full_nli_norm_samples)

    if sample_const_lin is not None and sample_const_nl is not None:
        phase_correction = np.angle(
            np.sum(sample_const_nl * np.conj(sample_const_lin)))
        sample_const_nl_rot = sample_const_nl * np.exp(-1j * phase_correction)

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        axes[0].scatter(sample_const_lin.real, sample_const_lin.imag,
                        s=8, alpha=0.6)
        axes[0].set_title("Constellation: linear (no NLI)")
        axes[0].set_xlabel("I")
        axes[0].set_ylabel("Q")
        axes[0].grid(True)
        axes[0].set_aspect('equal', 'box')

        axes[1].scatter(sample_const_nl.real, sample_const_nl.imag,
                        s=8, alpha=0.6)
        axes[1].set_title("Constellation: nonlinear (with NLI)")
        axes[1].set_xlabel("I")
        axes[1].set_ylabel("Q")
        axes[1].grid(True)
        axes[1].set_aspect('equal', 'box')

        axes[2].scatter(sample_const_nl_rot.real, sample_const_nl_rot.imag,
                        s=8, alpha=0.6)
        axes[2].set_title("Constellation: NLI + phase rotation")
        axes[2].set_xlabel("I")
        axes[2].set_ylabel("Q")
        axes[2].grid(True)
        axes[2].set_aspect('equal', 'box')

        plt.tight_layout()
        os.makedirs(output.save_dir, exist_ok=True)
        plt.savefig(os.path.join(output.save_dir, output.save_name), dpi=200)
        if output.show_plot:
            plt.show()
