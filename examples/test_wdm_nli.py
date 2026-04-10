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
import sys
import json
import hashlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


def qam_axis_levels(order):
    """Normalized I/Q coordinates of the ideal square-QAM symbol centers."""
    m = int(round(math.sqrt(order)))
    if m * m != order:
        raise ValueError("QAM order must be a perfect square")

    levels = np.arange(-(m - 1), m, 2, dtype=np.float64)
    const = (levels[None, :] + 1j * levels[:, None]).reshape(-1)
    scale = np.sqrt(np.mean(np.abs(const)**2))
    return levels / scale


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
    """Return complex field A(t) and center-channel transmitted symbols."""
    offsets = channel_spacing_thz * (np.arange(n_channels) -
                                     (n_channels - 1) / 2.0)

    field = np.zeros_like(t, dtype=np.complex128)
    center_symbols = None

    for f_thz in offsets:
        symbols = qam_symbols(modulation_order, symbols_per_channel, rng)
        baseband = upsample_and_filter(symbols, sps, taps)
        baseband = scale_to_power(baseband, power_per_channel_w)

        field += baseband * np.exp(1j * (2 * np.pi * f_thz * t))
        if np.isclose(f_thz, 0.0):
            center_symbols = symbols

    if center_symbols is None:
        raise ValueError("No center channel found at 0 THz.")
    return field, center_symbols


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


def estimate_nli_noise_from_aw(aw_nl, aw_lin, freq, center_thz, bandwidth_thz):
    """Estimate NLI noise power in a band around center_thz."""
    aw_nli = aw_nl - aw_lin

    mask = np.abs(freq - center_thz) <= bandwidth_thz / 2.0
    df = freq[1] - freq[0]

    noise_power = np.sum(np.abs(aw_nli[mask])**2) * df
    signal_power = np.sum(np.abs(aw_lin[mask])**2) * df
    rx_power = np.sum(np.abs(aw_nl[mask])**2) * df
    snr = np.inf if noise_power == 0 else signal_power / noise_power
    nli_norm = np.inf if rx_power == 0 else noise_power / rx_power
    return noise_power, signal_power, rx_power, snr, nli_norm


def extract_center_channel(At, dt, symbol_rate_thz, rolloff):
    """Apply a smooth raised-cosine demux filter around the center channel."""
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


def matched_filter_output(At, taps):
    return np.convolve(At, taps, mode='same')


def sample_matched_filter(y, sps, guard_symbols, offset=0):
    start = guard_symbols * sps + offset
    end = len(y) - guard_symbols * sps
    if end <= start:
        return np.array([], dtype=np.complex128)
    return y[start:end:sps]


def trim_guard_symbols(symbols, guard_symbols):
    if guard_symbols <= 0:
        return symbols
    if len(symbols) <= 2 * guard_symbols:
        return np.array([], dtype=np.complex128)
    return symbols[guard_symbols:-guard_symbols]


def estimate_constellation_gain(samples, reference_symbols):
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


def constellation_nmse(samples, reference_symbols):
    if len(samples) == 0 or len(reference_symbols) == 0:
        return float("inf")
    n = min(len(samples), len(reference_symbols))
    ref = reference_symbols[:n]
    sam = samples[:n]
    gain = estimate_constellation_gain(sam, ref)
    equalized = sam / gain
    ref_power = float(np.mean(np.abs(ref)**2))
    if np.isclose(ref_power, 0.0):
        return float("inf")
    return float(np.mean(np.abs(equalized - ref)**2) / ref_power)


def find_best_sampling_offset(y, sps, guard_symbols, reference_symbols):
    best_offset = 0
    best_samples = np.array([], dtype=np.complex128)
    best_nmse = float("inf")

    for offset in range(sps):
        samples = sample_matched_filter(y, sps, guard_symbols, offset=offset)
        nmse = constellation_nmse(samples, reference_symbols)
        if nmse < best_nmse:
            best_offset = offset
            best_samples = samples
            best_nmse = nmse

    return best_samples, best_offset, best_nmse


def train_symbol_equalizer(samples, reference_symbols, n_taps):
    """Least-squares symbol-spaced FIR equalizer."""
    if n_taps % 2 == 0:
        raise ValueError("n_taps must be odd.")

    n = min(len(samples), len(reference_symbols))
    if n < n_taps:
        return np.array([1.0 + 0j], dtype=np.complex128), reference_symbols[:n]

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


def apply_symbol_equalizer(samples, taps):
    n_taps = len(taps)
    half = n_taps // 2
    if len(samples) < n_taps:
        return np.array([], dtype=np.complex128)

    equalized = []
    for i in range(half, len(samples) - half):
        equalized.append(np.dot(samples[i - half:i + half + 1], taps))
    return np.asarray(equalized, dtype=np.complex128)


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


def aw_from_time(At, dt):
    n = len(At)
    return np.fft.fftshift(np.fft.ifft(At)) * n * dt


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


def constellation_density_cmap():
    cmap = LinearSegmentedColormap.from_list(
        "constellation_density",
        ["#f8fafc", "#dbeafe", "#7dd3fc", "#0ea5e9", "#0f766e", "#0f172a"],
    )
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    return cmap


def plot_constellation_triplet(constellations, axis_limit, save_path,
                              symbol_levels,
                              *, hide_axis_text=False, show_plot=False):
    density_bins = 80
    density_cmap = constellation_density_cmap()
    fig = plt.figure(figsize=(14, 5))
    grid = fig.add_gridspec(
        1, 3,
        left=0.04,
        right=0.99,
        bottom=0.10,
        top=0.96,
        wspace=0.0,
    )
    axes = grid.subplots(sharex=True, sharey=True)

    for ax, (title, constellation) in zip(axes, constellations):
        ax.set_facecolor("#f8fafc")
        hist, xedges, yedges = np.histogram2d(
            constellation.real,
            constellation.imag,
            bins=density_bins,
            range=[[-axis_limit, axis_limit], [-axis_limit, axis_limit]],
        )
        hist = np.ma.masked_less_equal(hist.T, 0.0)
        ax.imshow(
            hist,
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap=density_cmap,
            norm=LogNorm(vmin=1, vmax=float(np.max(hist))),
            interpolation="nearest",
            alpha=0.95,
            aspect="auto",
            zorder=1,
        )
        ax.scatter(
            constellation.real, constellation.imag,
            s=4, c="#0f172a", alpha=0.12, linewidths=0, zorder=2)
        # ax.set_title(title)
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
        ax.set_xticks(symbol_levels)
        ax.set_yticks(symbol_levels)
        ax.grid(True, color="0.75", linewidth=0.8)
        ax.set_aspect("equal", adjustable="box")

        if hide_axis_text:
            ax.tick_params(
                axis="both",
                which="both",
                labelbottom=False,
                labelleft=False,
                bottom=False,
                left=False,
            )
        else:
            ax.set_xlabel("I")
            ax.set_ylabel("Q")

    fig.savefig(save_path, dpi=200)
    if show_plot:
        plt.show()
    plt.close(fig)


def to_db(value):
    if value <= 0:
        return float("-inf")
    return 10.0 * math.log10(value)


def model_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def stable_json_dumps(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def compute_spec_hash(data_spec):
    digest = hashlib.sha256(stable_json_dumps(data_spec).encode("utf-8"))
    return digest.hexdigest()[:12]


def build_output_paths(save_dir, save_name, spec_hash):
    base = Path(save_name)
    unique_stem = f"{base.stem}_{spec_hash}"
    return {
        "constellation_plot": Path(save_dir) / f"{unique_stem}{base.suffix}",
        "constellation_plot_no_ticks": Path(save_dir) / f"{unique_stem}_no_ticks{base.suffix}",
        "results_json": Path(save_dir) / f"{unique_stem}_results.json",
    }


def complex_dict_to_array(payload):
    return np.asarray(payload["real"], dtype=np.float64) + 1j * np.asarray(
        payload["imag"], dtype=np.float64)


def results_bundle_is_compatible(bundle, data_spec, spec_hash):
    return (
        bundle.get("spec_hash") == spec_hash and
        bundle.get("data_spec") == data_spec
    )


def load_results_bundle(path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def summarize_from_bundle(bundle):
    return bundle.get("summaries", {})


def print_summary_block(label, summary):
    rows = [
        ("nli_noise_avg", f"{summary['nli_noise_avg']:.6g}", "arb",
         f"{summary['nli_noise_avg_db']:.3f} dB"),
        ("nli_noise_std", f"{summary['nli_noise_std']:.6g}", "arb",
         f"{summary['nli_noise_std_db']:.3f} dB"),
        ("rx_power_avg", f"{summary['rx_power_avg']:.6g}", "arb",
         f"{summary['rx_power_avg_db']:.3f} dB"),
        ("nli_norm_avg", f"{summary['nli_norm_avg']:.6g}", "",
         f"{summary['nli_norm_avg_db']:.3f} dB"),
        ("snr_avg", f"{summary['snr_avg']:.6g}", "",
         f"{summary['snr_avg_db']:.3f} dB")
    ]
    print(f"\nResults ({label})")
    print_table(rows)


def maybe_regenerate_plots_from_bundle(bundle, plot_outputs, show_plot):
    plot_payload = bundle.get("plot_payload")
    if plot_payload is None:
        return

    constellations_payload = plot_payload.get("constellations")
    if not constellations_payload:
        return

    constellations = [
        (title, complex_dict_to_array(values))
        for title, values in constellations_payload.items()
    ]
    axis_limit = float(plot_payload["axis_limit"])
    symbol_levels = np.asarray(plot_payload["symbol_levels"], dtype=np.float64)
    plot_constellation_triplet(
        constellations,
        axis_limit,
        plot_outputs["constellation_plot"],
        symbol_levels,
        show_plot=show_plot,
    )
    plot_constellation_triplet(
        constellations,
        axis_limit,
        plot_outputs["constellation_plot_no_ticks"],
        symbol_levels,
        hide_axis_text=True,
    )


def complex_array_to_dict(values):
    arr = np.asarray(values, dtype=np.complex128)
    return {
        "real": arr.real.tolist(),
        "imag": arr.imag.tolist(),
    }


def summarize_results(noise_samples, snr_samples, rx_samples, nli_norm_samples):
    noise_avg = float(np.mean(noise_samples))
    noise_std = float(np.std(noise_samples))
    snr_avg = float(np.mean(snr_samples))
    rx_power_avg = float(np.mean(rx_samples))
    nli_norm_avg = float(np.mean(nli_norm_samples))

    return {
        "nli_noise_avg": noise_avg,
        "nli_noise_avg_db": to_db(noise_avg),
        "nli_noise_std": noise_std,
        "nli_noise_std_db": to_db(noise_std),
        "rx_power_avg": rx_power_avg,
        "rx_power_avg_db": to_db(rx_power_avg),
        "nli_norm_avg": nli_norm_avg,
        "nli_norm_avg_db": to_db(nli_norm_avg),
        "snr_avg": snr_avg,
        "snr_avg_db": to_db(snr_avg),
    }


def save_results_bundle(save_path, *, config, config_path, data_spec, spec_hash,
                        derived, summaries, trial_results, plot_outputs,
                        plot_payload):
    payload = {
        "config_path": str(config_path),
        "config": model_to_dict(config),
        "data_spec": data_spec,
        "spec_hash": spec_hash,
        "derived_parameters": derived,
        "summaries": summaries,
        "trial_results": trial_results,
        "plot_outputs": {key: str(value) for key, value in plot_outputs.items()},
        "plot_payload": plot_payload,
    }
    with save_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


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


class CacheConfig(BaseModel):
    mode: str = "recompute"


class SimulationConfig(BaseModel):
    modulation: ModulationConfig
    wdm: WdmConfig
    fiber: FiberConfig
    monte_carlo: MonteCarloConfig
    solver: SolverConfig
    output: OutputConfig
    cache: CacheConfig = CacheConfig()


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
    cache = config.cache

    if cache.mode not in {"recompute", "load"}:
        raise ValueError("cache.mode must be 'recompute' or 'load'")

    if mod.samples_per_symbol <= 0:
        raise ValueError("samples_per_symbol must be positive")
    if wdm.n_channels % 2 == 0:
        raise ValueError("n_channels must be odd to keep a center channel at 0")

    equalizer_taps = 11
    data_spec = {
        "modulation": model_to_dict(mod),
        "wdm": model_to_dict(wdm),
        "fiber": model_to_dict(fiber),
        "monte_carlo": model_to_dict(mc),
        "solver": model_to_dict(solver),
        "receiver": {
            "architecture": "coherent_sp_cdc_demux_mf_timing_lms_ls_v1",
            "equalizer_taps": equalizer_taps,
        },
    }
    spec_hash = compute_spec_hash(data_spec)
    output_paths = build_output_paths(output.save_dir, output.save_name, spec_hash)

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

    derived_parameters = {
        "resolution": resolution,
        "symbol_rate_thz": symbol_rate_thz,
        "dt_ps": dt,
        "time_window_ps": time_window,
        "df_thz": df,
        "f_max_thz": f_max,
        "channel_bw_thz": channel_bw_thz,
        "max_offset_thz": max_offset,
        "spec_hash": spec_hash,
    }
    derived_parameters["equalizer_taps"] = equalizer_taps

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
        ("cache_mode", cache.mode, "", ""),
        ("spec_hash", spec_hash, "", ""),
        ("save_dir", output.save_dir, "", ""),
        ("save_name", output.save_name, "", ""),
        ("data_save_name", output_paths["results_json"].name, "", ""),
        ("show_plot", output.show_plot, "", "")
    ]

    print_table(rows)

    os.makedirs(output.save_dir, exist_ok=True)

    if cache.mode == "load" and output_paths["results_json"].exists():
        bundle = load_results_bundle(output_paths["results_json"])
        if results_bundle_is_compatible(bundle, data_spec, spec_hash):
            print(f"\nLoaded cached results: {output_paths['results_json']}")
            maybe_regenerate_plots_from_bundle(bundle, output_paths, output.show_plot)
            for label, summary in summarize_from_bundle(bundle).items():
                print_summary_block(label, summary)
            sys.exit(0)
        print("\nCached data incompatible with current TOML. Recomputing.")
    elif cache.mode == "load":
        print("\nCached data not found. Recomputing.")

    rng = np.random.default_rng(mc.rng_seed)
    taps = rrc_taps(mod.rrc_rolloff, mod.rrc_span_symbols, mod.samples_per_symbol)

    center_noise_samples = []
    center_snr_samples = []
    center_rx_power_samples = []
    center_nli_norm_samples = []

    full_noise_samples = []
    full_snr_samples = []
    full_rx_power_samples = []
    full_nli_norm_samples = []

    center_noise_eq_samples = []
    center_snr_eq_samples = []
    center_rx_power_eq_samples = []
    center_nli_norm_eq_samples = []

    full_noise_eq_samples = []
    full_snr_eq_samples = []
    full_rx_power_eq_samples = []
    full_nli_norm_eq_samples = []

    plot_const_lin = None
    plot_const_nl = None
    plot_const_nl_rot = None
    plot_symbols_ref = None
    plot_sampling_offset = None
    plot_linear_nmse = None
    plot_equalizer_taps = None
    summary_results = {}

    for trial in range(mc.n_trials):
        t = np.linspace(-time_window / 2, time_window / 2, resolution)
        A_t, center_tx_symbols = make_wdm_field(
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
        freq = (sol_nl.W - sol_nl.w_0) / (2 * np.pi)

        center_noise, center_signal, center_rx, center_snr, center_nli_norm = (
            estimate_nli_noise_from_aw(sol_nl.AW[-1, :], sol_lin.AW[-1, :],
                                       freq, 0.0, channel_bw_thz))
        full_noise, full_signal, full_rx, full_snr, full_nli_norm = (
            estimate_nli_noise_from_aw(sol_nl.AW[-1, :], sol_lin.AW[-1, :],
                                       freq, 0.0, total_wdm_bw_thz))

        center_noise_samples.append(center_noise)
        center_snr_samples.append(center_snr)
        center_rx_power_samples.append(center_rx)
        center_nli_norm_samples.append(center_nli_norm)

        full_noise_samples.append(full_noise)
        full_snr_samples.append(full_snr)
        full_rx_power_samples.append(full_rx)
        full_nli_norm_samples.append(full_nli_norm)

        At_lin = sol_lin.At[-1, :]
        At_nl = sol_nl.At[-1, :]

        guard_symbols = mod.rrc_span_symbols
        center_tx_symbols = trim_guard_symbols(center_tx_symbols, guard_symbols)
        disp_model = gnlse.DispersionFiberFromTaylor(
            fiber.loss_db_per_m, np.array(fiber.betas))
        At_lin_comp = compensate_dispersion(
            At_lin, t, fiber.fiber_length_m, disp_model)
        At_nl_comp = compensate_dispersion(
            At_nl, t, fiber.fiber_length_m, disp_model)
        At_lin_center = extract_center_channel(
            At_lin_comp, dt, symbol_rate_thz, mod.rrc_rolloff)
        At_nl_center = extract_center_channel(
            At_nl_comp, dt, symbol_rate_thz, mod.rrc_rolloff)

        y_lin = matched_filter_output(At_lin_center, taps)
        y_nl = matched_filter_output(At_nl_center, taps)
        sample_lin_mf, sampling_offset, _ = find_best_sampling_offset(
            y_lin, mod.samples_per_symbol, guard_symbols, center_tx_symbols)
        sample_nl_mf = sample_matched_filter(
            y_nl, mod.samples_per_symbol, guard_symbols, offset=sampling_offset)
        eq_taps, eq_ref_symbols = train_symbol_equalizer(
            sample_lin_mf, center_tx_symbols, equalizer_taps)
        sample_const_lin = apply_symbol_equalizer(sample_lin_mf, eq_taps)
        sample_const_nl = apply_symbol_equalizer(sample_nl_mf, eq_taps)
        linear_nmse = constellation_nmse(sample_const_lin, eq_ref_symbols)

        if len(sample_const_lin) > 0 and len(sample_const_nl) > 0:
            phase_correction = np.angle(
                np.sum(sample_const_nl * np.conj(sample_const_lin)))
        else:
            phase_correction = 0.0

        At_nl_eq = At_nl_comp * np.exp(-1j * phase_correction)
        aw_lin_eq = aw_from_time(At_lin_comp, dt)
        aw_nl_eq = aw_from_time(At_nl_eq, dt)

        center_noise_eq, center_signal_eq, center_rx_eq, center_snr_eq, center_nli_norm_eq = (
            estimate_nli_noise_from_aw(aw_nl_eq, aw_lin_eq, freq, 0.0, channel_bw_thz))
        full_noise_eq, full_signal_eq, full_rx_eq, full_snr_eq, full_nli_norm_eq = (
            estimate_nli_noise_from_aw(aw_nl_eq, aw_lin_eq, freq, 0.0, total_wdm_bw_thz))

        center_noise_eq_samples.append(center_noise_eq)
        center_snr_eq_samples.append(center_snr_eq)
        center_rx_power_eq_samples.append(center_rx_eq)
        center_nli_norm_eq_samples.append(center_nli_norm_eq)

        full_noise_eq_samples.append(full_noise_eq)
        full_snr_eq_samples.append(full_snr_eq)
        full_rx_power_eq_samples.append(full_rx_eq)
        full_nli_norm_eq_samples.append(full_nli_norm_eq)

        if trial == 0:
            plot_const_lin = sample_const_lin
            plot_const_nl = sample_const_nl
            plot_const_nl_rot = sample_const_nl * np.exp(-1j * phase_correction)
            plot_symbols_ref = eq_ref_symbols
            plot_sampling_offset = sampling_offset
            plot_linear_nmse = linear_nmse
            plot_equalizer_taps = eq_taps

    def summarize(label, noise_samples, snr_samples, rx_samples, nli_norm_samples):
        summary = summarize_results(
            noise_samples, snr_samples, rx_samples, nli_norm_samples)

        results_rows = [
            ("nli_noise_avg", f"{summary['nli_noise_avg']:.6g}", "arb",
             f"{summary['nli_noise_avg_db']:.3f} dB"),
            ("nli_noise_std", f"{summary['nli_noise_std']:.6g}", "arb",
             f"{summary['nli_noise_std_db']:.3f} dB"),
            ("rx_power_avg", f"{summary['rx_power_avg']:.6g}", "arb",
             f"{summary['rx_power_avg_db']:.3f} dB"),
            ("nli_norm_avg", f"{summary['nli_norm_avg']:.6g}", "",
             f"{summary['nli_norm_avg_db']:.3f} dB"),
            ("snr_avg", f"{summary['snr_avg']:.6g}", "",
             f"{summary['snr_avg_db']:.3f} dB")
        ]
        print(f"\nResults ({label})")
        print_table(results_rows)
        summary_results[label] = summary

    summarize("center channel (raw)", center_noise_samples, center_snr_samples,
              center_rx_power_samples, center_nli_norm_samples)
    summarize("all channels (raw)", full_noise_samples, full_snr_samples,
              full_rx_power_samples, full_nli_norm_samples)
    summarize("center channel (CD+SPM comp)", center_noise_eq_samples,
              center_snr_eq_samples, center_rx_power_eq_samples,
              center_nli_norm_eq_samples)
    summarize("all channels (CD+SPM comp)", full_noise_eq_samples,
              full_snr_eq_samples, full_rx_power_eq_samples,
              full_nli_norm_eq_samples)

    save_path = output_paths["constellation_plot"]
    clean_save_path = output_paths["constellation_plot_no_ticks"]
    results_save_path = output_paths["results_json"]
    trial_results = {
        "center_channel_raw": {
            "noise": [float(x) for x in center_noise_samples],
            "snr": [float(x) for x in center_snr_samples],
            "rx_power": [float(x) for x in center_rx_power_samples],
            "nli_norm": [float(x) for x in center_nli_norm_samples],
        },
        "all_channels_raw": {
            "noise": [float(x) for x in full_noise_samples],
            "snr": [float(x) for x in full_snr_samples],
            "rx_power": [float(x) for x in full_rx_power_samples],
            "nli_norm": [float(x) for x in full_nli_norm_samples],
        },
        "center_channel_cd_spm_comp": {
            "noise": [float(x) for x in center_noise_eq_samples],
            "snr": [float(x) for x in center_snr_eq_samples],
            "rx_power": [float(x) for x in center_rx_power_eq_samples],
            "nli_norm": [float(x) for x in center_nli_norm_eq_samples],
        },
        "all_channels_cd_spm_comp": {
            "noise": [float(x) for x in full_noise_eq_samples],
            "snr": [float(x) for x in full_snr_eq_samples],
            "rx_power": [float(x) for x in full_rx_power_eq_samples],
            "nli_norm": [float(x) for x in full_nli_norm_eq_samples],
        },
    }
    plot_outputs = {
        "constellation_plot": save_path,
        "constellation_plot_no_ticks": clean_save_path,
        "results_json": results_save_path,
    }
    plot_payload = None

    if (plot_const_lin is not None and plot_const_nl is not None and
            plot_const_nl_rot is not None and plot_symbols_ref is not None):
        plot_gain = estimate_constellation_gain(plot_const_lin, plot_symbols_ref)
        constellations = [
            ("Constellation: linear (no NLI)", plot_const_lin / plot_gain),
            ("Constellation: nonlinear (with NLI)", plot_const_nl / plot_gain),
            ("Constellation: NLI + phase rotation", plot_const_nl_rot / plot_gain),
        ]
        stacked = np.concatenate([constellation for _, constellation in constellations])
        max_extent = float(np.max(np.abs(np.concatenate((stacked.real, stacked.imag)))))
        axis_limit = 1.05 * max_extent if max_extent > 0 else 1.0
        symbol_levels = qam_axis_levels(mod.order)

        plot_constellation_triplet(
            constellations, axis_limit, save_path, symbol_levels,
            show_plot=output.show_plot)
        plot_constellation_triplet(
            constellations, axis_limit, clean_save_path, symbol_levels,
            hide_axis_text=True)

        plot_payload = {
            "axis_limit": axis_limit,
            "density_bins": 80,
            "symbol_levels": [float(x) for x in symbol_levels],
            "plot_gain": {
                "real": float(np.real(plot_gain)),
                "imag": float(np.imag(plot_gain)),
            },
            "sampling_offset": int(plot_sampling_offset),
            "linear_nmse": float(plot_linear_nmse),
            "equalizer_taps": complex_array_to_dict(plot_equalizer_taps),
            "center_tx_symbols": complex_array_to_dict(plot_symbols_ref),
            "constellations": {
                title: complex_array_to_dict(constellation)
                for title, constellation in constellations
            },
        }
    save_results_bundle(
        results_save_path,
        config=config,
        config_path=config_path,
        data_spec=data_spec,
        spec_hash=spec_hash,
        derived=derived_parameters,
        summaries=summary_results,
        trial_results=trial_results,
        plot_outputs=plot_outputs,
        plot_payload=plot_payload,
    )
    print(f"\nSaved results bundle: {results_save_path}")
