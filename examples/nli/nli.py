"""
WDM Monte Carlo example with constellation plots and NLI estimation.

This is the refactored version using the matched receiver procedure from
Dar et al. 2015, with modular design separating:
- Signal generation (signal_generation.py)
- Matched receiver processing (receiver_matched.py)
- NLI estimation (nli_estimation.py)

This script implements the structured matched-receiver procedure.

Configuration is loaded from input/wdm_nli_config.toml via Pydantic.
"""

import json
import os
import sys
import hashlib
import importlib
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import new modules (support both `python -m examples.nli...` and direct execution)
try:
    from .signal_generation import (
        rrc_taps, qam_symbols, qam_axis_levels, make_wdm_field,
        GNLSEDispersionModel, propagate_wdm_signal_constrained
    )
    from .receiver_matched import (
        MatchedFilterConfig, ReceivedSignals, matched_receiver_procedure
    )
    from .nli_estimation import (
        estimate_nli_noise_from_spectrum,
        estimate_nli_noise_from_constellation,
        summarize_nli_results,
    )
except ImportError:
    from signal_generation import (
        rrc_taps, qam_symbols, qam_axis_levels, make_wdm_field,
        GNLSEDispersionModel, propagate_wdm_signal_constrained
    )
    from receiver_matched import (
        MatchedFilterConfig, ReceivedSignals, matched_receiver_procedure
    )
    from nli_estimation import (
        estimate_nli_noise_from_spectrum,
        estimate_nli_noise_from_constellation,
        summarize_nli_results,
    )

try:
    from tomllib import load as _toml_load
    _toml_binary_loader = True
except ImportError:  # Python < 3.11
    try:
        _toml_load = importlib.import_module("tomli").load
        _toml_binary_loader = True
    except ModuleNotFoundError:
        _toml_load = importlib.import_module("toml").load
        _toml_binary_loader = False


def toml_load(path: Path):
    """Load TOML using available parser (tomllib/tomli/toml)."""
    if _toml_binary_loader:
        with path.open("rb") as fh:
            return _toml_load(fh)
    with path.open("r", encoding="utf-8") as fh:
        return _toml_load(fh)


# ============================================================================
# VISUALIZATION AND UTILITY FUNCTIONS
# ============================================================================

def constellation_density_cmap():
    """Colormap for constellation density plots."""
    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    return cmap


def render_constellation_density(ax, constellation, axis_limit, *,
                                 density_geometry="square", density_bins=96,
                                 hex_gridsize=48):
    """Render constellation density on matplotlib axis."""
    density_cmap = constellation_density_cmap()
    ax.set_facecolor("#f8fafc")

    if density_geometry == "square":
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
    elif density_geometry == "hex":
        ax.hexbin(
            constellation.real,
            constellation.imag,
            gridsize=hex_gridsize,
            extent=(-axis_limit, axis_limit, -axis_limit, axis_limit),
            cmap=density_cmap,
            mincnt=1,
            bins="log",
            linewidths=0.0,
            alpha=0.95,
            zorder=1,
        )
    else:
        raise ValueError("density_geometry must be 'square' or 'hex'")

    ax.scatter(
        constellation.real, constellation.imag,
        s=4, c="#0f172a", alpha=0.12, linewidths=0, zorder=2)


def plot_constellation_triplet(constellations, axis_limit, save_path,
                              symbol_levels,
                              *, density_geometry="square",
                              density_bins=96,
                              hex_gridsize=48,
                              x_label="I",
                              y_label="Q",
                              hide_axis_text=False, show_plot=False):
    """Plot three constellations side-by-side."""
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
        render_constellation_density(
            ax, constellation, axis_limit,
            density_geometry=density_geometry,
            density_bins=density_bins,
            hex_gridsize=hex_gridsize,
        )
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
            ax.set_title(title, fontsize=10)
        else:
            ax.set_title(title, fontsize=13)
            ax.tick_params(axis="both", labelsize=12)
            if ax == axes[0]:
                ax.set_ylabel(y_label, fontsize=13)
            if ax == axes[1]:
                ax.set_xlabel(x_label, fontsize=13)

    fig.savefig(save_path, dpi=200)
    if show_plot:
        plt.show()
    plt.close(fig)


def model_to_dict(model):
    """Convert Pydantic model to dictionary."""
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def stable_json_dumps(data):
    """JSON dumps with stable key ordering."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def compute_spec_hash(data_spec):
    """Compute hash of specification for caching."""
    digest = hashlib.sha256(stable_json_dumps(data_spec).encode("utf-8"))
    return digest.hexdigest()[:12]


def build_output_paths(save_dir, save_name, spec_hash):
    """Build output file paths."""
    base = Path(save_name)
    unique_stem = f"{base.stem}_{spec_hash}"
    plot_dir = Path(save_dir)
    results_dir = Path("results") / plot_dir
    return {
        "constellation_plot": plot_dir / f"{unique_stem}{base.suffix}",
        "constellation_plot_no_ticks": plot_dir / f"{unique_stem}_no_ticks{base.suffix}",
        "results_json": results_dir / f"{unique_stem}_results.json",
    }


def print_table(rows):
    """Print formatted table."""
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


def complex_array_to_dict(values):
    """Convert complex array to dict representation."""
    arr = np.asarray(values, dtype=np.complex128)
    return {
        "real": arr.real.tolist(),
        "imag": arr.imag.tolist(),
    }


def serialize_paths(value):
    """Serialize Path objects in nested structures."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [serialize_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: serialize_paths(item) for key, item in value.items()}
    return value


def aw_from_time(At, dt):
    """Convert time-domain signal to frequency domain."""
    n = len(At)
    return np.fft.fftshift(np.fft.ifft(At)) * n * dt


def power_w_to_dbm(power_w):
    """Convert power in watts to dBm."""
    if power_w <= 0:
        return float("-inf")
    return 10.0 * np.log10(power_w / 1e-3)


def integrate_band_power_w(aw, freq_thz, center_thz, bandwidth_thz, time_window_ps):
    """Integrate spectral energy over a band and convert to average power [W]."""
    if len(freq_thz) < 2 or time_window_ps <= 0:
        return float("nan")
    mask = np.abs(freq_thz - center_thz) <= bandwidth_thz / 2.0
    df = freq_thz[1] - freq_thz[0]
    energy_w_ps = np.sum(np.abs(aw[mask]) ** 2) * df
    return float(energy_w_ps / time_window_ps)


def estimate_guard_symbols(fiber_betas, fiber_length_m, symbol_rate_thz, pulse_shape,
                           rrc_rolloff, *, min_guard_symbols=48, safety_margin_symbols=8):
    """Estimate a safe edge guard from dispersive temporal spreading."""
    if symbol_rate_thz <= 0:
        raise ValueError("symbol_rate_thz must be positive")
    if fiber_length_m < 0:
        raise ValueError("fiber_length_m must be non-negative")

    beta2 = abs(float(fiber_betas[0])) if fiber_betas else 0.0
    occupied_bw_thz = symbol_rate_thz * (1.0 + rrc_rolloff) if pulse_shape == "rrc" else symbol_rate_thz
    symbol_period_ps = 1.0 / symbol_rate_thz
    edge_delay_ps = 2.0 * np.pi * beta2 * fiber_length_m * (occupied_bw_thz / 2.0)
    estimated_symbols = int(math.ceil(edge_delay_ps / symbol_period_ps)) + safety_margin_symbols
    return max(min_guard_symbols, estimated_symbols)


# ============================================================================
# CONFIGURATION MODELS
# ============================================================================

class ModulationConfig(BaseModel):
    order: int
    symbol_rate_gbd: float
    samples_per_symbol: int
    rrc_rolloff: float
    rrc_span_symbols: int
    n_symbols: int
    pulse_shape: str = "rrc"


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
    ssfm_max_nonlinear_phase_deg: float
    ssfm_max_step_m: float
    propagation_backend: str = "custom_ssfm"


class OutputConfig(BaseModel):
    save_dir: str
    save_name: str
    show_plot: bool
    density_geometry: str = "square"
    density_bins: int = 96
    hex_gridsize: int = 48


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
    """Load configuration from TOML file."""
    data = toml_load(path)

    if hasattr(SimulationConfig, "model_validate"):
        return SimulationConfig.model_validate(data)
    return SimulationConfig.parse_obj(data)


# ============================================================================
# MAIN SIMULATION LOOP
# ============================================================================

def run_wdm_nli_simulation(config_path: str | Path | None = None):
    """Run the full WDM NLI simulation and return a structured results payload.

    Args:
        config_path: Path to TOML config. If None, uses input/wdm_nli_config.toml.

    Returns:
        Dictionary containing summaries, trial data, plot metadata and
        a compact `power_report` for external consumers.
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / "input" / "wdm_nli_config.toml"
    else:
        config_path = Path(config_path)

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
    if solver.ssfm_max_step_m <= 0:
        raise ValueError("solver.ssfm_max_step_m must be positive")
    if solver.ssfm_max_nonlinear_phase_deg <= 0:
        raise ValueError("solver.ssfm_max_nonlinear_phase_deg must be positive")
    if solver.propagation_backend not in {"custom_ssfm", "gnlse"}:
        raise ValueError("solver.propagation_backend must be 'custom_ssfm' or 'gnlse'")
    if mod.pulse_shape not in {"rrc", "nyquist_rect"}:
        raise ValueError("modulation.pulse_shape must be 'rrc' or 'nyquist_rect'")

    symbol_rate_thz = mod.symbol_rate_gbd * 1e-3
    guard_symbols = estimate_guard_symbols(
        fiber.betas,
        fiber.fiber_length_m,
        symbol_rate_thz,
        mod.pulse_shape,
        mod.rrc_rolloff,
    )

    # === DERIVED PARAMETERS ===
    equalizer_taps = 11
    data_spec = {
        "modulation": model_to_dict(mod),
        "wdm": model_to_dict(wdm),
        "fiber": model_to_dict(fiber),
        "monte_carlo": model_to_dict(mc),
        "solver": model_to_dict(solver),
        "receiver": {
            "architecture": "matched_receiver_dar2015",
            "equalizer_taps": equalizer_taps,
            "guard_symbols": guard_symbols,
            "ssfm_max_nonlinear_phase_deg": solver.ssfm_max_nonlinear_phase_deg,
            "ssfm_max_step_m": solver.ssfm_max_step_m,
            "propagation_backend": solver.propagation_backend,
        },
        "plotting": {
            "density_geometry": output.density_geometry,
            "density_bins": output.density_bins,
            "hex_gridsize": output.hex_gridsize,
        },
    }
    spec_hash = compute_spec_hash(data_spec)
    output_paths = build_output_paths(output.save_dir, output.save_name, spec_hash)

    resolution = mod.n_symbols * mod.samples_per_symbol
    dt = 1.0 / (symbol_rate_thz * mod.samples_per_symbol)
    time_window = dt * (resolution - 1)

    df = 1.0 / (resolution * dt)
    f_max = 0.5 / dt

    channel_bw_thz = (
        symbol_rate_thz * (1 + mod.rrc_rolloff)
        if mod.pulse_shape == "rrc"
        else symbol_rate_thz
    )
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
        "equalizer_taps": equalizer_taps,
        "guard_symbols": guard_symbols,
    }

    rows = [
        ("modulation", f"{mod.order}-QAM", "", ""),
        ("symbol_rate", f"{mod.symbol_rate_gbd:g}", "GBd", ""),
        ("samples_per_symbol", mod.samples_per_symbol, "", ""),
        ("pulse_shape", mod.pulse_shape, "", "rrc or nyquist_rect"),
        ("rrc_rolloff", f"{mod.rrc_rolloff:g}", "", ""),
        ("rrc_span_symbols", mod.rrc_span_symbols, "symbols", ""),
        ("guard_symbols", guard_symbols, "symbols", "dispersion-based safe edge trim"),
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
        ("ssfm_max_nonlinear_phase_deg", f"{solver.ssfm_max_nonlinear_phase_deg:g}", "deg", "backprop constraint"),
        ("ssfm_max_step_m", f"{solver.ssfm_max_step_m:g}", "m", "backprop constraint"),
        ("propagation_backend", solver.propagation_backend, "", "custom_ssfm or gnlse"),
        ("save_dir", output.save_dir, "", ""),
        ("save_name", output.save_name, "", ""),
        ("density_geometry", output.density_geometry, "", ""),
        ("density_bins", output.density_bins, "", "square histogram bins"),
        ("hex_gridsize", output.hex_gridsize, "", "hexbin grid size"),
        ("results_dir", output_paths["results_json"].parent, "", ""),
        ("data_save_name", output_paths["results_json"].name, "", ""),
        ("show_plot", output.show_plot, "", ""),
        ("receiver_arch", "Matched (Dar 2014)", "", "New implementation"),
    ]

    print_table(rows)

    os.makedirs(output_paths["constellation_plot"].parent, exist_ok=True)
    os.makedirs(output_paths["results_json"].parent, exist_ok=True)

    # === PREPARE MATCHED RECEIVER ===
    rng = np.random.default_rng(mc.rng_seed)
    if mod.pulse_shape == "rrc":
        taps = rrc_taps(mod.rrc_rolloff, mod.rrc_span_symbols, mod.samples_per_symbol)
        receiver_rolloff = mod.rrc_rolloff
        receiver_processing = "matched"
    else:
        taps = np.array([1.0], dtype=np.float64)
        receiver_rolloff = 0.0
        receiver_processing = "rect_fd"
    
    receiver_config = MatchedFilterConfig(
        sps=mod.samples_per_symbol,
        symbol_rate_thz=symbol_rate_thz,
        rrc_rolloff=receiver_rolloff,
        rrc_span_symbols=mod.rrc_span_symbols,
        guard_symbols=guard_symbols,
        equalizer_taps=equalizer_taps,
        fiber_nonlinearity=fiber.nonlinearity,
        ssfm_max_nonlinear_phase_deg=solver.ssfm_max_nonlinear_phase_deg,
        ssfm_max_step_m=solver.ssfm_max_step_m,
        propagation_backend=solver.propagation_backend,
        receiver_processing=receiver_processing,
    )
    
    dispersion_model = GNLSEDispersionModel(
        fiber.loss_db_per_m, 
        fiber.betas, 
        fiber.fiber_length_m,
        wavelength_nm=fiber.wavelength_nm,
    )

    # === MONTE CARLO TRIALS ===
    center_noise_samples = []
    center_snr_samples = []
    center_rx_power_samples = []
    center_nli_norm_samples = []

    center_noise_samples_matched = []
    center_snr_samples_matched = []
    center_rx_power_samples_matched = []
    center_nli_norm_samples_matched = []
    center_matched_scale_w_per_symbol = []

    total_rx_power_nl_samples_w = []
    total_rx_power_lin_samples_w = []
    center_rx_band_power_nl_samples_w = []
    center_rx_band_power_lin_samples_w = []

    plot_const_lin = None
    plot_const_nl = None
    plot_const_nl_rot = None
    plot_symbols_ref = None
    plot_gain = None
    
    summary_results = {}

    print(f"\nRunning {mc.n_trials} Monte Carlo trials with matched receiver (Dar 2014)...\n")

    for trial in range(mc.n_trials):
        # Time vector
        t = np.linspace(-time_window / 2, time_window / 2, resolution)
        
        # === SIGNAL GENERATION ===
        A_t, center_tx_symbols = make_wdm_field(
            t,
            wdm.channel_spacing_thz,
            wdm.n_channels,
            wdm.power_per_channel_w,
            mod.n_symbols,
            mod.samples_per_symbol,
            taps,
            mod.order,
            rng,
            pulse_shape=mod.pulse_shape,
            symbol_rate_thz=symbol_rate_thz,
            rrc_rolloff=mod.rrc_rolloff,
        )

        print(f"  Trial {trial + 1}/{mc.n_trials}: Running GNLSE...", end='', flush=True)
        sol_nl, nl_segments = propagate_wdm_signal_constrained(
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
            atol=solver.atol,
            max_step_m=solver.ssfm_max_step_m,
            max_nonlinear_phase_deg=solver.ssfm_max_nonlinear_phase_deg,
            propagation_backend=solver.propagation_backend,
        )
        sol_lin, lin_segments = propagate_wdm_signal_constrained(
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
            atol=solver.atol,
            max_step_m=solver.ssfm_max_step_m,
            max_nonlinear_phase_deg=solver.ssfm_max_nonlinear_phase_deg,
            propagation_backend=solver.propagation_backend,
        )
        print(" Done.")
        print(f"    forward segments: NL={nl_segments}, LIN={lin_segments}")

        # === FREQUENCY DOMAIN ANALYSIS (RAW SPECTRA) ===
        freq = (sol_nl.W - sol_nl.w_0) / (2 * np.pi)

        aw_nl_raw = aw_from_time(sol_nl.At[-1, :], dt)
        aw_lin_raw = aw_from_time(sol_lin.At[-1, :], dt)

        center_metrics_raw = estimate_nli_noise_from_spectrum(
            aw_nl_raw, aw_lin_raw, freq, 0.0, channel_bw_thz)

        center_noise_samples.append(center_metrics_raw.noise_power)
        center_snr_samples.append(center_metrics_raw.snr)
        center_rx_power_samples.append(center_metrics_raw.rx_power)
        center_nli_norm_samples.append(center_metrics_raw.nli_norm)

        center_rx_band_power_nl_samples_w.append(
            integrate_band_power_w(aw_nl_raw, freq, 0.0, channel_bw_thz, time_window)
        )
        center_rx_band_power_lin_samples_w.append(
            integrate_band_power_w(aw_lin_raw, freq, 0.0, channel_bw_thz, time_window)
        )

        # Physical total received power at output plane (time-domain, W)
        total_rx_power_nl_samples_w.append(float(np.mean(np.abs(sol_nl.At[-1, :]) ** 2)))
        total_rx_power_lin_samples_w.append(float(np.mean(np.abs(sol_lin.At[-1, :]) ** 2)))

        # === MATCHED RECEIVER (Dar 2014 procedure) ===
        print(f"  Trial {trial + 1}/{mc.n_trials}: Matched receiver...", end='', flush=True)
        
        received = ReceivedSignals(
            At_linear=sol_lin.At[-1, :],
            At_nonlinear=sol_nl.At[-1, :],
            time_vector=t,
            frequency_vector=freq
        )

        receiver_result = matched_receiver_procedure(
            received,
            receiver_config,
            dispersion_model,
            taps,
            center_tx_symbols
        )
        
        print(" Done.")

        # Extract results
        symbols_lin_eq = receiver_result["symbols_linear_eq"]
        symbols_nl_eq = receiver_result["symbols_nonlinear_eq"]
        symbols_nl_rot = receiver_result["symbols_nonlinear_phase_comp"]
        phase_rotation = receiver_result["phase_rotation"]
        channel_gain = receiver_result["constellation_gain"]
        ref_symbols = receiver_result["reference_symbols"]

        # === NLI ESTIMATION (GUIDE PRESCRIPTION / MATCHED RECEIVER) ===
        # After removing average phase rotation, evaluate the offset between
        # nonlinear constellation points and ideal no-NLI points.
        matched_metrics = estimate_nli_noise_from_constellation(
            symbols_nl_rot,
            symbols_lin_eq,
        )
        # Map matched-constellation powers to physical watts by calibration:
        # force linear equalized symbol power to match center-channel
        # linear spectral band power for this trial.
        lin_sym_power = float(np.mean(np.abs(symbols_lin_eq) ** 2)) if len(symbols_lin_eq) else float("nan")
        center_lin_band_power_w = center_rx_band_power_lin_samples_w[-1]
        if (np.isfinite(lin_sym_power) and lin_sym_power > 0 and
                np.isfinite(center_lin_band_power_w) and center_lin_band_power_w > 0):
            scale_w_per_symbol = center_lin_band_power_w / lin_sym_power
        else:
            scale_w_per_symbol = float("nan")

        center_matched_scale_w_per_symbol.append(scale_w_per_symbol)

        noise_power_w = matched_metrics.noise_power * scale_w_per_symbol
        signal_power_w = matched_metrics.signal_power * scale_w_per_symbol
        rx_power_w = matched_metrics.rx_power * scale_w_per_symbol

        snr_matched = (float("inf") if noise_power_w == 0
                       else signal_power_w / noise_power_w)
        nli_norm_matched = (float("inf") if rx_power_w == 0
                            else noise_power_w / rx_power_w)

        center_noise_samples_matched.append(float(noise_power_w))
        center_snr_samples_matched.append(float(snr_matched))
        center_rx_power_samples_matched.append(float(rx_power_w))
        center_nli_norm_samples_matched.append(float(nli_norm_matched))

        # Store first trial for plotting
        if trial == 0:
            plot_const_lin = symbols_lin_eq
            plot_const_nl = symbols_nl_eq
            plot_const_nl_rot = symbols_nl_rot
            plot_symbols_ref = ref_symbols
            plot_gain = channel_gain

    print(f"\n✓ All {mc.n_trials} trials completed.\n")

    # === RESULTS SUMMARY ===
    def print_results_section(label, noise_samples, snr_samples, rx_samples, nli_norm_samples,
                              quantity_mode="energy"):
        summary = summarize_nli_results(noise_samples, snr_samples, rx_samples, nli_norm_samples)

        # For raw spectral metrics, samples are energies [W·ps].
        # For matched constellation metrics (after calibration), samples are powers [W].
        if quantity_mode == "energy":
            tw_ps = time_window
            noise_power_w = summary["nli_noise_avg"] / tw_ps if tw_ps > 0 else float("nan")
            rx_power_w = summary["rx_power_avg"] / tw_ps if tw_ps > 0 else float("nan")
        elif quantity_mode == "power":
            noise_power_w = summary["nli_noise_avg"]
            rx_power_w = summary["rx_power_avg"]
        else:
            raise ValueError("quantity_mode must be 'energy' or 'power'")

        signal_power_w = noise_power_w * summary["snr_avg"] if np.isfinite(summary["snr_avg"]) else float("nan")

        summary.update({
            "nli_energy_avg_w_ps": summary["nli_noise_avg"],
            "rx_energy_avg_w_ps": summary["rx_power_avg"],
            "nli_power_avg_w": float(noise_power_w),
            "signal_power_avg_w": float(signal_power_w),
            "rx_power_avg_w": float(rx_power_w),
            "nli_power_avg_dbm": float(power_w_to_dbm(noise_power_w)) if np.isfinite(noise_power_w) else float("nan"),
            "signal_power_avg_dbm": float(power_w_to_dbm(signal_power_w)) if np.isfinite(signal_power_w) else float("nan"),
            "rx_power_avg_dbm": float(power_w_to_dbm(rx_power_w)) if np.isfinite(rx_power_w) else float("nan"),
        })

        rows = []
        if quantity_mode == "energy":
            rows.extend([
                ("nli_energy_avg", f"{summary['nli_energy_avg_w_ps']:.6g}", "W·ps",
                 f"{summary['nli_noise_avg_db']:.3f} dB re 1 W·ps"),
                ("nli_noise_std", f"{summary['nli_noise_std']:.6g}", "W·ps",
                 f"{summary['nli_noise_std_db']:.3f} dB"),
                ("rx_energy_avg", f"{summary['rx_energy_avg_w_ps']:.6g}", "W·ps",
                 f"{summary['rx_power_avg_db']:.3f} dB re 1 W·ps"),
            ])
        else:
            rows.extend([
                ("nli_noise_std", f"{summary['nli_noise_std']:.6g}", "W",
                 f"{summary['nli_noise_std_db']:.3f} dB re 1 W"),
            ])

        rows.extend([
            ("nli_noise_power_avg", f"{summary['nli_power_avg_w']:.6g}", "W",
             f"{summary['nli_power_avg_dbm']:.3f} dBm"),
            ("signal_power_avg", f"{summary['signal_power_avg_w']:.6g}", "W",
             f"{summary['signal_power_avg_dbm']:.3f} dBm"),
            ("rx_power_avg", f"{summary['rx_power_avg_w']:.6g}", "W",
             f"{summary['rx_power_avg_dbm']:.3f} dBm"),
            ("nli_norm_avg", f"{summary['nli_norm_avg']:.6g}", "",
             f"{summary['nli_norm_avg_db']:.3f} dB"),
            ("snr_avg", f"{summary['snr_avg']:.6g}", "",
             f"{summary['snr_avg_db']:.3f} dB")
        ])
        print(f"\nResults ({label})")
        print_table(rows)
        summary_results[label] = summary

    print_results_section(
        "center channel (matched constellation prescription)",
        center_noise_samples_matched,
        center_snr_samples_matched,
        center_rx_power_samples_matched,
        center_nli_norm_samples_matched,
        quantity_mode="power",
    )
    # print_results_section(
    #     "center channel (raw spectral diagnostics)",
    #     center_noise_samples,
    #     center_snr_samples,
    #     center_rx_power_samples,
    #     center_nli_norm_samples,
    #     quantity_mode="energy",
    # )

    mean_scale = (float(np.mean(center_matched_scale_w_per_symbol))
                  if center_matched_scale_w_per_symbol else float("nan"))
    print("\nMatched-power calibration")
    print_table([
        ("matched_scale_avg", f"{mean_scale:.6g}", "W / symbol-power-unit",
         "maps equalized constellation powers to physical W"),
    ])
    summary_results["matched_power_calibration"] = {
        "scale_w_per_symbol_avg": mean_scale,
        "scale_w_per_symbol_samples": [float(x) for x in center_matched_scale_w_per_symbol],
    }

    # Output-plane physical total power (all frequencies, time-domain)
    total_nl_avg_w = float(np.mean(total_rx_power_nl_samples_w)) if total_rx_power_nl_samples_w else float("nan")
    total_lin_avg_w = float(np.mean(total_rx_power_lin_samples_w)) if total_rx_power_lin_samples_w else float("nan")
    total_rows = [
        ("rx_total_power_nl_avg", f"{total_nl_avg_w:.6g}", "W", f"{power_w_to_dbm(total_nl_avg_w):.3f} dBm"),
        ("rx_total_power_lin_avg", f"{total_lin_avg_w:.6g}", "W", f"{power_w_to_dbm(total_lin_avg_w):.3f} dBm"),
    ]
    print("\nReceived plane total power (time-domain)")
    print_table(total_rows)
    summary_results["received_plane_total_power"] = {
        "rx_total_power_nl_avg_w": total_nl_avg_w,
        "rx_total_power_nl_avg_dbm": float(power_w_to_dbm(total_nl_avg_w)),
        "rx_total_power_lin_avg_w": total_lin_avg_w,
        "rx_total_power_lin_avg_dbm": float(power_w_to_dbm(total_lin_avg_w)),
    }

    # Center-channel received-power attenuation sanity check
    total_loss_db = fiber.loss_db_per_m * fiber.fiber_length_m
    attenuation_factor_power = 10.0 ** (-total_loss_db / 10.0)
    expected_rx_channel_power_w = wdm.power_per_channel_w * attenuation_factor_power

    center_lin_avg_w = float(np.mean(center_rx_band_power_lin_samples_w)) if center_rx_band_power_lin_samples_w else float("nan")
    center_nl_avg_w = float(np.mean(center_rx_band_power_nl_samples_w)) if center_rx_band_power_nl_samples_w else float("nan")
    center_delta_lin_db = (10.0 * np.log10(center_lin_avg_w / expected_rx_channel_power_w)
                           if center_lin_avg_w > 0 and expected_rx_channel_power_w > 0 else float("nan"))

    center_rows = [
        ("center_channel_lin_avg", f"{center_lin_avg_w:.6g}", "W", f"{power_w_to_dbm(center_lin_avg_w):.3f} dBm"),
        ("center_channel_nl_avg", f"{center_nl_avg_w:.6g}", "W", f"{power_w_to_dbm(center_nl_avg_w):.3f} dBm"),
        ("expected_center_channel", f"{expected_rx_channel_power_w:.6g}", "W", f"{power_w_to_dbm(expected_rx_channel_power_w):.3f} dBm"),
        ("center_lin_minus_expected", f"{center_delta_lin_db:+.6g}", "dB", "attenuation sanity"),
    ]
    print("\nCenter-channel received power sanity")
    print_table(center_rows)

    summary_results["center_channel_power_sanity"] = {
        "total_loss_db": float(total_loss_db),
        "expected_rx_channel_power_w": float(expected_rx_channel_power_w),
        "expected_rx_channel_power_dbm": float(power_w_to_dbm(expected_rx_channel_power_w)),
        "center_rx_power_lin_avg_w": center_lin_avg_w,
        "center_rx_power_lin_avg_dbm": float(power_w_to_dbm(center_lin_avg_w)),
        "center_rx_power_nl_avg_w": center_nl_avg_w,
        "center_rx_power_nl_avg_dbm": float(power_w_to_dbm(center_nl_avg_w)),
        "center_lin_minus_expected_db": float(center_delta_lin_db),
    }

    # === PLOTTING ===
    if (plot_const_lin is not None and plot_const_nl is not None and
            plot_const_nl_rot is not None and plot_symbols_ref is not None):
        has_physical_scale = np.isfinite(mean_scale) and mean_scale > 0
        amp_scale = np.sqrt(mean_scale) if has_physical_scale else 1.0
        axis_x_label = r"I ($W^{1/2}$)" if has_physical_scale else "I (arb.)"
        axis_y_label = r"Q ($W^{1/2}$)" if has_physical_scale else "Q (arb.)"

        constellations = [
            ("Linear (CD+SPM compensated)", (plot_const_lin / plot_gain) * amp_scale),
            ("Nonlinear (CD+SPM compensated)", (plot_const_nl / plot_gain) * amp_scale),
            ("Nonlinear + phase rotation", (plot_const_nl_rot / plot_gain) * amp_scale),
        ]
        
        stacked = np.concatenate([constellation for _, constellation in constellations])
        max_extent = float(np.max(np.abs(np.concatenate((stacked.real, stacked.imag)))))
        axis_limit = 1.05 * max_extent if max_extent > 0 else 1.0
        symbol_levels = qam_axis_levels(mod.order) * amp_scale

        print(f"\nGenerating constellation plots...")
        plot_constellation_triplet(
            constellations, axis_limit, 
            output_paths["constellation_plot"], symbol_levels,
            density_geometry=output.density_geometry,
            density_bins=output.density_bins,
            hex_gridsize=output.hex_gridsize,
            x_label=axis_x_label,
            y_label=axis_y_label,
            show_plot=output.show_plot,
        )
        plot_constellation_triplet(
            constellations, axis_limit, 
            output_paths["constellation_plot_no_ticks"], symbol_levels,
            density_geometry=output.density_geometry,
            density_bins=output.density_bins,
            hex_gridsize=output.hex_gridsize,
            x_label=axis_x_label,
            y_label=axis_y_label,
            hide_axis_text=True,
        )
        print(f"✓ Plots saved to:")
        print(f"  - {output_paths['constellation_plot']}")
        print(f"  - {output_paths['constellation_plot_no_ticks']}")

        plot_payload = {
            "constellations": {
                name: complex_array_to_dict(const)
                for name, const in constellations
            },
            "axis_limit": axis_limit,
            "symbol_levels": symbol_levels.tolist(),
            "axis_x_label": axis_x_label,
            "axis_y_label": axis_y_label,
            "has_physical_scale": bool(has_physical_scale),
            "density_geometry": output.density_geometry,
            "density_bins": output.density_bins,
            "hex_gridsize": output.hex_gridsize,
        }
    else:
        plot_payload = None

    # === SAVE RESULTS BUNDLE ===
    trial_results = {
        "center_channel_matched": {
            "noise": [float(x) for x in center_noise_samples_matched],
            "snr": [float(x) for x in center_snr_samples_matched],
            "rx_power": [float(x) for x in center_rx_power_samples_matched],
            "nli_norm": [float(x) for x in center_nli_norm_samples_matched],
            "scale_w_per_symbol": [float(x) for x in center_matched_scale_w_per_symbol],
        },
        "center_channel_raw": {
            "noise": [float(x) for x in center_noise_samples],
            "snr": [float(x) for x in center_snr_samples],
            "rx_power": [float(x) for x in center_rx_power_samples],
            "nli_norm": [float(x) for x in center_nli_norm_samples],
        },
    }

    plot_outputs = {
        "constellation_plot": output_paths["constellation_plot"],
        "constellation_plot_no_ticks": output_paths["constellation_plot_no_ticks"],
        "results_json": output_paths["results_json"],
    }

    matched_summary = summary_results.get("center channel (matched constellation prescription)", {})
    sanity_summary = summary_results.get("center_channel_power_sanity", {})
    power_report = {
        "center_channel": {
            "signal_power_avg_w": matched_summary.get("signal_power_avg_w"),
            "signal_power_avg_dbm": matched_summary.get("signal_power_avg_dbm"),
            "nli_power_avg_w": matched_summary.get("nli_power_avg_w"),
            "nli_power_avg_dbm": matched_summary.get("nli_power_avg_dbm"),
            "rx_power_avg_w": matched_summary.get("rx_power_avg_w"),
            "rx_power_avg_dbm": matched_summary.get("rx_power_avg_dbm"),
            "snr_avg": matched_summary.get("snr_avg"),
            "snr_avg_db": matched_summary.get("snr_avg_db"),
            "nli_norm_avg": matched_summary.get("nli_norm_avg"),
            "nli_norm_avg_db": matched_summary.get("nli_norm_avg_db"),
        },
        "sanity": {
            "expected_center_channel_power_w": sanity_summary.get("expected_rx_channel_power_w"),
            "expected_center_channel_power_dbm": sanity_summary.get("expected_rx_channel_power_dbm"),
            "center_rx_power_lin_avg_w": sanity_summary.get("center_rx_power_lin_avg_w"),
            "center_rx_power_lin_avg_dbm": sanity_summary.get("center_rx_power_lin_avg_dbm"),
            "center_rx_power_nl_avg_w": sanity_summary.get("center_rx_power_nl_avg_w"),
            "center_rx_power_nl_avg_dbm": sanity_summary.get("center_rx_power_nl_avg_dbm"),
        },
    }

    payload = {
        "config_path": str(config_path),
        "config": model_to_dict(config),
        "data_spec": data_spec,
        "spec_hash": spec_hash,
        "derived_parameters": derived_parameters,
        "power_report": power_report,
        "summaries": summary_results,
        "trial_results": trial_results,
        "plot_outputs": serialize_paths(plot_outputs),
        "plot_payload": plot_payload,
    }
    
    results_save_path = output_paths["results_json"]
    with results_save_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\n✓ Results bundle saved: {results_save_path}\n")
    return payload


if __name__ == '__main__':
    run_wdm_nli_simulation()
