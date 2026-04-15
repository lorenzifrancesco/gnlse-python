"""
Nonlinear Interference (NLI) noise estimation following Dar et al. 2015.

NLI is estimated by subtracting the linear propagation result from the 
nonlinear propagation result, isolating the nonlinear interference noise.

The procedure:
1. Propagate the same WDM signal through two GNLSE runs: linear (gamma=0) and nonlinear
2. Extract constellations from both runs using the matched receiver
3. Compute NLI noise power as the difference in received power
4. Calculate SNR and NLI normalization metrics

Reference:
    Dar et al. 2015: "Nonlinear interference noise accumulation in long-haul 
    optical systems with coherent detection"
"""

import numpy as np
from typing import NamedTuple


class NLIMetrics(NamedTuple):
    """Calculated NLI metrics."""
    noise_power: float
    signal_power: float
    rx_power: float
    snr: float
    nli_norm: float


def estimate_nli_noise_from_constellation(symbols_nl_phase_comp,
                                          symbols_ideal_no_nli):
    """
    Estimate NLI from constellation offsets after phase-rotation removal.

    Guide/Dar-style prescription:
    1) remove average nonlinear phase rotation
    2) compute offset to ideal no-NLI constellation points

    Args:
        symbols_nl_phase_comp: Nonlinear symbols after phase compensation
        symbols_ideal_no_nli: Ideal symbols in absence of nonlinearity
            (typically equalized linear-case symbols)

    Returns:
        NLIMetrics with noise_power, signal_power, rx_power, snr, nli_norm
        where powers are symbol-domain mean-square values.
    """
    if len(symbols_nl_phase_comp) == 0 or len(symbols_ideal_no_nli) == 0:
        return NLIMetrics(0.0, 0.0, 0.0, float("inf"), float("inf"))

    n = min(len(symbols_nl_phase_comp), len(symbols_ideal_no_nli))
    s_nl = np.asarray(symbols_nl_phase_comp[:n], dtype=np.complex128)
    s_id = np.asarray(symbols_ideal_no_nli[:n], dtype=np.complex128)

    err = s_nl - s_id
    noise_power = float(np.mean(np.abs(err) ** 2))
    signal_power = float(np.mean(np.abs(s_id) ** 2))
    rx_power = float(np.mean(np.abs(s_nl) ** 2))

    snr = float("inf") if noise_power == 0 else float(signal_power / noise_power)
    nli_norm = float("inf") if rx_power == 0 else float(noise_power / rx_power)

    return NLIMetrics(
        noise_power=noise_power,
        signal_power=signal_power,
        rx_power=rx_power,
        snr=snr,
        nli_norm=nli_norm,
    )


def estimate_nli_noise_from_spectrum(aw_nl, aw_lin, freq, center_thz, 
                                     bandwidth_thz):
    """
    Estimate NLI noise power by spectral analysis.
    
    Compute NLI in a specific frequency band by subtracting linear spectrum
    from nonlinear spectrum. This method works in frequency domain and 
    integrates over a specified bandwidth.
    
    Args:
        aw_nl: Nonlinear received spectrum (complex array)
        aw_lin: Linear received spectrum (complex array)
        freq: Frequency vector (THz)
        center_thz: Center frequency of the band of interest (THz)
        bandwidth_thz: Bandwidth of integration region (THz)
        
    Returns:
        NLIMetrics with noise_power, signal_power, rx_power, snr, nli_norm
    """
    # Compute NLI spectrum as difference
    aw_nli = aw_nl - aw_lin
    
    # Create mask for the band of interest
    mask = np.abs(freq - center_thz) <= bandwidth_thz / 2.0
    df = freq[1] - freq[0] if len(freq) > 1 else 1.0
    
    # Integrate power over the band
    noise_power = np.sum(np.abs(aw_nli[mask])**2) * df
    signal_power = np.sum(np.abs(aw_lin[mask])**2) * df
    rx_power = np.sum(np.abs(aw_nl[mask])**2) * df
    
    # Compute SNR and normalized NLI
    snr = np.inf if noise_power == 0 else signal_power / noise_power
    nli_norm = np.inf if rx_power == 0 else noise_power / rx_power
    
    return NLIMetrics(
        noise_power=float(noise_power),
        signal_power=float(signal_power),
        rx_power=float(rx_power),
        snr=float(snr),
        nli_norm=float(nli_norm)
    )


def summarize_nli_results(noise_samples, snr_samples, rx_samples, nli_norm_samples):
    """
    Compute statistical summary of NLI metrics across multiple trials.
    
    Args:
        noise_samples: List of NLI noise power values
        snr_samples: List of SNR values
        rx_samples: List of received power values
        nli_norm_samples: List of normalized NLI values
        
    Returns:
        Dictionary with summary statistics
    """
    noise_avg = float(np.mean(noise_samples)) if noise_samples else 0.0
    noise_std = float(np.std(noise_samples)) if noise_samples else 0.0
    snr_avg = float(np.mean(snr_samples)) if snr_samples else float('inf')
    rx_power_avg = float(np.mean(rx_samples)) if rx_samples else 0.0
    nli_norm_avg = float(np.mean(nli_norm_samples)) if nli_norm_samples else float('inf')
    
    return {
        "nli_noise_avg": noise_avg,
        "nli_noise_avg_db": 10.0 * np.log10(noise_avg) if noise_avg > 0 else float('-inf'),
        "nli_noise_std": noise_std,
        "nli_noise_std_db": 10.0 * np.log10(noise_std) if noise_std > 0 else float('-inf'),
        "rx_power_avg": rx_power_avg,
        "rx_power_avg_db": 10.0 * np.log10(rx_power_avg) if rx_power_avg > 0 else float('-inf'),
        "nli_norm_avg": nli_norm_avg,
        "nli_norm_avg_db": 10.0 * np.log10(nli_norm_avg) if nli_norm_avg > 0 else float('-inf'),
        "snr_avg": snr_avg,
        "snr_avg_db": 10.0 * np.log10(snr_avg) if snr_avg > 0 and snr_avg != float('inf') else float('-inf'),
    }

