# Matched Receiver Implementation and NLI Estimation

## Overview

This directory now contains a refactored WDM NLI simulation framework following 
the matched receiver procedure from **Dar et al. 2015**.

## File Structure

### Core Modules (New)

1. **`signal_generation.py`** - Signal generation and GNLSE propagation
   - RRC pulse shaping filter design
   - QAM constellation generation
   - WDM field generation (multiple channels)
   - GNLSE setup and propagation

2. **`receiver_matched.py`** - Matched receiver (Dar 2014 procedure)
   - Channel isolation with matched optical filter
   - Ideal back-propagation (CD + SPM compensation)
   - Symbol detection and sampling
   - Least-squares equalization
   - Complete `matched_receiver_procedure()` entry point

3. **`nli_estimation.py`** - NLI calculation and analysis
   - Spectral domain NLI estimation
   - Statistical summary functions

### Scripts

1. **`nli.py`** ✨ **MAIN** - Main script using matched receiver
   - Modular design using the three core modules
   - Monte Carlo trials with matched receiver
   - Constellation plotting and NLI statistics
   - Results saved in JSON bundle format

2. Supporting modules
   - `signal_generation.py`
   - `receiver_matched.py`
   - `nli_estimation.py`

### Reference

3. **`guide.md`** - Comprehensive documentation
   - Dar 2014 matched receiver procedure
   - Module structure and API reference
   - Step-by-step explanation with equations
   - Next steps for NLI power estimation

## Quick Start

### Run the new matched receiver implementation:

```bash
python nli.py
```

Configuration is loaded from: `../../input/wdm_nli_config.toml`

Results are saved to: `results/media/constellation/`

### Basic usage of the modules:

```python
from signal_generation import make_wdm_field, propagate_wdm_signal_constrained
from receiver_matched import matched_receiver_procedure, MatchedFilterConfig
from nli_estimation import estimate_nli_noise_from_spectrum

# 1. Generate WDM signal
A_t, center_symbols = make_wdm_field(...)

# 2. Propagate through constrained GNLSE (linear and nonlinear)
sol_lin, _ = propagate_wdm_signal_constrained(..., nonlinearity=0.0)
sol_nl, _ = propagate_wdm_signal_constrained(..., nonlinearity=gamma)

# 3. Process with matched receiver
receiver_config = MatchedFilterConfig(...)
result = matched_receiver_procedure(
    received, receiver_config, dispersion_model, rrc_taps, center_symbols
)

# 4. Estimate NLI noise
metrics = estimate_nli_noise_from_spectrum(
    aw_nl, aw_lin, freq, center_thz=0.0, bandwidth_thz=0.075
)
print(f"NLI power: {metrics.noise_power:.6g}")
print(f"SNR: {metrics.snr:.3f}")
print(f"NLI normalized: {metrics.nli_norm:.6g}")
```

## Key Functions

### Signal Generation
- `rrc_taps()` - Root-raised-cosine filter
- `make_wdm_field()` - Generate multi-channel WDM signal
- `propagate_field_constrained()` - Shared constrained propagator (forward/backward)
- `propagate_wdm_signal_constrained()` - Forward constrained propagation wrapper

### Matched Receiver (5-step procedure)
1. `extract_center_channel_matched()` - Isolate center channel
2. `backpropagate_channel_ssfm()` - Back-propagate to remove CD/SPM
3. `matched_filter_detection()` - Apply pulse-shaped matched filter
4. `find_optimal_sampling_offset()`, `train_ls_equalizer()` - Symbol detection
5. `estimate_phase_rotation()` - Phase compensation

**Main entry point**: `matched_receiver_procedure()`

### NLI Estimation
- `estimate_nli_noise_from_spectrum()` - Frequency domain NLI
- `summarize_nli_results()` - Statistical summary

## Data Specifications

### Input
- `input/wdm_nli_config.toml` - Simulation configuration (Pydantic models)

### Output (JSON)
```json
{
  "config_path": "...",
  "data_spec": { ... },
  "spec_hash": "...",
  "derived_parameters": { ... },
  "summaries": {
    "center channel (raw spectral)": {
      "nli_noise_avg": 1.23e-4,
      "nli_noise_avg_db": -39.1,
      ...
    }
  },
  "trial_results": {
    "center_channel_raw": { ... },
    "all_channels_raw": { ... }
  },
  "plot_payload": {
    "constellations": {
      "Linear (CD+SPM compensated)": { "real": [...], "imag": [...] },
      ...
    }
  }
}
```

## Implementation Notes

### Matched Receiver Procedure (Dar et al. 2015)

The receiver uses a **5-step procedure** to isolate and measure NLI:

1. **Channel Isolation**: Matched optical filter isolates center channel at 0 THz
   - Uses raised-cosine demultiplexer with RRC rolloff
   - Removes adjacent channels' power

2. **Ideal Back-Propagation**: Frequency-domain CD+SPM compensation
   - Eliminates dispersive phase: $\exp(-j \phi_{disp}(L))$
   - Noiseless ideal compensation (for analysis only)

3. **Matched Filter Detection**: Pulse-shaped symbol detection
   - Convolve with RRC filter taps
   - Prepares for optimal symbol sampling

4. **Symbol Equalization**: LS-trained symbol-spaced FIR equalizer
   - Train on linear case (no NLI reference)
   - Apply same equalizer to nonlinear case
   - Equalizes channel response and clock timing

5. **Phase Compensation**: Estimate and correct phase drift
   - NLI causes phase rotation: `exp(j * phase_nli)`
   - Rotate nonlinear constellation for comparison

### NLI Measurement Methods

**Spectral Method** (recommended):
- Compute $NLI = A_{NL}(f) - A_{LIN}(f)$
- Integrate power over frequency band
- Results: noise power, signal power, SNR, NLI normalization

Receiver-domain diagnostics are available from `matched_receiver_procedure()`
(equalized constellations, phase compensation, and training artifacts).

## Testing and Comparison

Run the consolidated implementation via `nli.py`.

## Future Extensions

Prepared for:
- Additional receiver architectures (DSP equalization, adaptive filtering)
- Multiple polarization support
- Time-varying NLI characterization
- Frequency-selective NLI models
- Interaction term calculations

## References

- **Dar et al. 2015**: "Nonlinear interference noise accumulation in long-haul 
  optical systems with coherent detection"
  - Defines matched receiver procedure
  - Provides theoretical NLI accumulation models

See `guide.md` for complete documentation and step-by-step explanation.
