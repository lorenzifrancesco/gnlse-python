# Calculation of nonlinear noise (NLI)

## Overview

Nonlinear interference (NLI) noise is estimated by comparing the output of two 
GNLSE simulations:
1. **Linear case**: Nonlinearity disabled (γ = 0) - produces ideal signal without NLI
2. **Nonlinear case**: Full nonlinearity enabled (γ ≠ 0) - contains both signal and NLI

NLI is computed as the difference: `NLI = Nonlinear_output - Linear_output`

Assumptions:
1. NLI is additive with respect to noise from receiver imperfections
2. NLI can be characterized through error vector magnitude (EVM) degradation
3. Ideal (noiseless) back-propagation is achievable for analysis

## Implementation: Matched Receiver (Dar et al. 2015)

This is now the standard procedure implemented in the refactored code.

### Reference

From Dar 2014:
> The results are obtained from a series of simulations considering a five-channel 
> WDM system implemented over standard single-mode fiber (dispersion of 17 ps/nm/km, 
> nonlinear coefficient γ = 1.3 [Wkm]−1, and attenuation of 0.2dB per km). We assume 
> Nyquist pulses with a perfectly square spectrum, a symbol-rate of 32 GSymbols/s 
> and a channel spacing of 50 GHz. The number of simulated symbols in each run was 
> 4096 and the total number of runs were performed with each set of system parameters 
> (each with independent and random data symbols) ranged between 100 and 500 so as to 
> accumulate sufficient statistics. As we are only interested in characterizing the NLIN, 
> we did not include amplified spontaneous emission (ASE) noise in any of the simulations.
> 
> **At the receiver, the channel of interest was isolated with a matched optical filter 
> and ideally back-propagated so as to eliminate the effects of self-phase-modulation 
> and chromatic dispersion.** All simulations were performed with a single polarization, 
> whereas the scaling of the theoretical results to the dual polarization case has been 
> discussed in [6]. For both forward and backward propagation, the scalar nonlinear 
> Schrödinger equation has been solved using the standard split-step-Fourier method 
> with a step size that was set to limit the maximum relative error per step.


#### Additional details
>For both forward and backward propagation, the scalar nonlinear Schrödinger equation has been solved using the standard split-step-Fourier method with a step size that was set to limit the maximum
> nonlinear phase variation to 0.02 degrees (and bounded from above by 1000 m). The sampling rate was 16 samples per-symbol. To extract the NLIN, we first removed the average phase rotation induced by the neighboring WDM channels and then evaluated the offset between the received constellation points and the ideal constellation points that would characterize detection in the absence of nonlinearity.


### Procedure

The matched receiver procedure consists of five main steps:

#### Step 1: Channel Isolation with Matched Optical Filter

Isolate the center channel (0 THz) using a smooth raised-cosine demultiplexer filter.

**Function**: `receiver_matched.extract_center_channel_matched()`

The filter response is:
- **Passband**: $|f| \leq \frac{f_s}{2}(1 - \alpha)$  (fully passed, gain = 1)
- **Transition**: $\frac{f_s}{2}(1 - \alpha) < |f| < \frac{f_s}{2}(1 + \alpha)$ (raised-cosine roll-off)
- **Stopband**: $|f| > \frac{f_s}{2}(1 + \alpha)$ (rejected, gain = 0)

where $f_s$ is the symbol rate and $\alpha$ is the RRC rolloff factor.

#### Step 2: Ideal Back-Propagation

Apply frequency-domain linear dispersion compensation (fiber back-propagation).

**Function**: `receiver_matched.backpropagate_channel_ssfm()`

This step eliminates:
- Chromatic dispersion (CD) accumulated through the fiber
- Self-phase modulation (SPM) accumulated through the fiber

The back-propagation is implemented as:
$$\tilde{A}(t) = \mathcal{F}^{-1}\left\{ \mathcal{F}\{A(t)\} \exp(-j \phi(L)) \right\}$$

where $\phi(L) = \int_0^L \text{Im}[D(\omega)] dz$ is the total dispersive phase.

#### Step 3: Matched Filter Detection

Apply the root-raised-cosine (RRC) matched filter for symbol detection.

**Function**: `receiver_matched.matched_filter_detection()`

Output is convolved with the RRC filter taps: `y(t) = A(t) * h(t)`

#### Step 4: Symbol Sampling and Equalization

1. Find optimal sampling phase (minimize NMSE against reference)
2. Train symbol-spaced FIR equalizer using least-squares on linear case
3. Apply equalizer to both linear and nonlinear symbols

**Functions**: 
- `receiver_matched.find_optimal_sampling_offset()` - Phase alignment
- `receiver_matched.train_ls_equalizer()` - LS equalizer training
- `receiver_matched.apply_equalizer()` - Apply trained equalizer

#### Step 5: Phase Rotation Compensation

Estimate and apply phase rotation between linear and nonlinear cases.

**Function**: `receiver_matched.estimate_phase_rotation()`

This compensates for differential phase drift between the two cases.

### Module Structure

The refactored code is split into three main modules:

- **`signal_generation.py`**: WDM signal generation and GNLSE propagation setup
  - RRC filter design: `rrc_taps()`
  - QAM constellation: `qam_symbols()`, `qam_axis_levels()`
  - WDM field: `make_wdm_field()`
  - Constrained propagation: `propagate_field_constrained()`, `propagate_wdm_signal_constrained()`

- **`receiver_matched.py`**: Matched receiver implementation (Dar 2014)
  - Channel isolation: `extract_center_channel_matched()`
  - Back-propagation: `backpropagate_channel_ssfm()`
  - Detection: `matched_filter_detection()`, `find_optimal_sampling_offset()`
  - Equalization: `train_ls_equalizer()`, `apply_equalizer()`
  - Complete procedure: `matched_receiver_procedure()`

- **`nli_estimation.py`**: NLI noise calculation and analysis
  - Spectral NLI: `estimate_nli_noise_from_spectrum()`
  - Results summary: `summarize_nli_results()`

### Main Scripts

- **`nli.py`**: Main simulation using matched receiver
  - Uses the modular components above
  - Performs Monte Carlo trials
  - Generates constellation plots and NLI statistics

- Use `nli.py` for analysis and simulation runs.
