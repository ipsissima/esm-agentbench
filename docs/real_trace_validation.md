# Real Trace Validation

This document describes the real-trace validation pipeline used to evaluate the spectral drift metrics on real execution traces.

## Summary of the math

The real-trace validator combines normalized spectral statistics with prompt-aware correction and relative-rank starvation detection:

- **Residual normalization (`R_norm`)**: For an embedding matrix `X` (T steps × d dims), compute the singular values of the centered matrix and choose `r0` as the smallest rank that explains 90% of the energy. The tail-energy ratio yields `R_norm = sqrt(tail / total)`, which is bootstrapped over coherent traces to estimate a null threshold.
- **Relative rank (`r_rel`)**: The effective rank `r_eff` normalized by the number of steps (`n_steps`) is used to detect starvation (very low rank relative to trace length). We evaluate this with a Mann–Whitney test between coherent vs starvation traces.
- **Prompt semantic correction**: The centroid of a trace’s embeddings is compared to the prompt embedding (cosine distance). We adjust the normalized bound via `theoretical_bound_prompt_adj = theoretical_bound_norm - gamma * d_sem` to avoid penalizing creative-but-correct traces that deviate lexically from the prompt.
- **Spike feature (`sv_max_ratio`)**: The largest singular-value ratio, `max(sigma_i^2)/sum(sigma_i^2)`, is used as a poison indicator (spikes indicate concentrated subspace dynamics).

The spectral theory is grounded by MP/BBP intuition and Davis–Kahan stability bounds: in practice, the tail energy (`R_norm`) tracks drift, and the relative rank (`r_rel`) flags starvation or memory truncation artifacts.

## Usage

Run real-trace validation with prompt adjustment and bootstrap calibration:

```bash
python tools/validate_real_traces.py --use-prompt-adj --gamma 0.3 --bootstrap-B 500
```

Tune metrics with a gamma grid in nested CV:

```bash
python tools/tune_metric.py --features-csv reports/features.csv --gamma-grid 0.0 0.1 0.2 0.3 0.4 --use-prompt-adj --n-boot 50
```

## Hyperparameters

- **`gamma`** (default `0.3`): prompt-adjustment shrinkage. Larger values emphasize semantic distance more strongly.
- **`bootstrap-B`** (default `500`): bootstrap iterations for `R_norm` null threshold.
- **`sv_max_ratio` cutoff** (default `0.6`): indicates poison-like spike behavior.

## Verified kernel notes

The validated kernel is optional and can be disabled by setting:

```bash
export ESM_SKIP_VERIFIED_KERNEL=1
```

If the verified kernel is unavailable, a one-time warning is emitted and the Python fallback is used.
