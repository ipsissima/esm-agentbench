# Numerical Thresholds Configuration

This document describes the configurable numerical thresholds used in ESM-AgentBench certificate generation and provides rationale and sensitivity guidance for each parameter.

## Overview

All numerical thresholds are exposed as configuration parameters that can be set via environment variables or programmatically through the `CertificateConfig` class. This ensures that critical numerical decisions are auditable, configurable, and testable.

## Configuration Parameters

### 1. Witness Condition Number Threshold

**Parameter:** `witness_condition_number_threshold`  
**Default:** `1e8`  
**Environment Variable:** `WITNESS_COND_THRESHOLD`  
**Type:** float (>= 1.0)

**Purpose:**  
Maximum allowed condition number for witness matrices (X0, X1, A) used in kernel verification.

**Rationale:**  
Condition numbers exceeding 1e8 indicate near-singularity, where numerical errors can dominate the computation. This threshold provides a conservative guard against ill-conditioned systems that may produce unreliable bounds.

**Mathematical Basis:**  
For a matrix with condition number κ, relative errors in the solution can be amplified by up to κ. With double-precision arithmetic (machine epsilon ≈ 1e-16), a condition number of 1e8 leaves 8 digits of precision, which is sufficient for reliable bound computation.

**Sensitivity:**  
- **Lower (e.g., 1e6):** More conservative; rejects more witnesses but ensures higher numerical reliability
- **Higher (e.g., 1e10):** More permissive; accepts more witnesses but risks numerical instability
- **Recommended Range:** 1e6 to 1e9

---

### 2. Witness Gap Threshold

**Parameter:** `witness_gap_threshold`  
**Default:** `1e-6`  
**Environment Variable:** `WITNESS_GAP_THRESHOLD`  
**Type:** float (>= 0.0)

**Purpose:**  
Minimum singular value gap for witness validation, ensuring subspaces are well-separated.

**Rationale:**  
Wedin's Theorem (which underlies our spectral stability guarantees) requires singular values to be well-separated for reliable subspace identification under perturbation. Gaps smaller than 1e-6 indicate near-degenerate subspaces that may not remain distinct under numerical noise.

**Mathematical Basis:**  
Wedin's bound for subspace perturbation is inversely proportional to the singular value gap. A gap of 1e-6 ensures that perturbations up to 1e-8 (typical numerical noise) do not collapse the subspace structure.

**Sensitivity:**  
- **Lower (e.g., 1e-8):** More permissive; allows tighter clusters but risks instability
- **Higher (e.g., 1e-5):** More conservative; requires larger separation but improves robustness
- **Recommended Range:** 1e-7 to 1e-5

---

### 3. Explained Variance Threshold

**Parameter:** `explained_variance_threshold`  
**Default:** `0.90` (90%)  
**Environment Variable:** `EXPLAINED_VARIANCE_THRESHOLD`  
**Type:** float (0.0 to 1.0)

**Purpose:**  
Minimum explained variance for PCA rank selection (r_eff).

**Rationale:**  
90% explained variance captures dominant dynamics while avoiding overfitting to noise. This threshold balances:
- **Too low (e.g., 50%):** Misses important dynamics, leading to large tail energy
- **Too high (e.g., 99%):** Overfits to noise, making residuals artificially small

**Empirical Validation:**  
Tested on agent trajectories with 128-384 dimensional embeddings:
- Typical r_eff values: 4-12 (for 90% threshold)
- Mean reconstruction error: 0.05-0.15
- Stable across diverse task types (coding, planning, debugging)

**Sensitivity:**  
- **Lower (e.g., 0.85):** Smaller r_eff, faster computation, higher tail energy
- **Higher (e.g., 0.95):** Larger r_eff, slower computation, lower tail energy but risk of noise overfitting
- **Recommended Range:** 0.85 to 0.95

---

### 4. Out-of-Sample Validation K

**Parameter:** `oos_validation_k`  
**Default:** `3`  
**Environment Variable:** `OOS_VALIDATION_K`  
**Type:** int (>= 1)

**Purpose:**  
Number of out-of-sample steps held out for cross-validation residual estimation.

**Rationale:**  
K=3 provides sufficient holdout data for OOS residual estimation while preserving enough training data for operator fitting. For traces with fewer than 12 steps, K is adaptively reduced to `max(1, T // 4)` to ensure adequate training set size.

**Cross-Validation Strategy:**  
For each of the last K steps, we:
1. Fit the temporal operator A on all previous steps
2. Predict the held-out step
3. Accumulate squared prediction errors

This prevents the zero-residual pathology where in-sample fit can be exact.

**Sensitivity:**  
- **Lower (K=1):** Minimal holdout, less reliable OOS estimate
- **Higher (K=5):** More reliable OOS estimate but requires longer traces (T > 20)
- **Recommended Range:** 2 to 5

---

### 5. Out-of-Sample Residual Floor

**Parameter:** `oos_residual_floor`  
**Default:** `0.1`  
**Environment Variable:** `OOS_RESIDUAL_FLOOR`  
**Type:** float (>= 0.0)

**Purpose:**  
Conservative residual floor for degenerate or short traces where OOS estimation is unreliable.

**Rationale:**  
Returning 0.0 residual on traces with T < 4 steps produces overly optimistic certificates. A floor of 0.1 indicates "minimal but uncertain" predictability, preventing zero-residual pathology on traces too short for meaningful cross-validation.

**When Applied:**  
- Traces with T < 4 steps
- Denominator (sum of squared true values) < 1e-12 (near-zero energy)
- Numerical failures in ridge regression (both `solve()` and `lstsq()` fail)

**Sensitivity:**  
- **Lower (e.g., 0.01):** More optimistic for short traces, risks overconfidence
- **Higher (e.g., 0.5):** More conservative for short traces, may be overly pessimistic
- **Recommended Range:** 0.05 to 0.2

---

## Configuration Examples

### Via Environment Variables

```bash
# Conservative settings (high reliability, stricter validation)
export WITNESS_COND_THRESHOLD=1e6
export WITNESS_GAP_THRESHOLD=1e-5
export EXPLAINED_VARIANCE_THRESHOLD=0.95
export OOS_VALIDATION_K=5
export OOS_RESIDUAL_FLOOR=0.2

# Permissive settings (accept more traces, faster)
export WITNESS_COND_THRESHOLD=1e10
export WITNESS_GAP_THRESHOLD=1e-7
export EXPLAINED_VARIANCE_THRESHOLD=0.85
export OOS_VALIDATION_K=2
export OOS_RESIDUAL_FLOOR=0.05

# Default (balanced)
# Leave unset to use defaults
```

### Programmatic Configuration

```python
from esmassessor.config import CertificateConfig, reset_config

# Create custom config
config = CertificateConfig(
    witness_condition_number_threshold=1e7,
    witness_gap_threshold=1e-6,
    explained_variance_threshold=0.90,
    oos_validation_k=3,
    oos_residual_floor=0.1,
)

# Use custom config
# (Set environment variables or pass to certificate generation)
```

---

## Sensitivity Testing

To test the impact of threshold changes, use the provided sensitivity test script:

```bash
# Run sensitivity tests for all thresholds
python tests/test_threshold_sensitivity.py

# Run for specific parameter
python tests/test_threshold_sensitivity.py --param oos_residual_floor --values 0.01,0.05,0.1,0.2,0.5
```

The script generates:
- Certificate bounds for each parameter value
- Acceptance rate (fraction of traces passing validation)
- Mean bound change relative to defaults

---

## Recommendations for Different Use Cases

### Production Deployment (High Reliability)

```bash
export WITNESS_COND_THRESHOLD=1e7
export WITNESS_GAP_THRESHOLD=1e-6
export EXPLAINED_VARIANCE_THRESHOLD=0.92
export OOS_VALIDATION_K=3
export OOS_RESIDUAL_FLOOR=0.15
```

### Research / Experimentation (Balanced)

Use defaults (no environment variables set).

### Quick Prototyping (Permissive)

```bash
export WITNESS_COND_THRESHOLD=1e9
export WITNESS_GAP_THRESHOLD=1e-7
export EXPLAINED_VARIANCE_THRESHOLD=0.85
export OOS_VALIDATION_K=2
export OOS_RESIDUAL_FLOOR=0.05
```

---

## Audit Trail

All threshold decisions are logged at INFO level with the following format:

```
INFO: Using OOS validation with K=3, residual_floor=0.1 (configured)
INFO: PCA rank r_eff=8 selected (90.5% explained variance, threshold=0.90)
INFO: Witness condition number κ(X0)=4.2e6 < threshold=1e8 (PASS)
```

This ensures full auditability of certificate generation parameters.

---

## References

1. **Wedin's Theorem**: G. W. Stewart and J.-G. Sun, *Matrix Perturbation Theory*, Academic Press, 1990.
2. **Condition Number Analysis**: N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, SIAM, 2002.
3. **Cross-Validation Theory**: T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning*, Springer, 2009.

---

## Version History

- **v1.0** (2026-02-01): Initial configuration exposure with documented rationale
- Future: Add automated sensitivity studies and adaptive threshold selection
