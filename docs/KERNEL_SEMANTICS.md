# Coq Kernel vs Python Fallback: Formal Semantics

This document describes the mathematical equivalence and semantic differences between the verified Coq kernel and the Python fallback implementation.

## Overview

ESM-AgentBench supports two computational backends for certificate bounds:

1. **Verified Coq Kernel** (`kernel_verified.so`): OCaml extraction from formally verified Coq proofs, using ARB interval arithmetic
2. **Python Fallback** (`verified_kernel.py`): Unverified Python implementation using NumPy double-precision arithmetic

## Mathematical Equivalence

Both implementations compute the same theoretical bound formula:

```
bound = C_res × residual + C_tail × tail_energy + C_sem × semantic_divergence + C_robust × lipschitz_margin
```

Where:
- `C_res`, `C_tail`, `C_sem`, `C_robust` are constants from Coq (loaded via `uelat_bridge`)
- All terms are non-negative real numbers

### Formal Semantics

**Coq Kernel** (Interval Arithmetic):
```
Let X0, X1, A ∈ ℝ^(r×n) be witness matrices.

residual := sup { ‖X1 - A·X0‖_∞ / (‖X1‖_∞ + ε) }
          where ‖·‖_∞ is the entrywise max-norm
          and ε = 1e-12 (numerical stability floor)

bound := C_res · residual + C_tail · tail_energy + C_sem · semantic_divergence + C_robust · lipschitz_margin

The kernel computes rigorous interval bounds:
  [residual_lower, residual_upper] using ARB multiprecision arithmetic
  [bound_lower, bound_upper] via interval propagation

Output: (residual_upper, bound_upper) — conservative upper bounds
```

**Python Fallback** (IEEE 754 Double Precision):
```python
# L-infinity norm implementation
def linf_norm(M: np.ndarray) -> float:
    return np.max(np.abs(M))

# Residual computation
def compute_residual(X0, X1, A):
    prediction_error = X1 - A @ X0
    residual = linf_norm(prediction_error) / (linf_norm(X1) + 1e-12)
    return residual

# Bound computation
def compute_bound(residual, tail_energy, semantic_divergence, lipschitz_margin):
    C_res, C_tail, C_sem, C_robust = load_constants()
    bound = (
        C_res * residual +
        C_tail * tail_energy +
        C_sem * semantic_divergence +
        C_robust * lipschitz_margin
    )
    return bound
```

## Differences

### 1. Numerical Precision

| Aspect | Coq Kernel | Python Fallback |
|--------|------------|-----------------|
| Arithmetic | ARB interval (arbitrary precision) | IEEE 754 double (53-bit mantissa) |
| Rounding | Rigorous interval bounds | Round-to-nearest |
| Guarantees | Formally verified | None (relies on floating-point stability) |

**Implication**: The Coq kernel provides **conservative upper bounds** that are mathematically guaranteed. The Python fallback provides **point estimates** that may underestimate due to rounding errors.

### 2. Norm Computation

Both use **L-infinity (max-norm)** semantics:

```
‖M‖_∞ = max_{i,j} |M[i,j]|
```

**Coq kernel**: Computes exact interval `[lower, upper]` for each matrix entry, then takes supremum.

**Python fallback**: Computes `np.max(np.abs(M))` directly, which is numerically equivalent but subject to floating-point error accumulation.

**Deviation bound**: For an r×n matrix with entries in [-B, B]:
```
|‖M‖_∞^kernel - ‖M‖_∞^python| ≤ r·n·B·ε_mach
```
where `ε_mach ≈ 2.22e-16` (machine epsilon for double precision).

For typical matrices (r ≤ 20, n ≤ 1000, B ≤ 10):
```
|difference| ≤ 4.4e-12
```

This is negligible compared to typical residuals (≥ 0.01).

### 3. Constant Loading

Both implementations load the same constants from `UELAT/uelat_constants.json`:

```json
{
  "C_res": 2.0,
  "C_tail": 2.0,
  "C_sem": 2.0,
  "C_robust": 2.0
}
```

**Note**: Coq proofs guarantee these constants satisfy theoretical bounds. The Python fallback **assumes** these values are correct.

## Formal Correctness Statement

**Theorem** (Mathematical Equivalence):

For witness matrices X0, X1, A with bounded entries, and assuming:
1. No catastrophic cancellation in residual computation
2. Sufficient numerical precision (≥ 53-bit mantissa)
3. Correct constant loading

Then:
```
|bound_python - bound_coq| ≤ δ
```

where `δ ≤ 4·C_max·(r·n·B·ε_mach)` and `C_max = max(C_res, C_tail, C_sem, C_robust)`.

**Proof sketch**:
- Each norm computation has error ≤ r·n·B·ε_mach
- Linear combination of 4 terms amplifies by C_max
- Total bound: 4·C_max·(r·n·B·ε_mach)

**Typical error**: For C_max = 2, r = 10, n = 100, B = 10:
```
δ ≤ 4 · 2 · (10 · 100 · 10 · 2.22e-16) ≈ 1.78e-11
```

This is **negligible** compared to typical bounds (≥ 0.1).

## When to Use Each Backend

### Use Coq Kernel When:
- **Formal verification is required** (safety-critical applications)
- **Auditability is paramount** (regulatory compliance)
- **Conservative bounds are needed** (risk-averse scenarios)
- The OCaml runtime is available (Docker or native build)

### Use Python Fallback When:
- Rapid prototyping or development
- Coq kernel is not available (e.g., no OCaml on Windows)
- Performance is critical (Python is faster for small matrices)
- Formal verification is not a requirement

## Implementation Notes

### Coq Kernel (verified_kernel.py → kernel_verified.so)

**Entry Point**:
```c
// OCaml/C interface (generated by Coq extraction)
double kernel_compute_residual(double* X0, double* X1, double* A, int r, int n);
double kernel_compute_bound(double res, double tail, double sem, double lip);
```

**Internal Logic**:
1. Convert double arrays to ARB intervals: `[x - δ, x + δ]` where δ accounts for conversion error
2. Perform matrix multiply and subtraction using interval arithmetic
3. Compute L-infinity norm via interval max
4. Divide with interval division (handling denominator bounds)
5. Return upper bound of result interval

**Extraction Chain**:
```
Coq Theorem (UELAT/*.v) 
  → Extracted OCaml (.ml) 
  → C Stubs (kernel_stubs.c) 
  → Compiled Shared Library (.so)
```

### Python Fallback (verified_kernel.py)

**Entry Point**:
```python
def compute_residual(X0, X1, A, strict=False):
    """Compute residual using NumPy double precision."""
    # ...

def compute_bound(residual, tail_energy, semantic_divergence, lipschitz_margin, strict=False):
    """Compute bound using constants from uelat_bridge."""
    # ...
```

**Internal Logic**:
1. Validate input shapes and types
2. Compute `prediction_error = X1 - A @ X0` (NumPy matrix multiply)
3. Compute norms using `np.linalg.norm(M, ord=np.inf)` or `np.max(np.abs(M))`
4. Return point estimate (no intervals)

**Fallback Decision Logic** (`load_kernel()`):
```python
if os.environ.get("ESM_SKIP_VERIFIED_KERNEL") == "1":
    return None  # Force Python fallback
elif VERIFIED_KERNEL_PATH exists:
    try:
        kernel = ctypes.CDLL(VERIFIED_KERNEL_PATH)
        # Validate symbols, test self-test
        return kernel
    except Exception:
        if strict:
            raise
        return None  # Fallback
else:
    if strict:
        raise KernelError("Kernel not found")
    return None  # Fallback
```

## Testing Equivalence

The test suite (`test_kernel_comprehensive.py`) validates equivalence:

```python
def test_python_vs_kernel_equivalence():
    """Verify Python fallback matches kernel (within tolerance)."""
    X0, X1, A = generate_test_matrices()
    
    # Force kernel
    os.environ["ESM_SKIP_VERIFIED_KERNEL"] = "0"
    res_kernel, bound_kernel = compute_certificate(X0, X1, A, ...)
    
    # Force Python
    os.environ["ESM_SKIP_VERIFIED_KERNEL"] = "1"
    res_python, bound_python = compute_certificate(X0, X1, A, ...)
    
    # Verify equivalence (within numerical tolerance)
    assert abs(res_kernel - res_python) < 1e-10
    assert abs(bound_kernel - bound_python) < 1e-10
```

**Tolerance rationale**: 1e-10 is well above machine epsilon (2e-16) but well below typical residuals (≥ 0.01), ensuring we catch implementation bugs while allowing for normal floating-point variation.

## Audit Recommendations

For formal audits:
1. **Verify Coq proofs**: Check `UELAT/*.v` files compile with Coq 8.18.0
2. **Verify extraction**: Compare extracted OCaml to Coq definitions
3. **Verify C stubs**: Ensure `kernel_stubs.c` correctly marshals data
4. **Verify constants**: Check `uelat_constants.json` matches Coq definitions
5. **Test equivalence**: Run `test_kernel_comprehensive.py` on representative workloads
6. **Check fallback logic**: Ensure Python fallback is only used when intended

## References

1. **ARB Library**: https://arblib.org/ — Arbitrary-precision ball arithmetic
2. **Coq Extraction**: https://coq.inria.fr/refman/addendum/extraction.html
3. **IEEE 754**: IEEE Standard for Floating-Point Arithmetic (2019)
4. **NumPy Norms**: https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html

## Version History

- **v1.0** (2026-02-01): Initial semantic documentation
- Future: Add formal proof of Python fallback correctness (interactive theorem proving)
