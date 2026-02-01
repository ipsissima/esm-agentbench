# Numerical Fallback Logging and Provenance Improvements

This document describes the improvements made to address the immediate priorities from the comprehensive code review.

## Overview

Three key improvements were implemented to enhance numerical stability logging, embedder provenance tracking, and cross-platform compatibility:

1. **Numerical Fallback Logging**: All numerical fallbacks now log diagnostic metadata
2. **Embedder Provenance Checks**: Validates and logs embedder identity for audit trails
3. **Cross-Platform Kernel Checks**: Supports macOS (otool) and documents Windows approach

## 1. Numerical Fallback Logging

### Problem Statement
Silent numerical fallbacks hide failures and produce unverifiable certificates. When numerical instability occurs (e.g., ill-conditioned matrices), the code would silently fall back to alternative algorithms without logging.

### Solution
Added comprehensive logging with diagnostic metadata at all numerical fallback points:

#### Changes in `certificates/make_certificate.py`

##### `_fit_temporal_operator_ridge`
Fits a linear temporal operator with ridge regularization. Logs fallbacks:
- **solve() → lstsq()**: Logs when primary solver fails
- **lstsq() → pseudoinverse**: Logs when secondary solver also fails

**Diagnostic metadata logged:**
- Gram matrix shape
- Condition number (computed upfront)
- Input matrix shapes (X0, X1)
- Error messages

**Example log output:**
```
WARNING - _fit_temporal_operator_ridge: solve() failed, falling back to lstsq. 
gram_reg shape=(5, 5), condition_number=1.23e+10, X0 shape=(5, 20), X1 shape=(5, 20), 
error=Singular matrix
```

##### `_compute_oos_residual`
Computes out-of-sample residual using time-series cross-validation. Logs fallbacks:
- **solve() → lstsq()**: Logs when solver fails during cross-validation
- **lstsq() → high residual (1.0)**: Logs when both solvers fail
- **T < 4**: Logs when trace is too short for meaningful OOS estimation

**Diagnostic metadata logged:**
- Condition number
- Matrix shapes
- Training sample counts
- Error messages

**Example log output:**
```
INFO - _compute_oos_residual: Trace too short (T=3), returning fallback residual=0.0. 
Minimum T=4 required for out-of-sample validation.
```

### Benefits
- CI failures are now visible in logs
- Numerical instability can be diagnosed and debugged
- Facilitates formal auditing of numerical stability
- Supports the formal proof obligations in the kernel contract

## 2. Embedder Provenance Checks

### Problem Statement
Semantic divergence computation depends on the embedding model. Without tracking which embedder was used, certificates cannot be verified to use consistent embeddings.

### Solution
Enhanced `core/certificate.py` with embedder validation and provenance tracking.

#### Changes in `compute_certificate_from_trace`

**When `task_embedding` is provided:**
1. **Validates** that `embedder_id` is also provided
2. **Logs warning** if `embedder_id` is missing
3. **Logs info** about the embedder being used
4. **Adds `embedder_id`** to certificate metadata for audit trail

**Example log outputs:**

*Missing embedder_id (warning):*
```
WARNING - compute_certificate_from_trace: task_embedding provided but embedder_id is missing. 
Semantic divergence depends on the embedding model. Consider requiring embedder_id to match 
a stored canonical id for verifiable certificates.
```

*With embedder_id (info):*
```
INFO - compute_certificate_from_trace: Computing certificate with 
embedder_id='sentence-transformers/all-MiniLM-L6-v2'. 
Embeddings shape=(10, 5), task_embedding shape=(5,)
```

**Certificate metadata:**
```python
{
    "theoretical_bound": 0.234,
    "residual": 0.123,
    ...
    "embedder_id": "sentence-transformers/all-MiniLM-L6-v2"  # Added for audit trail
}
```

### Benefits
- Enables verification that semantic divergence used consistent embeddings
- Creates audit trail for certificate provenance
- Supports future enhancement: matching against stored canonical embedder IDs
- Prevents semantic divergence from being computed with wrong embedder

## 3. Cross-Platform Kernel Dependency Checks

### Problem Statement
Kernel dependency checks were Linux-biased, using only `ldd`. This silently skipped validation on macOS and Windows, potentially missing library dependency issues.

### Solution
Enhanced `certificates/verified_kernel.py` with platform-specific dependency checking.

#### Changes in `_check_library_dependencies`

**Platform detection:**
- **Linux**: Uses `ldd` (existing behavior)
- **macOS**: Uses `otool -L` for .dylib files
- **Windows**: Documents manual verification requirement
- **Unknown platforms**: Logs warning and skips

**New helper functions:**
- `_parse_ldd_output`: Parses Linux ldd output for missing dependencies
- `_parse_otool_output`: Parses macOS otool output for missing dependencies

**Example log outputs:**

*macOS:*
```
INFO - _check_library_dependencies: Using otool -L for macOS dependency checking.
```

*Windows:*
```
WARNING - _check_library_dependencies: Windows platform detected. 
Automatic dependency checking not available. 
Please manually verify kernel dependencies using: 
'dumpbin /dependents <kernel.dll>' or similar tools.
```

*Tool unavailable:*
```
WARNING - _check_library_dependencies: ldd not available on linux. Error: FileNotFoundError
```

### Benefits
- Proper dependency validation on macOS
- Clear documentation for Windows users
- Better diagnostics when tools are unavailable
- Supports the kernel safety constraints documented in verified_kernel.py

## Testing

### New Test Suites

#### `tests/test_numerical_fallback_logging.py` (9 tests)
- Tests numerical fallback logging with mocked failures
- Tests embedder provenance checks and warnings
- Tests certificate metadata inclusion
- Tests ill-conditioned systems

**Key tests:**
- `test_fit_temporal_operator_ridge_logs_lstsq_fallback`: Uses mocking to ensure fallback is logged
- `test_compute_oos_residual_logs_short_trace`: Validates short trace logging
- `test_embedder_provenance_warning_when_missing`: Validates warning when embedder_id missing
- `test_embedder_id_added_to_certificate`: Validates metadata inclusion

#### `tests/test_kernel_dependency_checks.py` (10 tests)
- Tests platform detection (Linux, macOS, Windows, unknown)
- Tests ldd and otool output parsing
- Tests handling of missing tools and dependencies
- Uses mocking for platform-specific behavior

**Key tests:**
- `test_check_library_dependencies_linux`: Validates ldd usage on Linux
- `test_check_library_dependencies_macos`: Validates otool usage on macOS
- `test_check_library_dependencies_windows`: Validates Windows skip message
- `test_parse_ldd_output_with_missing_deps`: Tests dependency detection

### Test Results
All 19 new tests pass. All existing tests continue to pass (50+ tests validated).

## Demonstration

Run the demonstration script to see the logging in action:

```bash
python demo_logging_improvements.py
```

This demonstrates:
1. Short trace logging
2. Normal computation (no fallback)
3. Missing embedder_id warning
4. Embedder provenance logging

## Alignment with Code Review Requirements

This implementation addresses all "Immediate (fixes you should do now)" items:

### ✅ Log numerical fallbacks & diagnostic metadata
- `_fit_temporal_operator_ridge`: Logs lstsq → pseudoinverse fallbacks
- `_compute_oos_residual`: Logs solve → lstsq → high residual fallbacks
- Includes: condition number, matrix shapes, error messages

### ✅ Enforce embedder provenance checks
- Validates embedder_id when task_embedding provided
- Logs warnings for missing embedder_id
- Records embedder_id in certificate metadata

### ✅ Kernel symbol/dependency checks: add macOS/Windows path
- macOS: Uses `otool -L`
- Windows: Documents manual verification
- Platform detection with proper logging

## Future Enhancements

Based on the code review, potential future improvements include:

### Medium Term
- Split `make_certificate.py` into smaller modules
- Replace full SVD with randomized/truncated SVD for large matrices
- Vectorize per-step computations
- Compute/record condition numbers in reports/

### Long Term
- Formalize Python ↔ Kernel proof obligations in acceptance tests
- Expose deterministic, reproducible mode with GPG signing
- Replace ambiguous norm uses with explicit spectral norms

## References

- Problem statement: Comprehensive code review section 7 (Immediate fixes)
- Source files: `certificates/make_certificate.py`, `core/certificate.py`, `certificates/verified_kernel.py`
- Documentation: `docs/NUMERICAL_CONTRACT.md`, `docs/SPECTRAL_THEORY.md`
