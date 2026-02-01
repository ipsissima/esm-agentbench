# Numerical Contract: Verified Kernel ↔ Certificate Pipeline

This document specifies the exact JSON schemas and contracts between the Python
certificate pipeline and the verified numeric kernel. The kernel computes
interval-bounded values for spectral certificate metrics.

## Overview

The verification pipeline consists of:

1. **Python Pipeline**: Computes witness matrices (SVD, Koopman operator)
2. **Verified Kernel**: Computes interval bounds on certificate values
3. **Bundle**: Packages all artifacts with cryptographic signatures

## Kernel Input JSON Schema (`kernel_input.json`)

The kernel input contains all data needed for verified computation:

```json
{
  "schema_version": "1.0",
  "trace_id": "<string: SHA256 hash of trace>",
  "metadata": {
    "embedder_id": "<string: embedding model identifier>",
    "timestamp": "<string: ISO-8601 timestamp>"
  },
  "parameters": {
    "rank": "<int: target SVD rank>",
    "precision_bits": "<int: interval arithmetic precision, e.g. 128, 160, 256>",
    "kernel_mode": "<string: 'prototype' | 'arb' | 'mpfi'>"
  },
  "observables": {
    "X_aug": {
      "rows": "<int: number of timesteps T>",
      "cols": "<int: embedding dimension D+1>",
      "dtype": "float64",
      "data_matrix": "<string: base64-encoded big-endian float64, row-major>",
      "sha256": "<string: hash of raw matrix bytes for integrity>",
      "description": "<string: human-readable description>"
    }
  },
  "koopman_fit": {
    "A_precompute": {
      "rows": "<int>",
      "cols": "<int>",
      "dtype": "float64",
      "data_matrix": "<string: base64>",
      "sha256": "<string>"
    }
  },
  "external_subspace": null
}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | string | Schema version for compatibility |
| `trace_id` | string | Unique identifier (SHA256 of trace) |
| `metadata.embedder_id` | string | Canonical embedding model ID |
| `metadata.timestamp` | string | ISO-8601 creation timestamp |
| `parameters.rank` | int | Target rank for SVD truncation |
| `parameters.precision_bits` | int | Interval arithmetic precision |
| `parameters.kernel_mode` | string | Kernel implementation to use |
| `observables.X_aug` | object | Augmented trajectory matrix |
| `koopman_fit` | object | Pre-computed operator (optional) |
| `external_subspace` | object | External comparison matrix (optional) |

### Matrix Encoding

Matrices are encoded as:
- **Format**: Base64-encoded binary
- **Byte Order**: Big-endian (`>f8`)
- **Layout**: Row-major (C order)
- **Integrity**: SHA256 hash of raw bytes

Example Python encoding:
```python
import base64
import numpy as np

def encode_matrix(M):
    M = np.asarray(M, dtype='>f8')  # Big-endian float64
    raw_bytes = M.flatten().tobytes()
    return base64.b64encode(raw_bytes).decode('ascii')
```

## Kernel Output JSON Schema (`kernel_output.json`)

The kernel output contains interval-bounded computed values:

```json
{
  "schema_version": "1.0",
  "trace_id": "<string>",
  "kernel_id": "<string: git hash or version>",
  "precision_bits": "<int>",
  "computed": {
    "sigma": [["<low>", "<high>"], ...],
    "gamma": ["<low>", "<high>"],
    "tail_energy": ["<low>", "<high>"],
    "pca_explained": ["<low>", "<high>"],
    "koopman": {
      "A": null,
      "koopman_sigma": [["<low>", "<high>"], ...],
      "koopman_singular_gap": ["<low>", "<high>"]
    },
    "residuals": {
      "insample_residual": ["<low>", "<high>"],
      "oos_residual": ["<low>", "<high>"],
      "r_t_intervals": [[["<low>", "<high>"], ...]]
    },
    "per_step": {
      "off_ratio": [["<low>", "<high>"], ...],
      "r_norm": [["<low>", "<high>"], ...]
    },
    "E_norm": ["<low>", "<high>"],
    "sinTheta": {
      "frobenius": ["<low>", "<high>"],
      "max_sin": ["<low>", "<high>"]
    }
  },
  "checks": {
    "theoretical_bound": {
      "lhs_interval": ["<low>", "<high>"],
      "tau": "<float: threshold>",
      "pass": "<bool>"
    },
    "wedin_bound": {
      "E_over_gamma": "<float>",
      "pass_estimate": "<bool>"
    }
  },
  "provenance": {
    "input_hash": "<string: SHA256 of input JSON>",
    "kernel_binary_hash": "<string: SHA256 of kernel binary>",
    "runtime": {
      "container": "<string: Docker image reference>",
      "mp_precision": "<int: actual precision used>"
    }
  },
  "signature": "<string: GPG signature or null>"
}
```

### Interval Format

All interval values are represented as two-element arrays of strings:
```json
["<lower_bound>", "<upper_bound>"]
```

Using strings preserves arbitrary precision. Parsers should convert to appropriate
numeric types (e.g., `mpmath.mpf` in Python, `Arb` in C).

### Computed Fields

| Field | Type | Description |
|-------|------|-------------|
| `sigma` | array | Singular value intervals `[σ₁, σ₂, ...]` |
| `gamma` | interval | Singular gap `σᵣ - σᵣ₊₁` |
| `tail_energy` | interval | `1 - (Σσᵢ²)/(Σσⱼ²)` for `i≤r` |
| `pca_explained` | interval | Fraction of variance retained |
| `koopman.koopman_sigma` | array | Singular values of Koopman operator |
| `koopman.koopman_singular_gap` | interval | Gap in Koopman spectrum |
| `residuals.insample_residual` | interval | Training fit error |
| `residuals.oos_residual` | interval | Out-of-sample prediction error |
| `residuals.r_t_intervals` | array | Per-step residual norms |
| `per_step.off_ratio` | array | Per-step off-manifold ratios |
| `per_step.r_norm` | array | Per-step residual norms |
| `E_norm` | interval | Operator perturbation `‖A - Ã‖` |
| `sinTheta.frobenius` | interval | Frobenius norm of sin(angles) |
| `sinTheta.max_sin` | interval | Maximum subspace angle sine |

### Check Fields

| Field | Type | Description |
|-------|------|-------------|
| `theoretical_bound.lhs_interval` | interval | Computed bound value |
| `theoretical_bound.tau` | float | Threshold for passing |
| `theoretical_bound.pass` | bool | `lhs ≤ tau` |
| `wedin_bound.E_over_gamma` | float | `‖E‖/γ` ratio for Wedin |
| `wedin_bound.pass_estimate` | bool | Wedin bound satisfied |

## Theoretical Bound Formula

The theoretical bound is computed as:

```
bound = C_res × residual + C_tail × tail_energy + C_sem × semantic_divergence + C_robust × lipschitz_margin
```

Where `C_res`, `C_tail`, `C_sem`, `C_robust` are verified constants from Coq proofs.

### Default Constants (from UELAT verification)

| Constant | Value | Description |
|----------|-------|-------------|
| C_res | 1.0 | Residual coefficient |
| C_tail | 1.0 | Tail energy coefficient |
| C_sem | 1.0 | Semantic divergence coefficient |
| C_robust | 1.0 | Robustness coefficient |

## Kernel Implementation Requirements

The verified kernel must:

1. **Read kernel_input.json** from specified path
2. **Validate integrity** using SHA256 hashes
3. **Compute intervals** at specified precision
4. **Write kernel_output.json** to specified path
5. **Return exit code**:
   - `0`: Success, all checks pass
   - `1`: Success, but some checks fail
   - `2`: Error during computation

## Example Usage

### Python Pipeline

```python
from certificates.make_certificate import export_kernel_input, compute_certificate
from certificates.kernel_client import run_kernel, verify_kernel_output

# Compute certificate and export kernel input
cert = compute_certificate(embeddings, r=10)
export_kernel_input(
    X_aug=X_aug,
    trace_id=trace_hash,
    output_path="/tmp/kernel_input.json",
    precision_bits=160,
)

# Run verified kernel
output = run_kernel("/tmp/kernel_input.json", mode="prototype")

# Verify checks pass
if verify_kernel_output(output):
    print("Certificate verified!")
```

### Command Line

```bash
# Export kernel input
python -m certificates.make_certificate \
    --trace-file trace.json \
    --export-kernel-input /tmp/kernel_input.json

# Run prototype kernel
python kernel/prototype/prototype_kernel.py \
    /tmp/kernel_input.json \
    /tmp/kernel_output.json \
    --precision 160

# Check output
jq '.checks.theoretical_bound.pass' /tmp/kernel_output.json
```

### Docker Kernel

```bash
# Run production kernel
docker run --rm \
    -v /tmp/kernel_input.json:/data/kernel_input.json:ro \
    -v /tmp/kernel_output.json:/data/kernel_output.json:rw \
    -e PRECISION_BITS=256 \
    ipsissima/kernel:latest

# Verify output
jq '.checks' /tmp/kernel_output.json
```

## Reproducibility

For reproducible verification:

1. **Pin kernel version**: Use specific Docker image SHA
2. **Specify precision**: Higher precision = tighter intervals
3. **Record provenance**: All hashes stored in output
4. **Sign artifacts**: GPG signature on bundle

## Security Considerations

- Input validation prevents injection attacks
- SHA256 integrity checks detect tampering
- GPG signatures provide non-repudiation
- Container isolation limits kernel access
