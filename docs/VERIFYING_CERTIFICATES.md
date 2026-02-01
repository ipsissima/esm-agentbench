# Verifying Spectral Certificates

This guide explains how to verify spectral certificates using the verified
numeric kernel. Verification ensures that certificate values are computed
correctly using interval arithmetic.

## Quick Start

### 1. Verify with Prototype Kernel (Development)

```bash
# Install dependencies
pip install mpmath numpy scipy

# Run certificate computation with kernel export
python -c "
from certificates.make_certificate import compute_certificate, export_kernel_input
import numpy as np

# Generate sample embeddings
X = np.random.randn(20, 128)
X_aug = np.concatenate([X, np.ones((20, 1))], axis=1)

# Export kernel input
export_kernel_input(
    X_aug=X_aug,
    trace_id='test',
    output_path='/tmp/kernel_input.json',
    precision_bits=160,
)
"

# Run prototype kernel
python kernel/prototype/prototype_kernel.py \
    /tmp/kernel_input.json \
    /tmp/kernel_output.json \
    --precision 160

# Check result
jq '.checks.theoretical_bound' /tmp/kernel_output.json
```

### 2. Verify with Production Kernel (Docker)

```bash
# Build the kernel image
./scripts/build_verified_kernel.sh

# Run verification
docker run --rm \
    -v /tmp/kernel_input.json:/data/kernel_input.json:ro \
    -v /tmp/kernel_output.json:/data/kernel_output.json:rw \
    ipsissima/kernel:latest

# Verify output
jq '.checks' /tmp/kernel_output.json
```

## Verification Steps

### Step 1: Obtain Certificate Bundle

Certificate bundles contain all artifacts needed for verification:

```
bundle/
├── trace.json           # Original trace data
├── kernel_input.json    # Kernel input
├── kernel_output.json   # Kernel output (verified)
├── certificate.json     # Spectral certificate
├── metadata.json        # Bundle metadata
└── signature.asc        # GPG signature
```

### Step 2: Verify Bundle Integrity

```bash
# Extract bundle
tar xzf bundle.tar.gz

# Verify GPG signature
gpg --verify bundle/signature.asc bundle/metadata.json

# Check file hashes
python -c "
from certificates.cert_bundle import verify_bundle
result = verify_bundle('bundle/')
print('Valid:', result['valid'])
print('Checks:', result['checks'])
"
```

### Step 3: Re-run Kernel Verification

```bash
# Re-run kernel to verify computed values match
python kernel/prototype/prototype_kernel.py \
    bundle/kernel_input.json \
    /tmp/recomputed_output.json \
    --precision 160

# Compare outputs
diff <(jq -S '.computed' bundle/kernel_output.json) \
     <(jq -S '.computed' /tmp/recomputed_output.json)
```

### Step 4: Verify Theoretical Bound

The theoretical bound should satisfy:

```
bound = C_res × residual + C_tail × tail_energy + C_sem × semantic_divergence
bound ≤ τ (threshold)
```

```python
import json

with open('bundle/kernel_output.json') as f:
    output = json.load(f)

bound = output['checks']['theoretical_bound']
print(f"Computed bound: {bound['lhs_interval']}")
print(f"Threshold τ: {bound['tau']}")
print(f"Pass: {bound['pass']}")
```

## Building the Verified Kernel

### Build Prototype (Python)

No build needed - uses `kernel/prototype/prototype_kernel.py` directly.

Requirements:
- Python 3.8+
- numpy
- scipy
- mpmath (optional, for interval arithmetic)

### Build Production Kernel (Docker)

```bash
# Build reproducible Docker image
./scripts/build_verified_kernel.sh

# Verify image hash
docker inspect ipsissima/kernel:latest --format='{{.Id}}'
```

The production kernel uses:
- ARB library for certified interval arithmetic
- FLINT for fast number theory operations
- Reproducible Debian base image

## Precision and Accuracy

### Precision Levels

| Bits | Use Case | Notes |
|------|----------|-------|
| 64 | Quick checks | Standard double precision |
| 128 | Default | Good for most certificates |
| 160 | High precision | Tighter intervals |
| 256 | Research | Very tight bounds |
| 512+ | Formal proofs | Maximum rigor |

### Interval Interpretation

Kernel outputs intervals `[low, high]` that provably contain the true value:

```python
# Example: residual interval
residual = ["0.0123", "0.0125"]
# True residual is guaranteed to be in [0.0123, 0.0125]
```

## Troubleshooting

### Kernel Not Found

```bash
# Check prototype kernel location
ls -la kernel/prototype/prototype_kernel.py

# Set environment variable for custom location
export ESM_KERNEL_LOCAL_PY=/path/to/prototype_kernel.py
```

### Docker Image Not Available

```bash
# Build locally
cd kernel/arb_kernel
docker build -t ipsissima/kernel:latest .
```

### Hash Mismatch

If integrity check fails:

1. Verify the source file wasn't modified
2. Check encoding (must be big-endian float64)
3. Re-export with correct parameters

```python
from certificates.make_certificate import export_kernel_input
import hashlib
import numpy as np

# Recompute hash
X_aug = ...  # Your matrix
computed_hash = hashlib.sha256(X_aug.tobytes()).hexdigest()
print(f"Computed hash: {computed_hash}")
```

### Signature Verification Failed

```bash
# Import signing key
gpg --import public-key.asc

# Check key trust
gpg --list-keys

# Verify with verbose output
gpg --verify -v bundle/signature.asc bundle/metadata.json
```

## Programmatic Verification

### Python API

```python
from certificates.kernel_client import run_kernel_and_verify
from certificates.cert_bundle import verify_bundle

# Run and verify kernel
try:
    output = run_kernel_and_verify(
        "kernel_input.json",
        precision_bits=160,
        mode="prototype",
    )
    print("Verification passed!")
except Exception as e:
    print(f"Verification failed: {e}")

# Verify bundle
result = verify_bundle("bundle/")
if result["valid"]:
    print("Bundle integrity verified")
else:
    print(f"Errors: {result['errors']}")
```

### CI Integration

```yaml
# .github/workflows/verify.yml
- name: Verify certificate
  run: |
    python -m certificates.kernel_client \
      --input ${{ env.KERNEL_INPUT }} \
      --verify
```

## Security Model

### What is Verified

- **Interval bounds**: Computed values provably contain true values
- **Integrity**: SHA256 hashes detect tampering
- **Authenticity**: GPG signatures verify origin

### What is NOT Verified

- **Embedding quality**: Kernel trusts input embeddings
- **Model correctness**: Assumes embedding model works correctly
- **Hardware**: Assumes no hardware faults

### Trust Assumptions

1. The verified kernel binary is correct
2. The hardware performs arithmetic correctly
3. No adversary can forge GPG signatures
4. SHA256 is collision-resistant

## Further Reading

- [NUMERICAL_CONTRACT.md](NUMERICAL_CONTRACT.md) - JSON schema specification
- [certificates/make_certificate.py](../certificates/make_certificate.py) - Certificate computation
- [kernel/prototype/prototype_kernel.py](../kernel/prototype/prototype_kernel.py) - Prototype implementation
