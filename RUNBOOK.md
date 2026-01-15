# Runbook

## Verified kernel artifact flow

The verified kernel artifact (`UELAT/kernel_verified.so`) is produced in CI by the `build_verified_kernel` jobs in:

- `.github/workflows/agentbeats_phase1.yml`
- `.github/workflows/ci.yml`

Each job builds the kernel inside `coqorg/coq:8.18.0`, writes a `UELAT/kernel_verified.so.sha256` checksum, and uploads the `verified-kernel` artifact for downstream jobs to download and verify.

### Build locally with Docker

From the repository root:

```
docker run --rm -v $(pwd):/work -w /work \
  coqorg/coq:8.18.0 \
  bash -lc "bash ./build_kernel.sh"
```

This produces `UELAT/kernel_verified.so` and you can generate a checksum with:

```
sha256sum UELAT/kernel_verified.so > UELAT/kernel_verified.so.sha256
```
