# Runbook

## Verified kernel artifact flow

The verified kernel artifact (`.kernel_out/kernel_verified.so`) is produced in CI by the `build_verified_kernel` jobs in:

- `.github/workflows/agentbeats_phase1.yml`
- `.github/workflows/ci.yml`

Each job builds the kernel inside `coqorg/coq:8.18.0`, writes a `.kernel_out/kernel_verified.so.sha256` checksum, and uploads the `verified-kernel` artifact for downstream jobs to download and verify.

### Build locally with Docker

From the repository root:

```
mkdir -p .kernel_out
docker run --rm -v $(pwd):/work -v $(pwd)/.kernel_out:/kernel_out -w /work \
  coqorg/coq:8.18.0 \
  bash -lc "KERNEL_OUTPUT=/kernel_out/kernel_verified.so bash ./build_kernel.sh"
```

This produces `.kernel_out/kernel_verified.so` and you can generate a checksum with:

```
sha256sum .kernel_out/kernel_verified.so > .kernel_out/kernel_verified.so.sha256
```
