# Runbook

## Verified kernel artifact flow

The verified kernel artifact (`.kernel_out/kernel_verified.so`) is produced in CI by the `build_verified_kernel` jobs in:

- `.github/workflows/agentbeats_phase1.yml`
- `.github/workflows/ci.yml`

Each job builds the kernel inside `coqorg/coq:8.18.0`, writes a `.kernel_out/kernel_verified.so.sha256` checksum (and optional signature), and uploads the `verified-kernel` artifact for downstream jobs to download and verify. The validation jobs download the artifact, verify its checksum/signature, and set `VERIFIED_KERNEL_PATH` so they do not rebuild the kernel on the judge machines by default.

### Build locally with Docker

From the repository root:

```
mkdir -p .kernel_out
docker run --rm -u "$(id -u):$(id -g)" \
  -v "$(pwd)":/work -v "$(pwd)/.kernel_out":/kernel_out -w /work \
  coqorg/coq:8.18.0 \
  bash -lc "chmod +x ./build_kernel.sh && KERNEL_OUTPUT=/kernel_out/kernel_verified.so ./build_kernel.sh"
```

This produces `.kernel_out/kernel_verified.so` and you can generate a checksum with:

```
sha256sum .kernel_out/kernel_verified.so > .kernel_out/kernel_verified.so.sha256
```

### Use the prebuilt artifact

If you download the `verified-kernel` artifact (e.g., from CI), place it under `UELAT/` and set `VERIFIED_KERNEL_PATH` to avoid rebuilding:

```
export VERIFIED_KERNEL_PATH="$PWD/UELAT/kernel_verified.so"
sha256sum -c UELAT/kernel_verified.so.sha256
```

If a signature is present, verify it with:

```
python3 -m pip install --no-cache-dir PyNaCl
python3 tools/verify_signature.py \
  --index UELAT/kernel_verified.so.sha256 \
  --sig UELAT/kernel_verified.so.sha256.sig \
  --pubkey UELAT/public.key
```

### Rebuild from source (judges can reproduce)

Judges can reproduce the kernel build from sources using the Docker builder above (or `dev-tools/Dockerfile.kernel` if preferred). If rebuilding locally, keep the same pinned Coq image and set `KERNEL_OUTPUT` as shown to generate `kernel_verified.so`, then provide it via `VERIFIED_KERNEL_PATH`.

### Reproducible builder (dev-tools/Dockerfile.kernel)

Build the pinned builder image once:

```
docker build -t esm-kernel-builder -f dev-tools/Dockerfile.kernel .
```

Run the build inside the image and write outputs to `.kernel_out`:

```
mkdir -p .kernel_out
docker run --rm \
  -v "$PWD":/work -v "$PWD/.kernel_out":/kernel_out -w /work \
  esm-kernel-builder \
  bash -lc "eval $(opam env --switch=esm-kernel) >/dev/null 2>&1 || true; \
    chmod +x ./build_kernel.sh && KERNEL_OUTPUT=/kernel_out/kernel_verified.so ./build_kernel.sh && \
    chown -R $(id -u):$(id -g) /kernel_out || true"
```

Then generate and verify the checksum:

```
sha256sum .kernel_out/kernel_verified.so > .kernel_out/kernel_verified.so.sha256
sha256sum -c .kernel_out/kernel_verified.so.sha256
```
