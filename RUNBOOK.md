# Runbook

## Verified kernel artifact flow

The verified kernel artifact (`.kernel_out/kernel_verified.so`) is produced in CI by the `build_verified_kernel` job in:

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
  OPAMROOT=/opt/opam bash -lc "eval $(opam env --switch=esm-kernel) >/dev/null 2>&1 || true; \
    chmod +x ./build_kernel.sh && KERNEL_OUTPUT=/kernel_out/kernel_verified.so ./build_kernel.sh && \
    chown -R $(id -u):$(id -g) /kernel_out || true"
```

Then generate and verify the checksum:

```
sha256sum .kernel_out/kernel_verified.so > .kernel_out/kernel_verified.so.sha256
sha256sum -c .kernel_out/kernel_verified.so.sha256
```

---

## Release signing workflow

The `Release Signing` workflow (`.github/workflows/release.yml`) runs on tag pushes (`v*`) and uses a GPG key stored in `GPG_PRIVATE_KEY` to sign release bundles. Configure the following secrets:

- `GPG_PRIVATE_KEY`: ASCII-armored private key.
- `GPG_KEY_ID`: Key ID or fingerprint to use for signing.

### Key rotation checklist

1. Generate a new keypair and export the armored private key.
2. Update `GPG_PRIVATE_KEY` and `GPG_KEY_ID` in repository secrets.
3. Revoke the old key and archive its public key for verification.
4. Announce the new key fingerprint in release notes.

---

## Quick-start for judges

### One-liner full verification

```
bash ci/verify_submission.sh
```

This script:
1. Builds the verified kernel with Docker Coq 8.18.0
2. Generates and verifies the SHA-256 checksum
3. Builds the judge Docker image
4. Runs judge mode (real agent traces + spectral validation)
5. Validates all 6 scenarios have `attack_succeeded.json` with `success: true`
6. Checks validation reports for `data_source: real_traces_only`
7. Runs pytest guardrails
8. Packages `submission.zip`

### Minimal judge run (Docker only)

```bash
# Build and run judge mode
docker build -t esm-agentbench .
docker run --rm esm-agentbench

# Verify outputs
cat scenarios/code_backdoor_injection/attack_succeeded.json | jq .success
```

### Run specific scenario

```bash
docker run --rm esm-agentbench python /app/run_judge_mode.py --scenario supply_chain_poisoning
```

### Skip kernel build (use existing artifact)

```bash
bash ci/verify_submission.sh --quick
```

---

## Agent card

The agent card is available at:
- TOML: `agent_card.toml`
- JSON: `.well-known/agent-card.json`

To update the entrypoint URL (e.g., for Cloud Run deployment):

```bash
python scripts/update_agent_card.py --url "https://your-service.run.app" --toml agent_card.toml
```
