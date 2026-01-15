# Building kernel_verified.so (UELAT)

Requirements:
- Ubuntu 22.04 (or similar)
- opam, coq 8.18.0, ocaml >= 4.14.0

Recommended reproducible flow (use dev-tools/Dockerfile.kernel):

```
docker build -t esm-kernel-builder -f dev-tools/Dockerfile.kernel .
mkdir -p .kernel_out
docker run --rm -v $(pwd):/work -v $(pwd)/.kernel_out:/kernel_out -w /work \
  esm-kernel-builder \
  OPAMROOT=/opt/opam bash -lc "eval $(opam env --switch=esm-kernel) >/dev/null 2>&1 || true; \
    chmod +x ./build_kernel.sh && KERNEL_OUTPUT=/kernel_out/kernel_verified.so ./build_kernel.sh && \
    chown -R $(id -u):$(id -g) /kernel_out || true"
```

This will run ./build_kernel.sh in /work and produce .kernel_out/kernel_verified.so on success.

Alternative: use the official Coq 8.18.0 image directly:

```
docker run --rm -v $(pwd):/work -v $(pwd)/.kernel_out:/kernel_out -w /work \
  coqorg/coq:8.18.0 \
  bash -lc "KERNEL_OUTPUT=/kernel_out/kernel_verified.so bash ./build_kernel.sh"
```

If building locally without Docker:
1. sudo apt install opam coq ocaml ocaml-native-compilers ocamlfind build-essential m4
2. OPAMROOT=/opt/opam opam init --bare --disable-sandboxing
3. OPAMROOT=/opt/opam opam switch create esm-kernel 4.14.0
4. OPAMROOT=/opt/opam eval $(opam env)
5. OPAMROOT=/opt/opam opam install coq.8.18.0 dune ocamlfind
6. ./build_kernel.sh

Note: If kernel_verified.so is included in the repo or artifact, CI will prefer it over building from source and verify it using the accompanying .sha256 checksum.

Note: CI produces the verified-kernel artifact in the build_verified_kernel jobs (see RUNBOOK.md).

Note: pyproject.toml is the canonical package metadata; requirements.txt is kept for convenience installs.
