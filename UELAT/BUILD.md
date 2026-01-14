# Verified Kernel Build Guide

This directory contains the Coq sources used to extract and compile the verified
kernel (`kernel_verified.so`). The build is sensitive to the exact Coq/OCaml
toolchain, so use the pinned versions below.

## Pinned toolchain

The verified kernel is built with:

- **OCaml**: 4.14.0
- **Coq**: 8.18.1
- **opam**: 2.x (use the pinned switch `esm-kernel`)

The opam dependencies are captured in [`UELAT/opam`](opam).

## Build with opam (local)

```bash
opam init --bare --disable-sandboxing
opam switch create esm-kernel 4.14.0 || opam switch set esm-kernel
eval "$(opam env)"
opam install -y coq.8.18.1 ocamlfind dune
./build_kernel.sh
```

The build script writes `UELAT/kernel_verified.so` when successful.

## Build with Docker

We provide a pinned Docker environment under `dev-tools/Dockerfile.kernel`:

```bash
docker build -f dev-tools/Dockerfile.kernel -t esm-kernel:toolchain .
docker run --rm -it -v "$PWD":/workspace esm-kernel:toolchain \
  bash -lc "cd /workspace && ./build_kernel.sh"
```

This produces `UELAT/kernel_verified.so` in the repository checkout.
