# Building kernel_verified.so (UELAT)

Requirements:
- Ubuntu 22.04 (or similar)
- opam, coq 8.18.1, ocaml >= 4.14.0

Recommended reproducible flow (use dev-tools/Dockerfile.kernel):

```
docker build -t esm-kernel-builder -f dev-tools/Dockerfile.kernel .
docker run --rm -v $(pwd):/work esm-kernel-builder
```

This will run ./build_kernel.sh in /work and produce UELAT/kernel_verified.so on success.

Alternative: use the official Coq 8.18.1 image directly:

```
docker run --rm -v $(pwd):/work -w /work \
  coqorg/coq:8.18.1 \
  bash -lc "cd UELAT && chmod +x ./build_kernel.sh && ./build_kernel.sh"
```

If building locally without Docker:
1. sudo apt install opam coq ocaml ocaml-native-compilers ocamlfind build-essential m4
2. opam init --bare --disable-sandboxing
3. opam switch create esm-kernel 4.14.0
4. eval $(opam env)
5. opam install coq.8.18.1 dune ocamlfind
6. ./build_kernel.sh

Note: If kernel_verified.so is included in the repo or artifact, CI will prefer it over building from source.

Note: pyproject.toml is the canonical package metadata; requirements.txt is kept for convenience installs.
