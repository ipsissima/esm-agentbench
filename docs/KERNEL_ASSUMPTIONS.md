# Verified Kernel Assumptions

This document states the assumptions and scope for the verified kernel used by
the `esm-agentbench` certificate pipeline.

## Axioms

- The kernel assumes standard IEEE 754 floating-point arithmetic for all
  computations.

## Scope

- The Verified Kernel proves the derivation of the bound from the witness
  matrices. It does **NOT** verify the SVD algorithm used to generate those
  witnesses.

## Preconditions (Witness Checker)

The witness checker enforces numerical preconditions before invoking the
verified kernel, including:

- **Condition Number < 1e4** for the witness Gram matrix.
- **Rank Consistency** between the declared rank `k` and the witness matrix
  dimensions.
- Finite-valued witness matrices and a non-degenerate singular value gap.
