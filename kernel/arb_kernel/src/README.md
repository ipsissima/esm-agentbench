# ARB Kernel Source

This directory will contain the production C/OCaml implementation of the verified kernel
using the ARB (Arb) library for rigorous interval arithmetic.

## Planned Implementation

The production kernel will implement:

1. **SVD with Interval Bounds**: Using ARB's certified linear algebra routines
2. **Koopman Operator Fitting**: Ridge regression with interval error bounds
3. **Residual Computation**: Interval-bounded prediction errors
4. **Theoretical Bound Verification**: Provably correct bound computation

## Current Status

The current implementation uses a Python prototype (`prototype_kernel.py`) as a fallback.
The production ARB implementation is planned for a future release.

## Building

```bash
# From the kernel/arb_kernel directory
docker build -t ipsissima/kernel:latest .
```

## References

- [ARB Library](https://arblib.org/) - C library for arbitrary-precision ball arithmetic
- [FLINT](https://flintlib.org/) - Fast Library for Number Theory (ARB dependency)
