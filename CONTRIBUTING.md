# Contributing

Contributions welcome. Please open issues or pull requests. For non-trivial changes, please discuss via issues first. Respect the license.

## Development Guidelines

### Coq Proof Development

The `UELAT/` directory contains formally verified Coq proofs. When modifying these:

1. **Test locally before pushing**: Use Docker to test Coq compilation:
   ```bash
   cd UELAT
   docker run --rm -v $(pwd):/work -w /work coqorg/coq:8.18.0 \
     bash -c "coqc -Q . '' Checker.v"
   ```

2. **Follow established patterns**: See `UELAT/BUILD.md` for recommended Coq proof patterns, especially for decomposing boolean conjunctions.

3. **Document complex proofs**: Add comments explaining non-obvious proof strategies.

4. **CI Validation**: All Coq proofs are compiled in CI using the `build_verified_kernel` job. Ensure your changes pass this check.

### Python Development

- Follow PEP 8 style guidelines
- Add tests for new functionality
- Run the test suite: `pytest tests/`

### Documentation

- Update relevant documentation in `docs/` when changing functionality
- Keep `UELAT/BUILD.md` up-to-date with Coq development patterns
- Update `README.md` for user-facing changes
