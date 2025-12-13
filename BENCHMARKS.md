# Benchmarks

## Validation Status

✅ **PASS** - All validation criteria met

See [validation_history/latest.md](validation_history/latest.md) for the full validation report.

## Internal Validation (Synthetic)

| Metric | Result | Threshold | Status |
| --- | --- | --- | --- |
| Gold vs Drift T-Statistic | -44.90 (p ~ 0.0) | p < 0.001 | ✅ |
| ROC AUC | 0.946 | > 0.8 | ✅ |
| Cohen's D | 2.01 | > 0.8 | ✅ |

_Source: `tools/drift_metric_validation_results.json` (n=1000 runs, 12 steps, 64-dim embeddings)._

## Validation History

This repository maintains a complete audit trail of all CI validation runs in the
[`validation_history/`](validation_history/) directory. Each CI run generates timestamped
reports with full metrics, proving the spectral certificate reliably distinguishes
coherent reasoning from drift/hallucination.

- **Latest Report**: [`validation_history/latest.md`](validation_history/latest.md)
- **Full History**: [`validation_history/`](validation_history/)

## Upcoming External Benchmarks

- HaluEval
- TruthfulQA
- GSM8K
