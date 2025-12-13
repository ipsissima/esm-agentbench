# Benchmarks

## Internal Validation (Synthetic)
| Metric | Result |
| --- | --- |
| Gold vs Drift T-Statistic | -44.90 (p ~ 0.0) |
| ROC AUC | 0.946 |
| Cohen's D | 2.01 |

_Source: `tools/drift_metric_validation_results.json` (n=1000 runs, 12 steps, 64-dim embeddings)._ 

## Upcoming External Benchmarks
- HaluEval
- TruthfulQA
- GSM8K
