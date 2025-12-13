# Validation Report

## Metadata

- **Timestamp**: 2025-12-13T23:01:30.151253+00:00
- **Commit**: `89fc822c9079`
- **Branch**: claude/add-validation-history-01H6J1vs47BCRbRKGV6Rc9Y1
- **Author**: andreu-toposcircuitry
- **Message**: Merge pull request #37 from ipsissima/codex/upgrade-benchmarking-script-for-real-datasets

## Overall Verdict

**✅ PASS** (3/3 checks passed)

### Validation Checks

| Check | Threshold | Value | Status |
|-------|-----------|-------|--------|
| p-value < 0.001 | 0.001 | 2.74e-305 | ✅ Pass |
| AUC-ROC > 0.8 | 0.8 | 0.9463 | ✅ Pass |
| Cohen's d > 0.8 (large effect) | 0.8 | 2.0079 | ✅ Pass |

## Drift Metric Validation (Synthetic)

- **Runs**: 1000
- **Steps**: 12
- **Embedding Dim**: 64

### Separation Metrics

- **AUC-ROC**: 0.9463
- **Best Accuracy**: 0.893
- **Optimal Threshold**: 0.2384

### Distribution Statistics

| Trace Type | Mean | Std | Median |
|------------|------|-----|--------|
| Gold | 0.1961 | 0.0325 | 0.1933 |
| Creative | 0.1656 | 0.0329 | 0.1635 |
| Drift | 0.2782 | 0.0478 | 0.2791 |
| Poison | 0.3245 | 0.0831 | 0.3267 |
| Starved | 0.2657 | 0.0421 | 0.2687 |

## Demo SWE Validation

- **Episodes**: 5
- **Mean Residual**: 6.74e-12
- **Mean Theoretical Bound**: 6.74e-12

---
*Generated at 2025-12-13T23:01:30.151253+00:00 by `tools/generate_validation_report.py`*
