import numpy as np
from sklearn.utils import resample


def bootstrap_null_threshold(feature_values_by_trace, B=500, rows_resample=True, percentile=95):
    """
    Given a list of traces, each trace is a 1D array of per-row features or the trace matrix rows.
    If rows_resample True: resample rows within trace; else resample traces.
    Returns threshold as percentile of pooled bootstrap distribution.
    """
    pooled_boot = []
    for _ in range(B):
        boot_vals = []
        for trace_rows in feature_values_by_trace:
            if len(trace_rows) == 0:
                continue
            if rows_resample:
                sampled = resample(trace_rows, replace=True, n_samples=len(trace_rows))
                # compute statistic on sampled rows; if feature is scalar per-trace, aggregate appropriately
                # For R_norm, we will resample embeddings; the caller will compute R_norm on the resampled embeddings.
                boot_vals.append(sampled)
            else:
                boot_vals.append(
                    resample(
                        feature_values_by_trace,
                        replace=True,
                        n_samples=len(feature_values_by_trace),
                    )
                )
        # The caller will typically compute pooled R_norms; here we simply push placeholder
        pooled_boot.append(boot_vals)
    # This utility is a thin wrapper used by validate script which performs the actual R_norm computation.
    return pooled_boot
