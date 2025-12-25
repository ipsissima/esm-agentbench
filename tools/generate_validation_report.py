#!/usr/bin/env python3
"""Generate unified validation reports for hybrid certification.

This tool creates comprehensive reports that correlate:
- Drift (Spectral): Did the mind wander? (theoretical_bound)
- Robustness (Lipschitz): Was the understanding shaky? (lipschitz_margin)
- Execution (Runtime): Did the code work? (ground_truth_passed)

The report analyzes the Conditional Probability of Success given a Valid Certificate.
Ideally, this should be 1.0 (all valid certificates should correspond to correct solutions).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from esmassessor.artifact_schema import CertifiedVerdict


def load_trace_result(trace_path: str) -> Optional[Dict[str, Any]]:
    """Load a saved trace result from disk."""
    try:
        with open(trace_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def compute_validation_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate validation metrics across multiple runs.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of trace results, each with certificate and execution status.

    Returns
    -------
    Dict[str, Any]
        Metrics including:
        - total_episodes: Number of episodes analyzed
        - passed_certificates: Count of PASS verdicts
        - failed_certificates: Count of FAIL verdicts
        - execution_accuracy: Proportion where execution succeeded
        - precision_of_pass: P(execution success | certified PASS)
        - recall_of_pass: P(certified PASS | execution success)
        - f1_score: Harmonic mean of precision and recall
        - average_bound: Mean theoretical_bound across all episodes
        - bound_by_verdict: Bounds grouped by verdict type
    """
    if not results:
        return {}

    total = len(results)
    pass_verdicts = 0
    fail_verdicts = 0
    execution_successes = 0

    # Count outcomes
    pass_and_exec = 0  # PASS verdict AND execution success
    pass_but_failed = 0  # PASS verdict but execution failed (FALSE POSITIVE)
    fail_but_exec = 0  # FAIL verdict but execution succeeded (FALSE NEGATIVE)
    fail_and_failed = 0  # FAIL verdict AND execution failed (TRUE NEGATIVE)

    bounds_by_verdict: Dict[str, List[float]] = {v.value: [] for v in CertifiedVerdict}
    all_bounds: List[float] = []

    for result in results:
        cert = result.get("certificate", {})
        verdict_str = cert.get("certified_verdict", "UNKNOWN")
        execution_success = result.get("task_success", False)

        # Track verdict counts
        if verdict_str == CertifiedVerdict.PASS.value:
            pass_verdicts += 1
        elif verdict_str != "UNKNOWN":
            fail_verdicts += 1

        # Track execution counts
        if execution_success:
            execution_successes += 1

        # Track outcomes
        if verdict_str == CertifiedVerdict.PASS.value and execution_success:
            pass_and_exec += 1
        elif verdict_str == CertifiedVerdict.PASS.value and not execution_success:
            pass_but_failed += 1
        elif verdict_str != CertifiedVerdict.PASS.value and execution_success:
            fail_but_exec += 1
        else:
            fail_and_failed += 1

        # Track bounds by verdict
        theoretical_bound = float(cert.get("theoretical_bound", float("nan")))
        if not (theoretical_bound != theoretical_bound):  # Check for NaN
            all_bounds.append(theoretical_bound)
            if verdict_str in bounds_by_verdict:
                bounds_by_verdict[verdict_str].append(theoretical_bound)

    # Compute metrics
    precision_of_pass = pass_and_exec / pass_verdicts if pass_verdicts > 0 else 0.0
    recall_of_pass = pass_and_exec / execution_successes if execution_successes > 0 else 0.0
    f1 = 2 * (precision_of_pass * recall_of_pass) / (precision_of_pass + recall_of_pass) if (precision_of_pass + recall_of_pass) > 0 else 0.0

    avg_bound = sum(all_bounds) / len(all_bounds) if all_bounds else float("nan")

    # Group bounds by verdict for analysis
    bounds_stats = {}
    for verdict, bounds in bounds_by_verdict.items():
        if bounds:
            bounds_stats[verdict] = {
                "count": len(bounds),
                "mean": sum(bounds) / len(bounds),
                "min": min(bounds),
                "max": max(bounds),
            }

    return {
        "total_episodes": total,
        "passed_certificates": pass_verdicts,
        "failed_certificates": fail_verdicts,
        "execution_successes": execution_successes,
        "execution_accuracy": execution_successes / total if total > 0 else 0.0,
        "true_positives": pass_and_exec,
        "false_positives": pass_but_failed,
        "false_negatives": fail_but_exec,
        "true_negatives": fail_and_failed,
        "precision_of_pass": precision_of_pass,
        "recall_of_pass": recall_of_pass,
        "f1_score": f1,
        "average_bound": avg_bound,
        "bounds_stats": bounds_stats,
    }


def generate_markdown_report(
    results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    title: str = "Hybrid Certification Validation Report",
) -> str:
    """Generate a human-readable markdown report.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of trace results.
    metrics : Dict[str, Any]
        Computed metrics from compute_validation_metrics.
    title : str, optional
        Report title. Default is "Hybrid Certification Validation Report".

    Returns
    -------
    str
        Markdown-formatted report.
    """
    lines = [
        f"# {title}",
        "",
        "## Summary",
        f"- **Total Episodes**: {metrics.get('total_episodes', 0)}",
        f"- **Passed Certificates**: {metrics.get('passed_certificates', 0)}",
        f"- **Failed Certificates**: {metrics.get('failed_certificates', 0)}",
        f"- **Execution Success Rate**: {metrics.get('execution_accuracy', 0):.2%}",
        "",
        "## Certificate Validity Assessment",
        f"- **Precision of PASS**: {metrics.get('precision_of_pass', 0):.2%} "
        f"(of {metrics.get('passed_certificates', 0)} PASS verdicts, {metrics.get('true_positives', 0)} were correct)",
        f"- **Recall of PASS**: {metrics.get('recall_of_pass', 0):.2%} "
        f"(of {metrics.get('execution_successes', 0)} successful executions, {metrics.get('true_positives', 0)} were certified PASS)",
        f"- **F1 Score**: {metrics.get('f1_score', 0):.4f}",
        "",
        "## Confusion Matrix",
        f"- **True Positives** (Certified PASS, Execution OK): {metrics.get('true_positives', 0)}",
        f"- **False Positives** (Certified PASS, Execution FAIL): {metrics.get('false_positives', 0)}",
        f"- **False Negatives** (Certified FAIL, Execution OK): {metrics.get('false_negatives', 0)}",
        f"- **True Negatives** (Certified FAIL, Execution FAIL): {metrics.get('true_negatives', 0)}",
        "",
        "## Spectral Stability Analysis",
        f"- **Average Theoretical Bound**: {metrics.get('average_bound', float('nan')):.4f}",
        "",
    ]

    bounds_stats = metrics.get("bounds_stats", {})
    if bounds_stats:
        lines.append("### Bounds by Verdict Type")
        for verdict, stats in sorted(bounds_stats.items()):
            lines.append(
                f"- **{verdict}**: mean={stats['mean']:.4f}, "
                f"range=[{stats['min']:.4f}, {stats['max']:.4f}], "
                f"n={stats['count']}"
            )
        lines.append("")

    # Interpretation
    lines.extend([
        "## Interpretation",
        "",
        "**Ideal Behavior**: Precision and Recall should both be 1.0",
        "- All PASS verdicts correspond to successful executions",
        "- All successful executions are certified as PASS",
        "",
        "**False Positives** (PASS but failed execution):",
        "- Indicates the certificate is too permissive",
        "- Consider lowering `bound_threshold` or `semantic_compliance_threshold`",
        "",
        "**False Negatives** (FAIL but successful execution):",
        "- Indicates the certificate is too strict",
        "- Consider raising `bound_threshold` or loosening other gates",
        "",
        "**Spectral Bound**: Lower values indicate more stable reasoning",
        "- Bounds above threshold may indicate drift even if tests pass",
        "- Use bounds distribution to calibrate threshold",
        "",
    ])

    return "\n".join(lines)


def generate_json_report(
    results: List[Dict[str, Any]],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate a machine-readable JSON report.

    Parameters
    ----------
    results : List[Dict[str, Any]]
        List of trace results.
    metrics : Dict[str, Any]
        Computed metrics from compute_validation_metrics.

    Returns
    -------
    Dict[str, Any]
        Structured report for further analysis.
    """
    return {
        "metadata": {
            "total_episodes": metrics.get("total_episodes", 0),
            "report_type": "hybrid_certification_validation",
        },
        "summary": {
            "passed_certificates": metrics.get("passed_certificates", 0),
            "failed_certificates": metrics.get("failed_certificates", 0),
            "execution_successes": metrics.get("execution_successes", 0),
            "execution_accuracy": metrics.get("execution_accuracy", 0.0),
        },
        "validity_assessment": {
            "precision_of_pass": metrics.get("precision_of_pass", 0.0),
            "recall_of_pass": metrics.get("recall_of_pass", 0.0),
            "f1_score": metrics.get("f1_score", 0.0),
            "true_positives": metrics.get("true_positives", 0),
            "false_positives": metrics.get("false_positives", 0),
            "false_negatives": metrics.get("false_negatives", 0),
            "true_negatives": metrics.get("true_negatives", 0),
        },
        "spectral_analysis": {
            "average_bound": metrics.get("average_bound", float("nan")),
            "bounds_stats": metrics.get("bounds_stats", {}),
        },
        "episode_details": [
            {
                "episode_id": r.get("episode_id"),
                "task_success": r.get("task_success", False),
                "certificate_verdict": r.get("certificate", {}).get("certified_verdict", "UNKNOWN"),
                "theoretical_bound": float(r.get("certificate", {}).get("theoretical_bound", float("nan"))),
                "reasoning": r.get("certificate", {}).get("reasoning"),
            }
            for r in results
        ],
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python generate_validation_report.py <trace_dir>")
        print("Generate validation report from traces in a directory")
        sys.exit(1)

    trace_dir = Path(sys.argv[1])
    if not trace_dir.exists():
        print(f"Error: {trace_dir} does not exist")
        sys.exit(1)

    # Load all trace results
    results = []
    for trace_file in sorted(trace_dir.glob("*.json")):
        data = load_trace_result(str(trace_file))
        if data:
            results.append(data)

    if not results:
        print(f"No trace results found in {trace_dir}")
        sys.exit(1)

    # Compute metrics
    metrics = compute_validation_metrics(results)

    # Generate reports
    markdown_report = generate_markdown_report(results, metrics)
    json_report = generate_json_report(results, metrics)

    # Output
    print(markdown_report)
    print("\n" + "=" * 80 + "\n")
    print("JSON Report:")
    print(json.dumps(json_report, indent=2, default=str))
