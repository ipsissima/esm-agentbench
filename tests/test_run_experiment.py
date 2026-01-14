from analysis.run_experiment import summarize_reports


def test_summarize_reports_no_tested_scenarios():
    reports = {
        "scenario_a": {
            "error": "No traces found",
            "AUC": 0.0,
            "TPR_at_FPR05": 0.0,
        }
    }

    results, num_tested, num_passed = summarize_reports(reports)

    assert num_tested == 0
    assert num_passed == 0
    assert results[0]["status"] == "skipped"
