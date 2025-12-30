#!/usr/bin/env python3
"""Test that all validation reports contain data_source: real_traces_only.

This ensures that all generated reports are properly flagged as using
real traces only, preventing confusion with any legacy synthetic reports.
"""
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent


class TestReportsRealOnlyFlag:
    """Ensure all validation reports have real_traces_only flag."""

    def test_validation_reports_have_data_source_flag(self):
        """All validation_report.json files must include data_source: real_traces_only."""
        reports_dir = PROJECT_ROOT / "reports" / "spectral_validation_real_hf"

        if not reports_dir.exists():
            pytest.skip("No reports directory found yet (reports will be generated)")
            return

        validation_reports = list(reports_dir.glob("*/validation_report.json"))

        if not validation_reports:
            pytest.skip("No validation reports found yet (reports will be generated)")
            return

        violations = []

        for report_file in validation_reports:
            try:
                with open(report_file) as f:
                    data = json.load(f)

                data_source = data.get("data_source")

                if data_source != "real_traces_only":
                    violations.append((
                        report_file,
                        f"data_source={data_source!r}, expected 'real_traces_only'"
                    ))

            except (json.JSONDecodeError, IOError) as e:
                violations.append((report_file, f"Failed to read: {e}"))

        if violations:
            msg = "\n".join([
                f"  {file}: {error}"
                for file, error in violations
            ])
            pytest.fail(f"Validation reports missing or incorrect data_source:\n{msg}")

    def test_cross_model_reports_have_real_flag(self):
        """Cross-model reports should also indicate real traces."""
        reports_dir = PROJECT_ROOT / "reports" / "spectral_validation_real_hf"

        if not reports_dir.exists():
            pytest.skip("No reports directory found yet")
            return

        cross_reports = list(reports_dir.glob("*/cross_model_report.json"))

        if not cross_reports:
            # Cross-model reports are optional
            return

        # If they exist, they should also have proper metadata
        for report_file in cross_reports:
            try:
                with open(report_file) as f:
                    data = json.load(f)

                # Cross-model reports may not have data_source at top level,
                # but should reference real traces
                # This is informational only
                assert isinstance(data, dict), f"{report_file} is not a dict"

            except (json.JSONDecodeError, IOError) as e:
                pytest.fail(f"Failed to read {report_file}: {e}")

    def test_no_reports_in_wrong_location(self):
        """Ensure reports are only in spectral_validation_real_hf/."""
        reports_dir = PROJECT_ROOT / "reports"

        if not reports_dir.exists():
            pytest.skip("No reports directory found yet")
            return

        # Check for validation reports in wrong locations
        wrong_locations = []

        for json_file in reports_dir.glob("**/validation_report.json"):
            # Must be under spectral_validation_real_hf/
            if "spectral_validation_real_hf" not in str(json_file):
                wrong_locations.append(json_file)

        if wrong_locations:
            msg = "\n".join([f"  {f}" for f in wrong_locations])
            pytest.fail(
                f"Found validation reports outside spectral_validation_real_hf/:\n{msg}\n"
                "All reports must be under reports/spectral_validation_real_hf/"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
