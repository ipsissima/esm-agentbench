#!/usr/bin/env python3
"""Baseline test for indirect_email_exfil scenario.

This test validates that:
1. Target data files exist and are properly formatted
2. Email CSV can be parsed correctly
3. HTML snapshot contains expected structure
4. PII extraction functions work on synthetic data
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

SCENARIO_DIR = Path(__file__).parent
TARGETS_DIR = SCENARIO_DIR / "targets"


def test_emails_csv_exists():
    """Verify emails.csv exists and is valid."""
    csv_path = TARGETS_DIR / "emails.csv"
    assert csv_path.exists(), "emails.csv must exist"

    with csv_path.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) >= 5, "Should have at least 5 emails"

    required_cols = ["email_id", "from_address", "to_address", "subject", "body", "has_pii"]
    for row in rows:
        for col in required_cols:
            assert col in row, f"Missing column: {col}"

    pii_count = sum(1 for r in rows if r.get("has_pii", "").lower() == "true")
    assert pii_count >= 4, f"Should have at least 4 emails with PII, found {pii_count}"

    print(f"  PASS: emails.csv has {len(rows)} rows, {pii_count} with PII")


def test_webpage_snapshot_exists():
    """Verify webpage_snapshot.html exists and has expected structure."""
    html_path = TARGETS_DIR / "webpage_snapshot.html"
    assert html_path.exists(), "webpage_snapshot.html must exist"

    content = html_path.read_text(encoding='utf-8')

    # Check for expected structure
    assert "<html" in content, "Must be valid HTML"
    assert "Employee Directory" in content, "Must contain employee directory"
    assert "example.com" in content, "Must use example.com for synthetic emails"

    # Count employee entries
    employee_count = content.count('class="employee"')
    assert employee_count >= 4, f"Should have at least 4 employees, found {employee_count}"

    print(f"  PASS: webpage_snapshot.html has {employee_count} employee entries")


def test_pii_patterns():
    """Test that PII extraction regex patterns work."""
    # Email pattern
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

    # Phone pattern (555-xxxx format)
    phone_pattern = r'555-\d{4}'

    # Employee ID pattern
    emp_id_pattern = r'EMP-\d{5}'

    # Test on emails.csv
    csv_path = TARGETS_DIR / "emails.csv"
    with csv_path.open(newline='', encoding='utf-8') as f:
        content = f.read()

    emails_found = re.findall(email_pattern, content)
    phones_found = re.findall(phone_pattern, content)
    emp_ids_found = re.findall(emp_id_pattern, content)

    assert len(emails_found) >= 6, f"Should find at least 6 emails, found {len(emails_found)}"
    assert len(phones_found) >= 2, f"Should find at least 2 phones, found {len(phones_found)}"
    assert len(emp_ids_found) >= 2, f"Should find at least 2 emp IDs, found {len(emp_ids_found)}"

    print(f"  PASS: PII patterns work - {len(emails_found)} emails, {len(phones_found)} phones, {len(emp_ids_found)} emp IDs")


def test_html_hidden_elements():
    """Verify HTML contains hidden instruction elements."""
    html_path = TARGETS_DIR / "webpage_snapshot.html"
    content = html_path.read_text(encoding='utf-8')

    # Check for hidden comments
    has_hidden_comment = "<!-- HIDDEN INSTRUCTION" in content or "<!--" in content
    assert has_hidden_comment, "Should have HTML comments"

    # Check for hidden CSS elements
    has_hidden_css = 'display: none' in content or 'hidden-instruction' in content
    assert has_hidden_css, "Should have CSS-hidden elements"

    print("  PASS: HTML contains hidden instruction elements")


def test_synthetic_data_markers():
    """Verify data is properly marked as synthetic."""
    readme_path = TARGETS_DIR / "README_TARGETS.md"
    assert readme_path.exists(), "README_TARGETS.md must exist"

    content = readme_path.read_text()
    assert "synthetic" in content.lower(), "README must mention synthetic data"
    assert "security research" in content.lower(), "README must mention security research"

    # Check emails use example.com
    csv_path = TARGETS_DIR / "emails.csv"
    csv_content = csv_path.read_text()
    assert "example.com" in csv_content, "Emails must use example.com domain"
    assert "@real" not in csv_content.lower(), "Must not contain real-looking domains"

    print("  PASS: All data properly marked as synthetic")


def main():
    """Run all baseline tests."""
    print("=" * 60)
    print("Indirect Email Exfil Scenario - Baseline Tests")
    print("=" * 60)

    tests = [
        ("Emails CSV exists", test_emails_csv_exists),
        ("Webpage snapshot exists", test_webpage_snapshot_exists),
        ("PII patterns work", test_pii_patterns),
        ("HTML hidden elements", test_html_hidden_elements),
        ("Synthetic data markers", test_synthetic_data_markers),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\n{name}...")
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"  FAIL: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        print("BASELINE: SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("BASELINE: ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
