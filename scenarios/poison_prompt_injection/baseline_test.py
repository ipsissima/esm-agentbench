#!/usr/bin/env python3
"""Baseline test for poison_prompt_injection scenario.

This test validates that:
1. Target data files are properly formatted
2. Sample conversations load correctly
3. The ticket routing logic works for clean inputs
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCENARIO_DIR = Path(__file__).parent
TARGETS_DIR = SCENARIO_DIR / "targets"


def test_service_spec_exists():
    """Verify service specification document exists."""
    spec_path = TARGETS_DIR / "service_spec.md"
    assert spec_path.exists(), "service_spec.md must exist"
    content = spec_path.read_text()
    assert "Customer Support Chat Service" in content
    assert "Security Requirements" in content
    print("  PASS: service_spec.md exists and contains expected sections")


def test_sample_conversation_loads():
    """Verify sample conversation JSON is valid."""
    conv_path = TARGETS_DIR / "sample_conversation.json"
    assert conv_path.exists(), "sample_conversation.json must exist"

    with conv_path.open() as f:
        data = json.load(f)

    assert "tickets" in data
    assert len(data["tickets"]) >= 3
    assert "expected_routing" in data

    for ticket in data["tickets"]:
        assert "ticket_id" in ticket
        assert "body" in ticket
        assert "metadata" in ticket

    print(f"  PASS: sample_conversation.json loads with {len(data['tickets'])} tickets")


def test_credentials_schema_valid():
    """Verify credentials schema is valid JSON Schema."""
    schema_path = TARGETS_DIR / "credentials_schema.json"
    assert schema_path.exists(), "credentials_schema.json must exist"

    with schema_path.open() as f:
        schema = json.load(f)

    assert "$schema" in schema
    assert "properties" in schema
    assert "note" in schema
    assert "SYNTHETIC" in schema["note"]

    print("  PASS: credentials_schema.json is valid and marked as synthetic")


def test_ticket_routing_logic():
    """Test basic ticket routing based on content."""
    def route_ticket(ticket: dict) -> str:
        """Simple rule-based ticket routing."""
        body = ticket.get("body", "").lower()
        tags = ticket.get("metadata", {}).get("tags", [])

        if any(t in tags for t in ["billing", "payment", "refund"]):
            return "Billing"
        if any(t in tags for t in ["technical", "crash", "error"]):
            return "Technical"
        if any(t in tags for t in ["returns", "order", "shipping"]):
            return "Sales"
        if "billing" in body or "charge" in body:
            return "Billing"
        if "crash" in body or "error" in body:
            return "Technical"
        return "General"

    # Test with sample tickets
    conv_path = TARGETS_DIR / "sample_conversation.json"
    with conv_path.open() as f:
        data = json.load(f)

    expected = data["expected_routing"]
    passed = 0
    for ticket in data["tickets"]:
        tid = ticket["ticket_id"]
        routed = route_ticket(ticket)
        exp = expected.get(tid, "General")
        if routed == exp:
            passed += 1
        else:
            print(f"    MISMATCH: {tid} routed to {routed}, expected {exp}")

    assert passed == len(data["tickets"]), f"Routing mismatch: {passed}/{len(data['tickets'])}"
    print(f"  PASS: Ticket routing works for {passed} tickets")


def test_poisoned_ticket_structure():
    """Verify poisoned ticket has expected attack vectors."""
    poison_path = TARGETS_DIR / "poisoned_ticket.json"
    assert poison_path.exists(), "poisoned_ticket.json must exist"

    with poison_path.open() as f:
        ticket = json.load(f)

    assert "attack_vectors" in ticket
    assert len(ticket["attack_vectors"]) >= 3
    assert "HIDDEN INSTRUCTIONS" in ticket["body"]

    print(f"  PASS: poisoned_ticket.json has {len(ticket['attack_vectors'])} attack vectors")


def main():
    """Run all baseline tests."""
    print("=" * 60)
    print("Poison Prompt Injection Scenario - Baseline Tests")
    print("=" * 60)

    tests = [
        ("Service spec exists", test_service_spec_exists),
        ("Sample conversation loads", test_sample_conversation_loads),
        ("Credentials schema valid", test_credentials_schema_valid),
        ("Ticket routing logic", test_ticket_routing_logic),
        ("Poisoned ticket structure", test_poisoned_ticket_structure),
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
