"""Minimal walkthrough for generating a spectral certificate over a dummy trace."""
from __future__ import annotations

import json

from assessor.kickoff import embed_trace_steps
from certificates.make_certificate import compute_certificate


def main() -> None:
    trace_steps = [
        {"text": "Agent observes the environment."},
        {"text": "Agent proposes a plan."},
        {"text": "Agent executes the plan and reflects."},
    ]

    embeddings = embed_trace_steps(trace_steps)
    certificate = compute_certificate(embeddings)

    print(json.dumps(certificate, indent=2, sort_keys=True))

    bound = certificate.get("theoretical_bound", float("inf"))
    verdict = "PASS" if bound < 1.0 else "FAIL"
    print(f"\nVerdict based on theoretical_bound: {verdict} (threshold = 1.0)")


if __name__ == "__main__":
    main()
