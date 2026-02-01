from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from apps.make_certificate_app import run


def test_make_certificate_app_writes_output(tmp_path: Path) -> None:
    trace = {
        "embeddings": np.random.randn(5, 4).tolist(),
    }
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(trace))

    output_path = tmp_path / "certificate.json"
    certificate = run(
        str(trace_path),
        output_path=str(output_path),
        verify_with_kernel=False,
        rank=2,
    )

    assert output_path.exists()
    on_disk = json.loads(output_path.read_text())
    assert on_disk["theoretical_bound"] == certificate["theoretical_bound"]
