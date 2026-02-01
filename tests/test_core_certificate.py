import numpy as np

from core.certificate import compute_certificate_from_trace


def test_compute_certificate_from_trace_embeddings():
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.25, 0.25, 0.5],
            [0.0, 0.0, 1.0],
        ]
    )
    trace = {"embeddings": embeddings.tolist()}

    cert = compute_certificate_from_trace(trace, rank=2)

    assert "theoretical_bound" in cert
    assert "residual" in cert
    assert "kernel_output" not in cert
