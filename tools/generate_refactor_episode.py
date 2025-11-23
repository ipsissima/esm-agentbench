#!/usr/bin/env python
"""Generate a realistic multi-file refactor episode with runnable tests."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def ensure_dirs():
    (ROOT / "demo_swe" / "refactor_sources" / "pkg").mkdir(parents=True, exist_ok=True)
    (ROOT / "demo_swe" / "refactor_tests").mkdir(parents=True, exist_ok=True)


def write_sources():
    pkg_dir = ROOT / "demo_swe" / "refactor_sources" / "pkg"
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    helpers_code = """
import math

def normalize_vector(vec):
    total = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / total for v in vec]
"""
    (pkg_dir / "helpers.py").write_text(helpers_code, encoding="utf-8")

    geometry_code = """
from .helpers import normalize_vector


def compute_projection(u, v):
    '''Project vector u onto v using normalized helper.'''
    norm_v = normalize_vector(v)
    dot = sum(a * b for a, b in zip(u, norm_v))
    return [dot * b for b in norm_v]
"""
    (pkg_dir / "geometry.py").write_text(geometry_code, encoding="utf-8")

    legacy_code = """
# Legacy function slated for refactor; kept for backward compatibility.
from .geometry import compute_projection
"""
    (pkg_dir / "legacy.py").write_text(legacy_code, encoding="utf-8")


def write_tests():
    test_path = ROOT / "demo_swe" / "refactor_tests" / "test_refactor.py"
    test_code = """
import importlib


def test_projection_shape():
    mod = importlib.import_module('pkg.geometry')
    result = mod.compute_projection([3.0, 4.0], [1.0, 0.0])
    assert len(result) == 2
    assert abs(result[0] - 5.0) < 1e-6
    assert abs(result[1]) < 1e-6


def test_legacy_alias():
    legacy = importlib.import_module('pkg.legacy')
    assert hasattr(legacy, 'compute_projection')


def test_helper_normalizes():
    helpers = importlib.import_module('pkg.helpers')
    vec = helpers.normalize_vector([0.0, 5.0])
    assert abs(sum(v * v for v in vec) - 1.0) < 1e-6
"""
    test_path.write_text(test_code, encoding="utf-8")


def write_episode():
    episode = {
        "id": "ep_refactor",
        "prompt": "Refactor geometry helpers across multiple files and ensure imports remain stable.",
        "tests": [
            {
                "name": "refactor",
                "script": "import sys; sys.path.append('refactor_sources'); exec(open('refactor_tests/test_refactor.py').read())",
            }
        ],
    }
    ep_path = ROOT / "demo_swe" / "episodes" / "ep_refactor.json"
    ep_path.write_text(json.dumps(episode, indent=2), encoding="utf-8")


def main():
    ensure_dirs()
    write_sources()
    write_tests()
    write_episode()
    print("Generated refactor episode at demo_swe/episodes/ep_refactor.json")


if __name__ == "__main__":
    main()
