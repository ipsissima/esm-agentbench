
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
