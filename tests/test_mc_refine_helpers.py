from __future__ import annotations

from sorch.bench.mc_refine import _refine_around


def test_refine_around_includes_center_and_bounds():
    vals = _refine_around(200.0, step=50.0, span=100.0, bounds=(150.0, 250.0))
    assert 200.0 in vals
    assert min(vals) >= 150.0
    assert max(vals) <= 250.0
    assert vals == sorted(vals)
