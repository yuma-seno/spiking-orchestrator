from __future__ import annotations

import numpy as np

from sorch.core.mc_experiment import generate_input


def test_generate_input_shapes_and_reproducible():
    steps = 1234
    u1 = generate_input(seed=0, steps=steps, mode="uniform")
    u2 = generate_input(seed=0, steps=steps, mode="uniform")
    assert u1.shape == (steps,)
    assert u1.dtype == np.float32
    assert np.all(np.isfinite(u1))
    assert np.array_equal(u1, u2)


def test_generate_input_gaussian_clipped_range():
    u = generate_input(seed=1, steps=2000, mode="gaussian", std=2.0, clip=1.0)
    assert float(np.max(u)) <= 1.0
    assert float(np.min(u)) >= -1.0


def test_generate_input_burst_contains_silence():
    u = generate_input(
        seed=2,
        steps=2000,
        mode="burst",
        std=0.5,
        clip=1.0,
        tempo_on_min=10,
        tempo_on_max=20,
        tempo_off_min=10,
        tempo_off_max=20,
    )
    # off segments are exact zeros
    assert int(np.sum(u == 0.0)) > 0


def test_generate_input_convo_contains_silence_and_bias():
    u = generate_input(
        seed=3,
        steps=2000,
        mode="convo",
        std=0.5,
        clip=1.0,
        input_bias=0.2,
        tempo_on_min=10,
        tempo_on_max=20,
        tempo_off_min=10,
        tempo_off_max=20,
    )
    bias_val = np.float32(0.2)
    # off segments become exactly bias (because gate=0 then +bias)
    assert int(np.sum(u == bias_val)) > 0
    # on segments should still vary (not all bias)
    assert int(np.sum(u != bias_val)) > 0
    # bias should make mean positive-ish even after clipping
    assert float(np.mean(u)) > 0.0
