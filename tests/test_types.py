from __future__ import annotations

import numpy as np


def test_observation_from_batch_preserves_numpy_arrays() -> None:
    from c5r_sim_inference.protocol import ObservationBatch
    from c5r_sim_inference.types import Observation

    batch = ObservationBatch(
        sim_id="sim-1",
        step_indices=np.array([1, 2], dtype=np.int64),
        view_names=("overhead",),
        camera=np.zeros((2, 1, 3, 4, 3), dtype=np.uint8),
        qpos=np.ones((2, 4), dtype=np.float32),
        qvel=np.ones((2, 4), dtype=np.float32) * 2,
        ctrl=np.ones((2, 4), dtype=np.float32) * 3,
    )

    observation = Observation.from_batch(batch)

    assert observation.sim_id == "sim-1"
    assert observation.view_names == ("overhead",)
    assert observation.step_indices.dtype == np.int64
    assert observation.camera.dtype == np.uint8
    assert observation.qpos.dtype == np.float32
    assert observation.camera.shape == (2, 1, 3, 4, 3)
    assert observation.camera.flags.c_contiguous
    np.testing.assert_array_equal(observation.ctrl, np.ones((2, 4), dtype=np.float32) * 3)
