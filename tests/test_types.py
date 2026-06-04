from __future__ import annotations

import numpy as np
import pytest
import torch


def test_tensor_to_numpy_shares_storage_for_contiguous_cpu_tensor() -> None:
    from c5r_sim_inference.types import tensor_to_numpy

    tensor = torch.arange(6, dtype=torch.float32).reshape(3, 2)

    array = tensor_to_numpy(tensor, dtype=torch.float32, name="actions", rank=2)

    assert array.dtype == np.float32
    assert array.flags.c_contiguous
    assert array.__array_interface__["data"][0] == tensor.data_ptr()


def test_tensor_to_numpy_normalizes_non_contiguous_tensor() -> None:
    from c5r_sim_inference.types import tensor_to_numpy

    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()

    array = tensor_to_numpy(tensor, dtype=torch.float32, name="actions", rank=2)

    assert array.shape == (4, 3)
    assert array.flags.c_contiguous
    assert array.__array_interface__["data"][0] != tensor.data_ptr()


def test_tensor_to_numpy_casts_dtype_and_detaches_grad() -> None:
    from c5r_sim_inference.types import tensor_to_numpy

    tensor = torch.arange(4, dtype=torch.float64, requires_grad=True)

    array = tensor_to_numpy(tensor, dtype=torch.float32, name="initial_state", rank=1)

    assert array.dtype == np.float32
    np.testing.assert_array_equal(array, np.arange(4, dtype=np.float32))


def test_initial_state_and_actions_validate_rank_and_empty_batch() -> None:
    from c5r_sim_inference.types import actions_to_numpy, initial_state_to_numpy

    with pytest.raises(ValueError, match="initial_state must have rank 1"):
        initial_state_to_numpy(torch.zeros(1, 2))

    with pytest.raises(ValueError, match="actions must have rank 2"):
        actions_to_numpy(torch.zeros(2))

    with pytest.raises(ValueError, match="actions batch must not be empty"):
        actions_to_numpy(torch.zeros(0, 3))


def test_tensor_to_numpy_rejects_non_finite_values() -> None:
    from c5r_sim_inference.types import actions_to_numpy

    with pytest.raises(ValueError, match="finite"):
        actions_to_numpy(torch.tensor([[float("inf")]], dtype=torch.float32))


def test_torch_observation_from_batch_converts_dtypes() -> None:
    from c5r_sim_inference.protocol import ObservationBatch
    from c5r_sim_inference.types import TorchObservation

    batch = ObservationBatch(
        sim_id="sim-1",
        step_indices=np.array([1, 2], dtype=np.int64),
        view_names=("overhead",),
        camera=np.zeros((2, 1, 3, 4, 3), dtype=np.uint8),
        qpos=np.ones((2, 4), dtype=np.float32),
        qvel=np.ones((2, 4), dtype=np.float32) * 2,
        ctrl=np.ones((2, 4), dtype=np.float32) * 3,
    )

    observation = TorchObservation.from_batch(batch)

    assert observation.sim_id == "sim-1"
    assert observation.view_names == ("overhead",)
    assert observation.step_indices.dtype == torch.int64
    assert observation.camera.dtype == torch.uint8
    assert observation.qpos.dtype == torch.float32
    assert observation.camera.shape == (2, 1, 3, 4, 3)
    torch.testing.assert_close(observation.ctrl, torch.ones(2, 4) * 3)
