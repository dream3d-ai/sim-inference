from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pytest


def test_tensor_array_roundtrip_accepts_single_row_zero_stride_view() -> None:
    from c5r_sim_inference.protocol import (
        fixed_shape_tensor_array,
        fixed_shape_tensor_to_numpy,
    )

    image = np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 3)
    camera = image[None, ...]

    assert camera.flags.c_contiguous
    assert camera.strides[0] == 0

    values = fixed_shape_tensor_array(camera, value_shape=(2, 3, 4, 3), name="camera")
    restored = fixed_shape_tensor_to_numpy(values, name="camera")

    assert restored.flags.c_contiguous
    np.testing.assert_array_equal(restored, camera)


def test_new_sim_batch_roundtrip() -> None:
    from c5r_sim_inference.protocol import (
        new_sim_batch_to_record_batch,
        record_batch_to_new_sim,
    )

    batch = new_sim_batch_to_record_batch(
        task_id="task_11",
        initial_state=np.arange(4, dtype=np.float32).reshape(1, 4),
        views=("overhead", "side"),
        width=64,
        height=48,
        substeps=3,
    )

    assert batch.num_rows == 1
    assert batch.schema.metadata is not None
    metadata = json.loads(batch.schema.metadata[b"c5r_sim"].decode("utf-8"))
    assert metadata["op"] == "new_sim"
    assert metadata["views"] == ["overhead", "side"]
    assert isinstance(batch.schema.field("initial_state").type, pa.FixedShapeTensorType)

    request = record_batch_to_new_sim(batch)

    assert request.task_id == "task_11"
    assert request.views == ("overhead", "side")
    assert request.width == 64
    assert request.height == 48
    assert request.substeps == 3
    np.testing.assert_array_equal(
        request.initial_state, np.arange(4, dtype=np.float32).reshape(1, 4)
    )


def test_new_sim_batch_allows_empty_views_for_state_only_requests() -> None:
    from c5r_sim_inference.protocol import (
        client_request_new_sim_batch_to_record_batch,
        record_batch_to_new_sim,
    )

    batch = client_request_new_sim_batch_to_record_batch(
        task_id="task_11",
        initial_state=np.arange(4, dtype=np.float32).reshape(1, 4),
        views=(),
        width=64,
        height=48,
        substeps=1,
        action_shape=(1, 4),
    )

    assert record_batch_to_new_sim(batch).views == ()


def test_client_request_new_sim_includes_scene_options() -> None:
    from c5r_sim_inference.protocol import (
        client_request_new_sim_batch_to_record_batch,
        record_batch_to_new_sim,
    )

    batch = client_request_new_sim_batch_to_record_batch(
        task_id="task_13",
        initial_state=np.arange(4, dtype=np.float32).reshape(1, 4),
        views=(),
        width=64,
        height=48,
        substeps=1,
        action_shape=(1, 4),
        scene_seed=123,
        rack_dynamic=False,
        tube_radius=0.01,
        tube_hole_index=3,
        tube_contact_friction=(0.7, 0.005, 0.0002),
        table_x_min=-0.1,
        table_x_max=0.3,
    )

    request = record_batch_to_new_sim(batch)

    assert request.scene_options is not None
    assert request.scene_options["kind"] == "rack_tube"
    assert request.scene_options["seed"] == 123
    physics = request.scene_options["physics"]
    assert physics["rack_dynamic"] is False
    assert physics["rack"]["tube_radius"] == 0.01
    assert physics["rack"]["tube_half_length"] == 0.059
    assert physics["material"]["tube_contact"]["friction"] == [0.7, 0.005, 0.0002]
    assert physics["randomization"]["placement"]["tube_hole_index"] == 3
    assert physics["randomization"]["placement"]["table_bounds"]["x_max"] == 0.3


def test_client_request_new_sim_includes_physics_randomization_profile() -> None:
    from c5r_sim_inference.protocol import (
        client_request_new_sim_batch_to_record_batch,
        record_batch_to_new_sim,
    )

    batch = client_request_new_sim_batch_to_record_batch(
        task_id="task_11",
        initial_state=np.zeros((1, 4), dtype=np.float32),
        views=(),
        width=64,
        height=48,
        substeps=1,
        action_shape=(1, 4),
        scene_seed=123,
        physics_randomization=True,
    )

    request = record_batch_to_new_sim(batch)

    assert request.scene_options is not None
    assert request.scene_options["physics_randomization"] == {"profile": "default"}


def test_action_batch_roundtrip() -> None:
    from c5r_sim_inference.protocol import (
        action_batch_to_record_batch,
        record_batch_to_actions,
    )

    actions = np.arange(6, dtype=np.float32).reshape(3, 1, 2)
    batch = action_batch_to_record_batch(
        sim_id="sim-1",
        start_step_index=4,
        actions=actions,
    )

    request = record_batch_to_actions(batch, expected_sim_id="sim-1")

    assert request.sim_id == "sim-1"
    assert request.start_step_index == 4
    np.testing.assert_array_equal(request.actions, actions)


def test_observation_batch_roundtrip() -> None:
    from c5r_sim_inference.protocol import (
        observation_batch_to_record_batch,
        record_batch_to_observations,
    )

    camera = np.arange(2 * 1 * 2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 1, 2, 3, 4, 3)
    qpos = np.ones((2, 1, 4), dtype=np.float32)
    qvel = qpos * 2
    ctrl = qpos * 3

    batch = observation_batch_to_record_batch(
        sim_id="sim-1",
        step_indices=np.array([0, 1], dtype=np.int64),
        view_names=("overhead", "side"),
        camera=camera,
        qpos=qpos,
        qvel=qvel,
        ctrl=ctrl,
    )

    observations = record_batch_to_observations(batch)

    assert observations.sim_id == "sim-1"
    assert observations.step_indices.tolist() == [0, 1]
    assert observations.view_names == ("overhead", "side")
    assert observations.camera.dtype == np.uint8
    assert observations.qpos.dtype == np.float32
    np.testing.assert_array_equal(observations.camera, camera)
    np.testing.assert_array_equal(observations.ctrl, ctrl)


def test_action_batch_rejects_bad_rank_and_non_finite_values() -> None:
    from c5r_sim_inference.protocol import action_batch_to_record_batch

    with pytest.raises(ValueError, match="actions must have shape"):
        action_batch_to_record_batch(
            sim_id="sim-1",
            start_step_index=1,
            actions=np.zeros((3,), dtype=np.float32),
        )

    with pytest.raises(ValueError, match="finite"):
        action_batch_to_record_batch(
            sim_id="sim-1",
            start_step_index=1,
            actions=np.array([[[float("nan")]]], dtype=np.float32),
        )


def test_observation_batch_rejects_wrong_camera_dtype() -> None:
    from c5r_sim_inference.protocol import observation_batch_to_record_batch

    with pytest.raises(ValueError, match="camera must have dtype uint8"):
        observation_batch_to_record_batch(
            sim_id="sim-1",
            step_indices=np.array([0], dtype=np.int64),
            view_names=("overhead",),
            camera=np.zeros((1, 1, 1, 2, 3, 3), dtype=np.float32),
            qpos=np.zeros((1, 1, 2), dtype=np.float32),
            qvel=np.zeros((1, 1, 2), dtype=np.float32),
            ctrl=np.zeros((1, 1, 2), dtype=np.float32),
        )
