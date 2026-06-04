from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import numpy as np
import pyarrow as pa

METADATA_KEY = b"c5r_sim"


@dataclass(frozen=True)
class NewSimRequest:
    task_id: str
    initial_state: np.ndarray
    views: tuple[str, ...]
    width: int
    height: int
    substeps: int


@dataclass(frozen=True)
class ActionBatchRequest:
    sim_id: str
    start_step_index: int
    actions: np.ndarray


@dataclass(frozen=True)
class ObservationBatch:
    sim_id: str
    step_indices: np.ndarray
    view_names: tuple[str, ...]
    camera: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray


def fixed_shape_tensor_array(
    values: np.ndarray, *, value_shape: tuple[int, ...], name: str
) -> pa.FixedShapeTensorArray:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    expected_rank = len(value_shape) + 1
    if values.ndim != expected_rank:
        raise ValueError(f"{name} must have rank {expected_rank}, got {values.ndim}")
    if tuple(values.shape[1:]) != value_shape:
        raise ValueError(
            f"{name} must have trailing shape {value_shape}, got {tuple(values.shape[1:])}"
        )
    if not values.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if values.shape[0] == 1 and values.strides[0] == 0:
        values = values.copy(order="C")
    return pa.FixedShapeTensorArray.from_numpy_ndarray(values)


def fixed_shape_tensor_to_numpy(values: pa.Array, *, name: str) -> np.ndarray:
    if not isinstance(values, pa.FixedShapeTensorArray):
        raise ValueError(f"{name} must be a fixed-shape tensor array")
    if values.null_count:
        raise ValueError(f"{name} must not contain null tensor rows")
    array = values.to_numpy_ndarray()
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return array


def new_sim_batch_to_record_batch(
    *,
    task_id: str,
    initial_state: np.ndarray,
    views: tuple[str, ...],
    width: int,
    height: int,
    substeps: int,
) -> pa.RecordBatch:
    _require_non_empty_string(task_id, name="task_id")
    views = _metadata_views({"views": views}, default_if_missing=False)
    width = _positive_int(width, name="width")
    height = _positive_int(height, name="height")
    substeps = _positive_int(substeps, name="substeps")
    state = np.asarray(initial_state)
    if state.ndim != 1:
        raise ValueError("initial_state must be a 1D array")
    if state.dtype != np.float32:
        state = state.astype(np.float32)
    if not np.all(np.isfinite(state)):
        raise ValueError("initial_state must contain only finite values")
    row_state = np.ascontiguousarray(state.reshape(1, *state.shape))
    initial_state_array = fixed_shape_tensor_array(
        row_state, value_shape=tuple(row_state.shape[1:]), name="initial_state"
    )
    schema = pa.schema(
        [
            pa.field("task_id", pa.string(), nullable=False),
            pa.field("initial_state", initial_state_array.type, nullable=False),
        ],
        metadata=_metadata(
            op="new_sim",
            views=list(views),
            width=width,
            height=height,
            substeps=substeps,
            tensors={"initial_state": _tensor_metadata(row_state)},
        ),
    )
    return pa.RecordBatch.from_arrays(
        [pa.array([task_id], type=pa.string()), initial_state_array], schema=schema
    )


def record_batch_to_new_sim(batch: pa.RecordBatch) -> NewSimRequest:
    metadata = _metadata_payload(batch, context="new_sim metadata")
    if metadata.get("op") == "client_requests":
        return _client_request_record_batch_to_new_sim(batch, metadata=metadata)
    _require_columns(batch, ("task_id", "initial_state"))
    if batch.num_rows != 1:
        raise ValueError("new_sim batch must contain exactly one row")
    _require_metadata_op(metadata, expected="new_sim", label="new_sim")
    task_id = batch.column("task_id")[0].as_py()
    _require_non_empty_string(task_id, name="task_id")
    initial_state_column = batch.column("initial_state")
    _require_fixed_shape_tensor_value_type(
        initial_state_column, dtype=pa.float32(), name="initial_state"
    )
    initial_state = fixed_shape_tensor_to_numpy(
        initial_state_column, name="initial_state"
    )[0]
    if initial_state.ndim != 1:
        raise ValueError("initial_state must be a 1D array")
    if not np.all(np.isfinite(initial_state)):
        raise ValueError("initial_state must contain only finite values")
    return NewSimRequest(
        task_id=task_id,
        initial_state=np.ascontiguousarray(initial_state),
        views=_metadata_views(metadata, default_if_missing=False),
        width=_positive_int(metadata.get("width"), name="width"),
        height=_positive_int(metadata.get("height"), name="height"),
        substeps=_positive_int(metadata.get("substeps"), name="substeps"),
    )


def action_batch_to_record_batch(
    *, sim_id: str, start_step_index: int, actions: np.ndarray
) -> pa.RecordBatch:
    _require_non_empty_string(sim_id, name="sim_id")
    start_step_index = _non_negative_int(start_step_index, name="start_step_index")
    action_values = np.asarray(actions)
    if action_values.ndim != 2:
        raise ValueError("actions must have shape (batch, action_dim)")
    if action_values.shape[0] == 0:
        raise ValueError("actions batch must not be empty")
    if action_values.dtype != np.float32:
        action_values = action_values.astype(np.float32)
    if not np.all(np.isfinite(action_values)):
        raise ValueError("actions must contain only finite values")
    action_values = np.ascontiguousarray(action_values)
    action_array = fixed_shape_tensor_array(
        action_values, value_shape=tuple(action_values.shape[1:]), name="actions"
    )
    schema = pa.schema(
        [
            pa.field("sim_id", pa.string(), nullable=False),
            pa.field("start_step_index", pa.int64(), nullable=False),
            pa.field("action", action_array.type, nullable=False),
        ],
        metadata=_metadata(op="step", tensors={"action": _tensor_metadata(action_values)}),
    )
    return pa.RecordBatch.from_arrays(
        [
            pa.array([sim_id] * action_values.shape[0], type=pa.string()),
            pa.array([start_step_index] * action_values.shape[0], type=pa.int64()),
            action_array,
        ],
        schema=schema,
    )


def client_request_new_sim_batch_to_record_batch(
    *,
    task_id: str,
    initial_state: np.ndarray,
    views: tuple[str, ...],
    width: int,
    height: int,
    substeps: int,
    action_shape: tuple[int, ...],
) -> pa.RecordBatch:
    _require_non_empty_string(task_id, name="task_id")
    views = _metadata_views({"views": views}, default_if_missing=False)
    width = _positive_int(width, name="width")
    height = _positive_int(height, name="height")
    substeps = _positive_int(substeps, name="substeps")
    action_shape = _validated_value_shape(action_shape, name="action_shape")
    state = np.asarray(initial_state)
    if state.ndim != 1:
        raise ValueError("initial_state must be a 1D array")
    if state.dtype != np.float32:
        state = state.astype(np.float32)
    if not np.all(np.isfinite(state)):
        raise ValueError("initial_state must contain only finite values")
    row_state = np.ascontiguousarray(state.reshape(1, *state.shape))
    initial_state_array = fixed_shape_tensor_array(
        row_state, value_shape=tuple(row_state.shape[1:]), name="initial_state"
    )
    action_type = _fixed_shape_tensor_type(np.float32, action_shape)
    metadata = _metadata(
        op="client_requests",
        views=list(views),
        width=width,
        height=height,
        substeps=substeps,
        tensors={
            "initial_state": _tensor_metadata(row_state),
            "action": {"dtype": "float32", "shape": [1, *action_shape]},
        },
    )
    schema = _client_request_schema(
        initial_state_type=initial_state_array.type,
        action_type=action_type,
        metadata=metadata,
    )
    return pa.RecordBatch.from_arrays(
        [
            pa.array(["new_sim"], type=pa.string()),
            pa.array([task_id], type=pa.string()),
            pa.array([None], type=pa.string()),
            pa.array([None], type=pa.int64()),
            initial_state_array,
            _null_fixed_shape_tensor_array(action_type, row_count=1),
        ],
        schema=schema,
    )


def client_request_action_batch_to_record_batch(
    *,
    sim_id: str,
    start_step_index: int,
    actions: np.ndarray,
    schema: pa.Schema,
) -> pa.RecordBatch:
    _require_non_empty_string(sim_id, name="sim_id")
    start_step_index = _non_negative_int(start_step_index, name="start_step_index")
    action_values = np.asarray(actions)
    if action_values.ndim != 2:
        raise ValueError("actions must have shape (batch, action_dim)")
    if action_values.shape[0] == 0:
        raise ValueError("actions batch must not be empty")
    if action_values.dtype != np.float32:
        action_values = action_values.astype(np.float32)
    if not np.all(np.isfinite(action_values)):
        raise ValueError("actions must contain only finite values")
    action_values = np.ascontiguousarray(action_values)
    _require_client_request_schema(schema)
    action_array = fixed_shape_tensor_array(
        action_values, value_shape=tuple(action_values.shape[1:]), name="actions"
    )
    expected_action_type = schema.field("action").type
    if action_array.type != expected_action_type:
        raise ValueError("actions shape does not match stream request schema")
    initial_state_type = schema.field("initial_state").type
    return pa.RecordBatch.from_arrays(
        [
            pa.array(["step"] * action_values.shape[0], type=pa.string()),
            pa.array([None] * action_values.shape[0], type=pa.string()),
            pa.array([sim_id] * action_values.shape[0], type=pa.string()),
            pa.array([start_step_index] * action_values.shape[0], type=pa.int64()),
            _null_fixed_shape_tensor_array(
                initial_state_type, row_count=action_values.shape[0]
            ),
            action_array,
        ],
        schema=schema,
    )


def record_batch_to_actions(
    batch: pa.RecordBatch, *, expected_sim_id: str | None = None
) -> ActionBatchRequest:
    metadata = _metadata_payload(batch, context="actions metadata")
    if metadata.get("op") == "client_requests":
        return _client_request_record_batch_to_actions(
            batch, metadata=metadata, expected_sim_id=expected_sim_id
        )
    _require_columns(batch, ("sim_id", "start_step_index", "action"))
    if batch.num_rows == 0:
        raise ValueError("actions batch must not be empty")
    _require_metadata_op(metadata, expected="step", label="actions")
    sim_ids = batch.column("sim_id").to_pylist()
    for value in sim_ids:
        _require_non_empty_string(value, name="sim_id")
    sim_id = sim_ids[0]
    if any(value != sim_id for value in sim_ids):
        raise ValueError("actions batch must contain one sim_id")
    if expected_sim_id is not None and sim_id != expected_sim_id:
        raise ValueError(f"actions sim_id must match {expected_sim_id}")
    start_step_index_column = batch.column("start_step_index")
    _require_arrow_int64_column(start_step_index_column, name="start_step_index")
    start_indices = start_step_index_column.to_pylist()
    for value in start_indices:
        _non_negative_int(value, name="start_step_index")
    start_step_index = int(start_indices[0])
    if any(int(value) != start_step_index for value in start_indices):
        raise ValueError("actions batch must contain one start_step_index")
    action_column = batch.column("action")
    _require_fixed_shape_tensor_value_type(
        action_column, dtype=pa.float32(), name="actions"
    )
    actions = fixed_shape_tensor_to_numpy(action_column, name="actions")
    if actions.ndim != 2:
        raise ValueError("actions must have shape (batch, action_dim)")
    if not np.all(np.isfinite(actions)):
        raise ValueError("actions must contain only finite values")
    return ActionBatchRequest(
        sim_id=sim_id,
        start_step_index=start_step_index,
        actions=np.ascontiguousarray(actions),
    )


def _client_request_record_batch_to_new_sim(
    batch: pa.RecordBatch, *, metadata: dict[str, Any]
) -> NewSimRequest:
    _require_client_request_schema(batch.schema)
    _require_columns(batch, ("op", "task_id", "initial_state"))
    if batch.num_rows != 1:
        raise ValueError("new_sim batch must contain exactly one row")
    _require_op_column(batch, expected="new_sim")
    task_id = batch.column("task_id")[0].as_py()
    _require_non_empty_string(task_id, name="task_id")
    initial_state_column = batch.column("initial_state")
    _require_fixed_shape_tensor_value_type(
        initial_state_column, dtype=pa.float32(), name="initial_state"
    )
    initial_state = fixed_shape_tensor_to_numpy(
        initial_state_column, name="initial_state"
    )[0]
    if initial_state.ndim != 1:
        raise ValueError("initial_state must be a 1D array")
    if not np.all(np.isfinite(initial_state)):
        raise ValueError("initial_state must contain only finite values")
    return NewSimRequest(
        task_id=task_id,
        initial_state=np.ascontiguousarray(initial_state),
        views=_metadata_views(metadata, default_if_missing=False),
        width=_positive_int(metadata.get("width"), name="width"),
        height=_positive_int(metadata.get("height"), name="height"),
        substeps=_positive_int(metadata.get("substeps"), name="substeps"),
    )


def _client_request_record_batch_to_actions(
    batch: pa.RecordBatch,
    *,
    metadata: dict[str, Any],
    expected_sim_id: str | None,
) -> ActionBatchRequest:
    _require_client_request_schema(batch.schema)
    _require_columns(batch, ("op", "sim_id", "start_step_index", "action"))
    if batch.num_rows == 0:
        raise ValueError("actions batch must not be empty")
    _require_op_column(batch, expected="step")
    sim_ids = batch.column("sim_id").to_pylist()
    for value in sim_ids:
        _require_non_empty_string(value, name="sim_id")
    sim_id = sim_ids[0]
    if any(value != sim_id for value in sim_ids):
        raise ValueError("actions batch must contain one sim_id")
    if expected_sim_id is not None and sim_id != expected_sim_id:
        raise ValueError(f"actions sim_id must match {expected_sim_id}")
    start_step_index_column = batch.column("start_step_index")
    _require_arrow_int64_column(start_step_index_column, name="start_step_index")
    start_indices = start_step_index_column.to_pylist()
    for value in start_indices:
        _non_negative_int(value, name="start_step_index")
    start_step_index = int(start_indices[0])
    if any(int(value) != start_step_index for value in start_indices):
        raise ValueError("actions batch must contain one start_step_index")
    action_column = batch.column("action")
    _require_fixed_shape_tensor_value_type(
        action_column, dtype=pa.float32(), name="actions"
    )
    actions = fixed_shape_tensor_to_numpy(action_column, name="actions")
    if actions.ndim != 2:
        raise ValueError("actions must have shape (batch, action_dim)")
    if not np.all(np.isfinite(actions)):
        raise ValueError("actions must contain only finite values")
    _require_metadata_op(metadata, expected="client_requests", label="client request")
    return ActionBatchRequest(
        sim_id=sim_id,
        start_step_index=start_step_index,
        actions=np.ascontiguousarray(actions),
    )


def observation_batch_to_record_batch(
    *,
    sim_id: str,
    step_indices: np.ndarray,
    view_names: tuple[str, ...],
    camera: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
    ctrl: np.ndarray,
) -> pa.RecordBatch:
    _require_non_empty_string(sim_id, name="sim_id")
    view_names = _metadata_views({"views": view_names}, default_if_missing=False)
    steps = _validated_step_indices(step_indices)
    camera_values = _contiguous_array(camera, "camera")
    qpos_values = _contiguous_array(qpos, "qpos")
    qvel_values = _contiguous_array(qvel, "qvel")
    ctrl_values = _contiguous_array(ctrl, "ctrl")
    _require_array_dtype(camera_values, dtype=np.uint8, name="camera")
    _require_array_dtype(qpos_values, dtype=np.float32, name="qpos")
    _require_array_dtype(qvel_values, dtype=np.float32, name="qvel")
    _require_array_dtype(ctrl_values, dtype=np.float32, name="ctrl")
    _validate_observation_tensor_shapes(
        camera_values, qpos_values, qvel_values, ctrl_values
    )
    row_count = steps.shape[0]
    if row_count == 0:
        raise ValueError("observation batch must not be empty")
    if camera_values.shape[0] != row_count:
        raise ValueError("camera row count must match step_indices row count")
    for name, values in (
        ("qpos", qpos_values),
        ("qvel", qvel_values),
        ("ctrl", ctrl_values),
    ):
        if values.shape[0] != row_count:
            raise ValueError(f"{name} row count must match step_indices row count")
    _validate_observation_views(view_names, camera_values)
    camera_array = fixed_shape_tensor_array(
        camera_values, value_shape=tuple(camera_values.shape[1:]), name="camera"
    )
    qpos_array = fixed_shape_tensor_array(
        qpos_values, value_shape=tuple(qpos_values.shape[1:]), name="qpos"
    )
    qvel_array = fixed_shape_tensor_array(
        qvel_values, value_shape=tuple(qvel_values.shape[1:]), name="qvel"
    )
    ctrl_array = fixed_shape_tensor_array(
        ctrl_values, value_shape=tuple(ctrl_values.shape[1:]), name="ctrl"
    )
    schema = pa.schema(
        [
            pa.field("sim_id", pa.string(), nullable=False),
            pa.field("step_index", pa.int64(), nullable=False),
            pa.field("camera", camera_array.type, nullable=False),
            pa.field("qpos", qpos_array.type, nullable=False),
            pa.field("qvel", qvel_array.type, nullable=False),
            pa.field("ctrl", ctrl_array.type, nullable=False),
        ],
        metadata=_metadata(
            op="observations",
            views=list(view_names),
            tensors={
                "camera": _tensor_metadata(camera_values),
                "qpos": _tensor_metadata(qpos_values),
                "qvel": _tensor_metadata(qvel_values),
                "ctrl": _tensor_metadata(ctrl_values),
            },
        ),
    )
    return pa.RecordBatch.from_arrays(
        [
            pa.array([sim_id] * row_count, type=pa.string()),
            pa.array(steps, type=pa.int64()),
            camera_array,
            qpos_array,
            qvel_array,
            ctrl_array,
        ],
        schema=schema,
    )


def record_batch_to_observations(batch: pa.RecordBatch) -> ObservationBatch:
    _require_columns(batch, ("sim_id", "step_index", "camera", "qpos", "qvel", "ctrl"))
    if batch.num_rows == 0:
        raise ValueError("observation batch must not be empty")
    sim_ids = batch.column("sim_id").to_pylist()
    for value in sim_ids:
        _require_non_empty_string(value, name="sim_id")
    sim_id = sim_ids[0]
    if any(value != sim_id for value in sim_ids):
        raise ValueError("observation batch must contain one sim_id")
    metadata = _metadata_payload(batch, context="observation metadata")
    _require_metadata_op(metadata, expected="observations", label="observations")
    step_index_column = batch.column("step_index")
    _require_arrow_int64_column(step_index_column, name="step_index")
    step_indices = _validated_step_indices(step_index_column.to_pylist())
    view_names = _metadata_views(metadata, default_if_missing=False)
    camera = fixed_shape_tensor_to_numpy(batch.column("camera"), name="camera")
    qpos = fixed_shape_tensor_to_numpy(batch.column("qpos"), name="qpos")
    qvel = fixed_shape_tensor_to_numpy(batch.column("qvel"), name="qvel")
    ctrl = fixed_shape_tensor_to_numpy(batch.column("ctrl"), name="ctrl")
    _require_array_dtype(camera, dtype=np.uint8, name="camera")
    _require_array_dtype(qpos, dtype=np.float32, name="qpos")
    _require_array_dtype(qvel, dtype=np.float32, name="qvel")
    _require_array_dtype(ctrl, dtype=np.float32, name="ctrl")
    _validate_observation_tensor_shapes(camera, qpos, qvel, ctrl)
    _validate_observation_views(view_names, camera)
    return ObservationBatch(
        sim_id=sim_id,
        step_indices=np.ascontiguousarray(step_indices),
        view_names=view_names,
        camera=camera,
        qpos=qpos,
        qvel=qvel,
        ctrl=ctrl,
    )


def _metadata(**payload: Any) -> dict[bytes, bytes]:
    return {METADATA_KEY: json.dumps(payload, sort_keys=True).encode("utf-8")}


def _metadata_payload(batch: pa.RecordBatch, *, context: str) -> dict[str, Any]:
    metadata = batch.schema.metadata or {}
    raw = metadata.get(METADATA_KEY)
    if raw is None:
        raise ValueError(f"{context} must include c5r_sim metadata")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be a JSON object")
    return payload


def _require_metadata_op(metadata: dict[str, Any], *, expected: str, label: str) -> None:
    if metadata.get("op") != expected:
        raise ValueError(f"{label} metadata op must be {expected!r}")


def _client_request_schema(
    *,
    initial_state_type: pa.DataType,
    action_type: pa.DataType,
    metadata: dict[bytes, bytes],
) -> pa.Schema:
    return pa.schema(
        [
            pa.field("op", pa.string(), nullable=False),
            pa.field("task_id", pa.string(), nullable=True),
            pa.field("sim_id", pa.string(), nullable=True),
            pa.field("start_step_index", pa.int64(), nullable=True),
            pa.field("initial_state", initial_state_type, nullable=True),
            pa.field("action", action_type, nullable=True),
        ],
        metadata=metadata,
    )


def _require_client_request_schema(schema: pa.Schema) -> None:
    metadata = schema.metadata or {}
    raw = metadata.get(METADATA_KEY)
    if raw is None:
        raise ValueError("client request metadata must include c5r_sim metadata")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("client request metadata must be valid JSON") from exc
    _require_metadata_op(payload, expected="client_requests", label="client request")
    _require_columns_for_schema(
        schema,
        ("op", "task_id", "sim_id", "start_step_index", "initial_state", "action"),
    )


def _require_op_column(batch: pa.RecordBatch, *, expected: str) -> None:
    ops = batch.column("op").to_pylist()
    if any(value != expected for value in ops):
        raise ValueError(f"client request op must be {expected!r}")


def _fixed_shape_tensor_type(
    dtype: np.dtype[Any], value_shape: tuple[int, ...]
) -> pa.DataType:
    values = np.zeros((1, *value_shape), dtype=dtype)
    return fixed_shape_tensor_array(
        values,
        value_shape=value_shape,
        name="tensor_type",
    ).type


def _null_fixed_shape_tensor_array(
    tensor_type: pa.DataType, *, row_count: int
) -> pa.ExtensionArray:
    storage_type = getattr(tensor_type, "storage_type", None)
    if storage_type is None:
        raise ValueError("tensor_type must be a fixed-shape tensor type")
    storage = pa.array([None] * row_count, type=storage_type)
    return pa.ExtensionArray.from_storage(tensor_type, storage)


def _validated_value_shape(values: tuple[int, ...], *, name: str) -> tuple[int, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{name} must be a non-empty tuple")
    if any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in values):
        raise ValueError(f"{name} dimensions must be positive integers")
    return values


def _tensor_metadata(values: np.ndarray) -> dict[str, Any]:
    return {"dtype": values.dtype.name, "shape": list(values.shape)}


def _metadata_views(
    metadata: dict[str, Any], *, default_if_missing: bool
) -> tuple[str, ...]:
    if "views" not in metadata:
        if default_if_missing:
            return ()
        raise ValueError("views metadata is required")
    views = metadata["views"]
    if not isinstance(views, (list, tuple)):
        raise ValueError("views metadata must be a list")
    parsed: list[str] = []
    for view in views:
        _require_non_empty_string(view, name="view")
        parsed.append(view)
    if not parsed:
        raise ValueError("views must not be empty")
    return tuple(parsed)


def _require_columns(batch: pa.RecordBatch, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in batch.schema.names]
    if missing:
        raise ValueError(f"record batch missing columns: {', '.join(missing)}")


def _require_columns_for_schema(schema: pa.Schema, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in schema.names]
    if missing:
        raise ValueError(f"record batch missing columns: {', '.join(missing)}")


def _require_non_empty_string(value: Any, *, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _non_negative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _validated_step_indices(values: Any) -> np.ndarray:
    steps = np.asarray(values)
    if steps.ndim != 1:
        raise ValueError("step_indices must be a 1D array")
    if not np.issubdtype(steps.dtype, np.integer):
        raise ValueError("step_indices must contain integers")
    if np.any(steps < 0):
        raise ValueError("step_indices must be non-negative")
    return np.ascontiguousarray(steps, dtype=np.int64)


def _contiguous_array(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    if array.ndim < 2:
        raise ValueError(f"{name} must include a leading row dimension")
    return array


def _require_array_dtype(
    array: np.ndarray, *, dtype: np.dtype[Any], name: str
) -> None:
    expected = np.dtype(dtype)
    if array.dtype != expected:
        raise ValueError(f"{name} must have dtype {expected.name}")


def _require_fixed_shape_tensor_value_type(
    values: pa.Array, *, dtype: pa.DataType, name: str
) -> None:
    if not isinstance(values, pa.FixedShapeTensorArray):
        raise ValueError(f"{name} must be a fixed-shape tensor array")
    if values.type.value_type != dtype:
        expected = np.dtype(dtype.to_pandas_dtype()).name
        raise ValueError(f"{name} must have dtype {expected}")


def _require_arrow_int64_column(values: pa.Array, *, name: str) -> None:
    if values.type != pa.int64():
        raise ValueError(f"{name} must be an int64 Arrow column")


def _validate_observation_tensor_shapes(
    camera: np.ndarray, qpos: np.ndarray, qvel: np.ndarray, ctrl: np.ndarray
) -> None:
    if camera.ndim != 5:
        raise ValueError("camera must have shape (batch, views, height, width, channels)")
    if camera.shape[-1] != 3:
        raise ValueError("camera must have 3 channels")
    for name, values in (("qpos", qpos), ("qvel", qvel), ("ctrl", ctrl)):
        if values.ndim != 2:
            raise ValueError(f"{name} must have shape (batch, dim)")


def _validate_observation_views(view_names: tuple[str, ...], camera: np.ndarray) -> None:
    if len(view_names) != camera.shape[1]:
        raise ValueError("views length must match camera view dimension")
