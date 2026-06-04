from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.flight as flight

from .protocol import (
    DEFAULT_VIEW_NAMES,
    client_request_action_batch_to_record_batch,
    client_request_new_sim_batch_to_record_batch,
    record_batch_to_observations,
)
from .types import Observation


class C5RSimClient:
    """Apache Flight client for creating and stepping C5R simulation streams."""

    def __init__(
        self,
        location: str | flight.Location,
        *,
        flight_client: flight.FlightClient | None = None,
    ) -> None:
        """Create a client for a Flight server location.

        Args:
            location: Flight URI or prebuilt ``flight.Location``.
            flight_client: Optional injected client, mainly for tests. When omitted,
                this instance owns and closes the created Flight client.
        """
        self._owns_client = flight_client is None
        self._client = flight_client or flight.FlightClient(location)

    def start_sim(
        self,
        *,
        task_id: str,
        initial_state: np.ndarray,
        views: tuple[str, ...] = DEFAULT_VIEW_NAMES,
        width: int,
        height: int,
        substeps: int,
        action_dim: int | None = None,
        scene_seed: int | None = None,
        physics_randomization: bool | str | None = None,
        rack_dynamic: bool = True,
        tube_radius: float | None = None,
        tube_half_length: float | None = None,
        tube_mass_kg: float | None = None,
        rack_contact_friction: tuple[float, float, float] = (0.65, 0.006, 0.0001),
        rack_contact_solref: tuple[float, float] = (0.006, 1.0),
        rack_contact_solimp: tuple[float, float, float] = (0.9, 0.95, 0.001),
        tube_contact_friction: tuple[float, float, float] = (0.8, 0.006, 0.0001),
        tube_contact_solref: tuple[float, float] = (0.006, 1.0),
        tube_contact_solimp: tuple[float, float, float] = (0.9, 0.95, 0.001),
        table_x_min: float = -0.08,
        table_x_max: float = 0.26,
        table_y_min: float = -0.20,
        table_y_max: float = 0.24,
        table_top_z: float = 0.760,
        rack_yaw_range: tuple[float, float] = (-3.141592653589793, 3.141592653589793),
        tube_starts_in_rack_hole: bool | None = None,
        tube_mode: str | None = None,
        tube_hole_index: int | str | None = "random",
        tube_table_spawn_height_m: float = 0.005,
        tube_table_roll_range: tuple[float, float] = (1.25, 1.9),
        tube_table_pitch_range: tuple[float, float] = (-0.35, 0.35),
    ) -> SimulationStream:
        """Start one interactive simulation stream.

        The initial request sends the task id, explicit robot state, desired camera
        views, render size, and simulation substep count. The server responds with
        the first observation at step index ``0``.

        Args:
            task_id: Server-configured task id used to resolve the scene.
            initial_state: Rank-2 NumPy array with shape ``(env, state_dim)``.
            views: Camera view names to render for every observation. Defaults to all views.
            width: Render width in pixels.
            height: Render height in pixels.
            substeps: Physics substeps to run for each action.
            action_dim: Optional action width. Defaults to ``initial_state.shape[1]``.
            scene_seed: Optional seed enabling randomized rack/tube scene generation.

        Returns:
            A live ``SimulationStream``. Close it when finished, or use it as a
            context manager.
        """
        _require_numpy_array(initial_state, name="initial_state")
        writer, reader = self._client.do_exchange(
            flight.FlightDescriptor.for_command(b"c5r_sim")
        )
        if action_dim is None:
            if initial_state.ndim != 2:
                raise ValueError("initial_state must have shape (env, state_dim)")
            action_dim = int(initial_state.shape[1])
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if initial_state.ndim != 2:
            raise ValueError("initial_state must have shape (env, state_dim)")
        initial_batch = client_request_new_sim_batch_to_record_batch(
            task_id=task_id,
            initial_state=initial_state,
            views=views,
            width=width,
            height=height,
            substeps=substeps,
            action_shape=(int(initial_state.shape[0]), action_dim),
            scene_seed=scene_seed,
            physics_randomization=physics_randomization,
            rack_dynamic=rack_dynamic,
            tube_radius=tube_radius,
            tube_half_length=tube_half_length,
            tube_mass_kg=tube_mass_kg,
            rack_contact_friction=rack_contact_friction,
            rack_contact_solref=rack_contact_solref,
            rack_contact_solimp=rack_contact_solimp,
            tube_contact_friction=tube_contact_friction,
            tube_contact_solref=tube_contact_solref,
            tube_contact_solimp=tube_contact_solimp,
            table_x_min=table_x_min,
            table_x_max=table_x_max,
            table_y_min=table_y_min,
            table_y_max=table_y_max,
            table_top_z=table_top_z,
            rack_yaw_range=rack_yaw_range,
            tube_starts_in_rack_hole=tube_starts_in_rack_hole,
            tube_mode=tube_mode,
            tube_hole_index=tube_hole_index,
            tube_table_spawn_height_m=tube_table_spawn_height_m,
            tube_table_roll_range=tube_table_roll_range,
            tube_table_pitch_range=tube_table_pitch_range,
        )
        _begin_if_available(writer, initial_batch.schema)
        writer.write_batch(initial_batch)
        observation = _read_observation(reader)
        return SimulationStream(
            sim_id=observation.sim_id,
            initial_observation=observation,
            writer=writer,
            reader=reader,
            request_schema=initial_batch.schema,
            owner=self if self._owns_client else None,
        )

    def close(self) -> None:
        """Close the underlying Flight client when this instance owns it."""
        close = getattr(self._client, "close", None)
        if close is not None:
            close()


@dataclass
class SimulationStream:
    """Live bidirectional Flight stream for a single server-side simulation."""

    sim_id: str
    initial_observation: Observation
    writer: Any
    reader: Any
    request_schema: pa.Schema
    owner: C5RSimClient | None = None
    next_step_index: int = 1
    _closed: bool = False

    def step(self, actions: np.ndarray) -> Observation:
        """Send a variable-size action batch and read the matching observations.

        Args:
            actions: Rank-3 NumPy array with shape ``(step, env, action_dim)``.

        Returns:
            An ``Observation`` whose leading dimension matches the action rows sent.
        """
        _require_numpy_array(actions, name="actions")
        action_batch = client_request_action_batch_to_record_batch(
            sim_id=self.sim_id,
            start_step_index=self.next_step_index,
            actions=actions,
            schema=self.request_schema,
        )
        self.writer.write_batch(action_batch)
        observation = _read_observation(self.reader)
        row_count = int(actions.shape[0])
        if observation.sim_id != self.sim_id:
            raise ValueError(
                f"observation sim_id {observation.sim_id!r} does not match {self.sim_id!r}"
            )
        if observation.camera.shape[0] != row_count:
            raise ValueError(
                "observation row count "
                f"{observation.camera.shape[0]} does not match action batch {row_count}"
            )
        self.next_step_index += row_count
        return observation

    def close(self) -> None:
        """Finish the Flight exchange and close stream resources."""
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        for target, method in (
            (self.writer, "done_writing"),
            (self.writer, "close"),
            (self.reader, "close"),
        ):
            if target is None:
                continue
            close = getattr(target, method, None)
            if close is None:
                continue
            try:
                close()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    def __enter__(self) -> SimulationStream:
        """Return this stream for ``with`` statement usage."""
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        """Close stream resources when leaving a ``with`` block."""
        try:
            self.close()
        except BaseException as close_error:
            if exc is not None:
                exc.__context__ = close_error
                return False
            raise
        return False


def _read_observation(reader: Any) -> Observation:
    """Read one observation record batch from a Flight exchange reader."""
    chunk = reader.read_chunk()
    batch = getattr(chunk, "data", chunk)
    if not isinstance(batch, pa.RecordBatch):
        raise TypeError(f"Flight reader returned {type(batch).__name__}")
    return Observation.from_batch(record_batch_to_observations(batch))


def _begin_if_available(writer: Any, schema: pa.Schema) -> None:
    """Begin a writer with ``schema`` when the writer exposes ``begin``."""
    begin = getattr(writer, "begin", None)
    if begin is not None:
        begin(schema)


def _require_numpy_array(values: Any, *, name: str) -> None:
    """Require SDK callers to pass NumPy arrays without framework conversion."""
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
