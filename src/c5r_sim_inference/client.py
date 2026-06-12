from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
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
        location: str | flight.Location | None = None,
        *,
        flight_client: flight.FlightClient | None = None,
        control_url: str | None = None,
        http_client: Any | None = None,
        flight_client_factory: Any | None = None,
    ) -> None:
        """Create a client for a Flight server location.

        Args:
            location: Flight URI or prebuilt ``flight.Location``.
            flight_client: Optional injected client, mainly for tests. When omitted,
                this instance owns and closes the created Flight client.
            control_url: Optional FastAPI control server URL. When provided,
                ``start_sim`` launches a fresh server session and connects to the
                returned Flight URI.
            http_client: Optional injected HTTP client, mainly for tests.
            flight_client_factory: Optional factory for creating Flight clients.
        """
        if location is None and control_url is None and flight_client is None:
            raise ValueError("location or control_url is required")
        if control_url is not None and flight_client is not None:
            raise ValueError("flight_client cannot be combined with control_url")
        self._location = location
        self._control_url = control_url.rstrip("/") if control_url is not None else None
        self._flight_client_factory = flight_client_factory or flight.FlightClient
        self._owns_client = flight_client is None and self._control_url is None
        self._client = (
            flight_client
            if flight_client is not None
            else (
                None
                if self._control_url is not None
                else self._flight_client_factory(location)
            )
        )
        self._owns_http_client = http_client is None and self._control_url is not None
        self._http_client = (
            http_client
            if http_client is not None
            else (httpx.Client() if self._control_url is not None else None)
        )

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
            scene_seed: Optional reproducibility seed for physics randomization.
            physics_randomization: Server-owned physics profile. ``True`` uses
                the default profile; strings select named profiles such as
                ``"low"``, ``"default"``, or ``"high"``.

        Returns:
            A live ``SimulationStream``. Close it when finished, or use it as a
            context manager.
        """
        _require_numpy_array(initial_state, name="initial_state")
        client = self._client
        control_session: ControlSession | None = None
        flight_client_to_close: Any | None = None
        if self._control_url is not None:
            control_session = self._launch_control_session()
            client = self._flight_client_factory(control_session.grpc_address)
            flight_client_to_close = client
        if client is None:
            raise RuntimeError("Flight client is not initialized")
        writer, reader = client.do_exchange(
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
            flight_client=flight_client_to_close,
            control_session=control_session,
        )

    def close(self) -> None:
        """Close the underlying Flight client when this instance owns it."""
        close = getattr(self._client, "close", None)
        if close is not None:
            close()
        if self._owns_http_client:
            http_close = getattr(self._http_client, "close", None)
            if http_close is not None:
                http_close()

    def _launch_control_session(self) -> ControlSession:
        if self._control_url is None or self._http_client is None:
            raise RuntimeError("control server is not configured")
        url = f"{self._control_url}/sessions"
        try:
            response = self._http_client.post(url)
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise RuntimeError("failed to launch C5R simulation session") from exc
        session_id = payload.get("session_id")
        grpc_address = payload.get("grpc_address")
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError("control server response is missing session_id")
        if not isinstance(grpc_address, str) or not grpc_address:
            raise RuntimeError("control server response is missing grpc_address")
        return ControlSession(
            control_url=self._control_url,
            session_id=session_id,
            grpc_address=grpc_address,
            http_client=self._http_client,
        )


@dataclass
class ControlSession:
    control_url: str
    session_id: str
    grpc_address: str
    http_client: Any

    def close(self) -> None:
        response = self.http_client.delete(
            f"{self.control_url}/sessions/{self.session_id}"
        )
        try:
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"failed to close C5R simulation session {self.session_id}"
            ) from exc


@dataclass
class SimulationStream:
    """Live bidirectional Flight stream for a single server-side simulation."""

    sim_id: str
    initial_observation: Observation
    writer: Any
    reader: Any
    request_schema: pa.Schema
    owner: C5RSimClient | None = None
    flight_client: Any | None = None
    control_session: ControlSession | None = None
    next_step_index: int = 1
    _closed: bool = False

    def step(self, actions: np.ndarray) -> Observation:
        """Send a variable-size action batch and read the matching observations.

        Args:
            actions: Rank-3 NumPy array with shape ``(env, step, action_dim)``.

        Returns:
            An ``Observation`` whose leading dimension matches the requested steps.
        """
        _require_numpy_array(actions, name="actions")
        action_values = _env_major_actions_to_step_major(actions)
        action_batch = client_request_action_batch_to_record_batch(
            sim_id=self.sim_id,
            start_step_index=self.next_step_index,
            actions=action_values,
            schema=self.request_schema,
        )
        self.writer.write_batch(action_batch)
        observation = _read_observation(self.reader)
        row_count = int(action_values.shape[0])
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
<<<<<<< HEAD
            (self.flight_client, "close"),
            (self.owner, "close"),
            (self.control_session, "close"),
=======
>>>>>>> 5b268b3a85cf43cb77b75f505d85f9a84f4a7b0d
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


def _env_major_actions_to_step_major(actions: np.ndarray) -> np.ndarray:
    """Convert SDK action batches from (env, step, action_dim) to protocol shape."""

    if actions.ndim != 3:
        raise ValueError("actions must have shape (env, step, action_dim)")
    return np.array(np.swapaxes(actions, 0, 1), dtype=actions.dtype, order="C", copy=True)
