from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.flight as flight
import torch

from .protocol import (
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
        initial_state: torch.Tensor,
        views: tuple[str, ...],
        width: int,
        height: int,
        substeps: int,
        action_dim: int | None = None,
    ) -> SimulationStream:
        """Start one interactive simulation stream.

        The initial request sends the task id, explicit robot state, desired camera
        views, render size, and simulation substep count. The server responds with
        the first observation at step index ``0``.

        Args:
            task_id: Server-configured task id used to resolve the scene.
            initial_state: Rank-1 CPU/GPU PyTorch tensor containing the robot state.
            views: Camera view names to render for every observation.
            width: Render width in pixels.
            height: Render height in pixels.
            substeps: Physics substeps to run for each action.
            action_dim: Optional action width. Defaults to ``len(initial_state)``.

        Returns:
            A live ``SimulationStream``. Close it when finished, or use it as a
            context manager.
        """
        writer, reader = self._client.do_exchange(
            flight.FlightDescriptor.for_command(b"c5r_sim")
        )
        initial_state_values = initial_state.detach().contiguous().cpu().numpy()
        if action_dim is None:
            action_dim = int(initial_state_values.shape[0])
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        initial_batch = client_request_new_sim_batch_to_record_batch(
            task_id=task_id,
            initial_state=initial_state_values,
            views=views,
            width=width,
            height=height,
            substeps=substeps,
            action_shape=(action_dim,),
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

    def step(self, actions: torch.Tensor) -> Observation:
        """Send a variable-size action batch and read the matching observations.

        Args:
            actions: Rank-2 PyTorch tensor with shape ``(batch, action_dim)``.

        Returns:
            A ``TorchObservation`` whose leading dimension matches the number of
            action rows sent.
        """
        action_values = actions.detach().contiguous().cpu().numpy()
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
            (self.owner, "close"),
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
