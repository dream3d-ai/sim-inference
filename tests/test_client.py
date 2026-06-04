from __future__ import annotations

from collections import deque

import numpy as np
import pyarrow as pa
import pyarrow.flight as flight
import pytest


class FakeWriter:
    def __init__(self) -> None:
        self.batches: list[pa.RecordBatch] = []
        self.done = False
        self.closed = False

    def begin(self, schema: pa.Schema) -> None:
        self.schema = schema

    def write_batch(self, batch: pa.RecordBatch) -> None:
        self.batches.append(batch)

    def done_writing(self) -> None:
        self.done = True

    def close(self) -> None:
        self.closed = True


class FakeReader:
    def __init__(self, batches: list[pa.RecordBatch]) -> None:
        self._batches = deque(batches)
        self.closed = False

    def read_chunk(self):
        if not self._batches:
            raise StopIteration
        return type("Chunk", (), {"data": self._batches.popleft()})()

    def close(self) -> None:
        self.closed = True


class FakeFlightClient:
    def __init__(self, batches: list[pa.RecordBatch]) -> None:
        self.writer = FakeWriter()
        self.reader = FakeReader(batches)
        self.descriptors: list[flight.FlightDescriptor] = []
        self.closed = False

    def do_exchange(self, descriptor: flight.FlightDescriptor):
        self.descriptors.append(descriptor)
        return self.writer, self.reader

    def close(self) -> None:
        self.closed = True


def _observation_batch(*, sim_id: str, start: int, rows: int):
    from c5r_sim_inference.protocol import observation_batch_to_record_batch

    return observation_batch_to_record_batch(
        sim_id=sim_id,
        step_indices=np.arange(start, start + rows, dtype=np.int64),
        view_names=("overhead",),
        camera=np.zeros((rows, 1, 1, 2, 3, 3), dtype=np.uint8),
        qpos=np.ones((rows, 1, 2), dtype=np.float32),
        qvel=np.ones((rows, 1, 2), dtype=np.float32) * 2,
        ctrl=np.ones((rows, 1, 2), dtype=np.float32) * 3,
    )


def test_start_sim_writes_new_sim_and_reads_initial_observation() -> None:
    from c5r_sim_inference.client import C5RSimClient
    from c5r_sim_inference.protocol import record_batch_to_new_sim

    flight_client = FakeFlightClient([_observation_batch(sim_id="sim-1", start=0, rows=1)])
    client = C5RSimClient("grpc://unused", flight_client=flight_client)

    stream = client.start_sim(
        task_id="task_11",
        initial_state=np.zeros((1, 2), dtype=np.float32),
        views=("overhead",),
        width=3,
        height=2,
        substeps=1,
    )

    assert stream.sim_id == "sim-1"
    assert stream.next_step_index == 1
    assert stream.initial_observation.step_indices.tolist() == [0]
    assert stream.initial_observation.camera.dtype == np.uint8
    assert len(flight_client.writer.batches) == 1
    request = record_batch_to_new_sim(flight_client.writer.batches[0])
    assert request.task_id == "task_11"
    assert request.views == ("overhead",)


def test_start_sim_writes_scene_options() -> None:
    from c5r_sim_inference.client import C5RSimClient
    from c5r_sim_inference.protocol import record_batch_to_new_sim

    flight_client = FakeFlightClient([_observation_batch(sim_id="sim-1", start=0, rows=1)])

    C5RSimClient("grpc://unused", flight_client=flight_client).start_sim(
        task_id="task_12",
        initial_state=np.zeros((1, 2), dtype=np.float32),
        views=(),
        width=3,
        height=2,
        substeps=1,
        scene_seed=17,
        tube_mode="table",
        tube_mass_kg=0.02,
    )

    request = record_batch_to_new_sim(flight_client.writer.batches[0])

    assert request.scene_options is not None
    assert request.scene_options["seed"] == 17
    assert request.scene_options["physics"]["rack"]["tube_mass_kg"] == 0.02
    assert (
        request.scene_options["physics"]["randomization"]["placement"]["tube_mode"] == "table"
    )


def test_start_sim_writes_physics_randomization_profile() -> None:
    from c5r_sim_inference.client import C5RSimClient
    from c5r_sim_inference.protocol import record_batch_to_new_sim

    flight_client = FakeFlightClient([_observation_batch(sim_id="sim-1", start=0, rows=1)])

    C5RSimClient("grpc://unused", flight_client=flight_client).start_sim(
        task_id="task_11",
        initial_state=np.zeros((1, 2), dtype=np.float32),
        views=(),
        width=3,
        height=2,
        substeps=1,
        scene_seed=17,
        physics_randomization="default",
    )

    request = record_batch_to_new_sim(flight_client.writer.batches[0])

    assert request.scene_options is not None
    assert request.scene_options["physics_randomization"] == {"profile": "default"}


def test_start_sim_defaults_to_all_views() -> None:
    from c5r_sim_inference.client import C5RSimClient
    from c5r_sim_inference.protocol import DEFAULT_VIEW_NAMES, record_batch_to_new_sim

    flight_client = FakeFlightClient([_observation_batch(sim_id="sim-1", start=0, rows=1)])

    C5RSimClient("grpc://unused", flight_client=flight_client).start_sim(
        task_id="task_11",
        initial_state=np.zeros((1, 2), dtype=np.float32),
        width=3,
        height=2,
        substeps=1,
    )

    request = record_batch_to_new_sim(flight_client.writer.batches[0])

    assert request.views == DEFAULT_VIEW_NAMES


def test_step_writes_action_batch_and_returns_matching_observations() -> None:
    from c5r_sim_inference.client import C5RSimClient
    from c5r_sim_inference.protocol import record_batch_to_actions

    flight_client = FakeFlightClient(
        [
            _observation_batch(sim_id="sim-1", start=0, rows=1),
            _observation_batch(sim_id="sim-1", start=1, rows=3),
        ]
    )
    stream = C5RSimClient("grpc://unused", flight_client=flight_client).start_sim(
        task_id="task_11",
        initial_state=np.zeros((1, 2), dtype=np.float32),
        views=("overhead",),
        width=3,
        height=2,
        substeps=1,
    )

    observation = stream.step(np.ones((3, 1, 2), dtype=np.float32))
    action_request = record_batch_to_actions(
        flight_client.writer.batches[1], expected_sim_id="sim-1"
    )

    assert action_request.start_step_index == 1
    assert action_request.actions.shape == (3, 1, 2)
    assert action_request.actions.dtype == np.float32
    assert observation.step_indices.tolist() == [1, 2, 3]
    assert observation.camera.shape[0] == 3
    assert stream.next_step_index == 4


def test_stream_request_batches_use_one_arrow_schema() -> None:
    from c5r_sim_inference.client import C5RSimClient
    from c5r_sim_inference.protocol import record_batch_to_actions, record_batch_to_new_sim

    flight_client = FakeFlightClient(
        [
            _observation_batch(sim_id="sim-1", start=0, rows=1),
            _observation_batch(sim_id="sim-1", start=1, rows=2),
        ]
    )
    stream = C5RSimClient("grpc://unused", flight_client=flight_client).start_sim(
        task_id="task_11",
        initial_state=np.zeros((1, 2), dtype=np.float32),
        views=("overhead",),
        width=3,
        height=2,
        substeps=1,
    )

    stream.step(np.ones((2, 1, 2), dtype=np.float32))

    assert flight_client.writer.batches[0].schema == flight_client.writer.batches[1].schema
    assert record_batch_to_new_sim(flight_client.writer.batches[0]).task_id == "task_11"
    assert record_batch_to_actions(
        flight_client.writer.batches[1], expected_sim_id="sim-1"
    ).actions.shape == (2, 1, 2)


def test_step_rejects_mismatched_response_row_count() -> None:
    from c5r_sim_inference.client import C5RSimClient

    flight_client = FakeFlightClient(
        [
            _observation_batch(sim_id="sim-1", start=0, rows=1),
            _observation_batch(sim_id="sim-1", start=1, rows=1),
        ]
    )
    stream = C5RSimClient("grpc://unused", flight_client=flight_client).start_sim(
        task_id="task_11",
        initial_state=np.zeros((1, 2), dtype=np.float32),
        views=("overhead",),
        width=3,
        height=2,
        substeps=1,
    )

    with pytest.raises(ValueError, match="observation row count"):
        stream.step(np.ones((2, 1, 2), dtype=np.float32))


def test_start_sim_requires_numpy_initial_state() -> None:
    from c5r_sim_inference.client import C5RSimClient

    flight_client = FakeFlightClient([_observation_batch(sim_id="sim-1", start=0, rows=1)])
    client = C5RSimClient("grpc://unused", flight_client=flight_client)

    with pytest.raises(TypeError, match="initial_state must be a numpy.ndarray"):
        client.start_sim(
            task_id="task_11",
            initial_state=[0.0, 0.0],
            views=("overhead",),
            width=3,
            height=2,
            substeps=1,
        )


def test_step_requires_numpy_actions() -> None:
    from c5r_sim_inference.client import C5RSimClient

    flight_client = FakeFlightClient(
        [
            _observation_batch(sim_id="sim-1", start=0, rows=1),
            _observation_batch(sim_id="sim-1", start=1, rows=1),
        ]
    )
    stream = C5RSimClient("grpc://unused", flight_client=flight_client).start_sim(
        task_id="task_11",
        initial_state=np.zeros((1, 2), dtype=np.float32),
        views=("overhead",),
        width=3,
        height=2,
        substeps=1,
    )

    with pytest.raises(TypeError, match="actions must be a numpy.ndarray"):
        stream.step([[1.0, 1.0]])


def test_step_rejects_mismatched_response_sim_id() -> None:
    from c5r_sim_inference.client import C5RSimClient

    flight_client = FakeFlightClient(
        [
            _observation_batch(sim_id="sim-1", start=0, rows=1),
            _observation_batch(sim_id="other", start=1, rows=1),
        ]
    )
    stream = C5RSimClient("grpc://unused", flight_client=flight_client).start_sim(
        task_id="task_11",
        initial_state=np.zeros((1, 2), dtype=np.float32),
        views=("overhead",),
        width=3,
        height=2,
        substeps=1,
    )

    with pytest.raises(ValueError, match="observation sim_id"):
        stream.step(np.ones((1, 1, 2), dtype=np.float32))


def test_context_manager_closes_writer_reader_and_client() -> None:
    from c5r_sim_inference.client import C5RSimClient

    flight_client = FakeFlightClient([_observation_batch(sim_id="sim-1", start=0, rows=1)])
    client = C5RSimClient("grpc://unused", flight_client=flight_client)

    with client.start_sim(
        task_id="task_11",
        initial_state=np.zeros((1, 2), dtype=np.float32),
        views=("overhead",),
        width=3,
        height=2,
        substeps=1,
    ):
        pass

    assert flight_client.writer.done
    assert flight_client.writer.closed
    assert flight_client.reader.closed


def test_package_exports_public_api() -> None:
    import c5r_sim_inference as sdk

    assert sdk.C5RSimClient is not None
    assert sdk.SimulationStream is not None
    assert sdk.Observation is not None
