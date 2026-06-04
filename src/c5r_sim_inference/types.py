from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .protocol import ObservationBatch


@dataclass(frozen=True)
class Observation:
    """NumPy view of a simulation observation batch."""

    sim_id: str
    step_indices: np.ndarray
    view_names: tuple[str, ...]
    camera: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray

    @classmethod
    def from_batch(cls, batch: ObservationBatch) -> Observation:
        """Convert a protocol observation into contiguous NumPy arrays."""
        return cls(
            sim_id=batch.sim_id,
            step_indices=_contiguous_array(batch.step_indices),
            view_names=batch.view_names,
            camera=_contiguous_array(batch.camera),
            qpos=_contiguous_array(batch.qpos),
            qvel=_contiguous_array(batch.qvel),
            ctrl=_contiguous_array(batch.ctrl),
        )


def _contiguous_array(values: np.ndarray) -> np.ndarray:
    """Return a C-contiguous NumPy array view or copy."""
    return np.ascontiguousarray(values)
