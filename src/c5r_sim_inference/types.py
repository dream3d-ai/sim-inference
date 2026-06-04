from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .protocol import ObservationBatch


@dataclass(frozen=True)
class Observation:
    """PyTorch view of a simulation observation batch."""

    sim_id: str
    step_indices: torch.Tensor
    view_names: tuple[str, ...]
    camera: torch.Tensor
    qpos: torch.Tensor
    qvel: torch.Tensor
    ctrl: torch.Tensor

    @classmethod
    def from_batch(cls, batch: ObservationBatch) -> Observation:
        """Convert a NumPy/Arrow protocol observation into PyTorch tensors."""
        return cls(
            sim_id=batch.sim_id,
            step_indices=_torch_from_numpy(batch.step_indices),
            view_names=batch.view_names,
            camera=_torch_from_numpy(batch.camera),
            qpos=_torch_from_numpy(batch.qpos),
            qvel=_torch_from_numpy(batch.qvel),
            ctrl=_torch_from_numpy(batch.ctrl),
        )


def _torch_from_numpy(values: np.ndarray) -> torch.Tensor:
    """Create a PyTorch tensor from a contiguous writeable NumPy array."""
    array = np.ascontiguousarray(values)
    if not array.flags.writeable:
        array = array.copy()
    return torch.from_numpy(array)
