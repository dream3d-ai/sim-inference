from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .protocol import ObservationBatch


@dataclass(frozen=True)
class TorchObservation:
    sim_id: str
    step_indices: torch.Tensor
    view_names: tuple[str, ...]
    camera: torch.Tensor
    qpos: torch.Tensor
    qvel: torch.Tensor
    ctrl: torch.Tensor

    @classmethod
    def from_batch(cls, batch: ObservationBatch) -> TorchObservation:
        return cls(
            sim_id=batch.sim_id,
            step_indices=_torch_from_numpy(batch.step_indices),
            view_names=batch.view_names,
            camera=_torch_from_numpy(batch.camera),
            qpos=_torch_from_numpy(batch.qpos),
            qvel=_torch_from_numpy(batch.qvel),
            ctrl=_torch_from_numpy(batch.ctrl),
        )


def initial_state_to_numpy(initial_state: torch.Tensor) -> np.ndarray:
    return tensor_to_numpy(
        initial_state, dtype=torch.float32, name="initial_state", rank=1
    )


def actions_to_numpy(actions: torch.Tensor) -> np.ndarray:
    array = tensor_to_numpy(actions, dtype=torch.float32, name="actions", rank=2)
    if array.shape[0] == 0:
        raise ValueError("actions batch must not be empty")
    return array


def tensor_to_numpy(
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    name: str,
    rank: int,
) -> np.ndarray:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != rank:
        raise ValueError(f"{name} must have rank {rank}, got {tensor.ndim}")
    normalized = tensor.detach()
    if normalized.device.type != "cpu" or normalized.dtype != dtype:
        normalized = normalized.to(device="cpu", dtype=dtype)
    if not normalized.is_contiguous():
        normalized = normalized.contiguous()
    array = normalized.numpy()
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _torch_from_numpy(values: np.ndarray) -> torch.Tensor:
    array = np.ascontiguousarray(values)
    if not array.flags.writeable:
        array = array.copy()
    return torch.from_numpy(array)
