# C5R Sim Inference

PyTorch client SDK for the C5R simulation Apache Flight server.

```python
import torch

from c5r_sim_inference import C5RSimClient

client = C5RSimClient("grpc://127.0.0.1:8815")

with client.start_sim(
    task_id="task_11",
    initial_state=torch.zeros(14),
    views=("overhead",),
    width=320,
    height=240,
    substeps=1,
) as sim:
    initial = sim.initial_observation
    observations = sim.step(torch.zeros(8, 14))
```

Inputs are PyTorch tensors. Already-contiguous CPU `float32` tensors use a
zero-copy PyTorch-to-NumPy handoff before Arrow serialization. GPU,
non-contiguous, dtype-changing, or grad-tracked tensors are normalized to one
CPU contiguous `float32` array before sending.

Observation tensors are returned as PyTorch tensors:

- `camera`: `torch.uint8`, shape `(batch, views, height, width, 3)`
- `qpos`, `qvel`, `ctrl`: `torch.float32`
- `step_indices`: `torch.int64`
