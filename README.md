# C5R Sim Inference

NumPy client SDK for the C5R simulation Apache Flight server.

```python
import numpy as np

from c5r_sim_inference import C5RSimClient

client = C5RSimClient("grpc://127.0.0.1:8815")

with client.start_sim(
    task_id="task_11",
    initial_state=np.zeros(14, dtype=np.float32),
    views=("overhead",),
    width=320,
    height=240,
    substeps=1,
) as sim:
    initial = sim.initial_observation
    observations = sim.step(np.zeros((8, 14), dtype=np.float32))
```

Inputs are NumPy arrays. Callers are responsible for converting tensors or other
framework objects into NumPy arrays before calling the SDK.

Observations are returned as NumPy arrays:

- `camera`: `np.uint8`, shape `(batch, views, height, width, 3)`
- `qpos`, `qvel`, `ctrl`: `np.float32`
- `step_indices`: `np.int64`
