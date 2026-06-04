# C5R Sim Inference

NumPy client SDK for the C5R simulation Apache Flight server.

```python
import numpy as np

from c5r_sim_inference import C5RSimClient

client = C5RSimClient("grpc://127.0.0.1:8815")

with client.start_sim(
    task_id="task_11",
    initial_state=np.zeros((1, 14), dtype=np.float32),
    views=("overhead",),
    width=320,
    height=240,
    physics_dt=1.0 / 30.0,
    scene_seed=10327,
    physics_randomization=True,
) as sim:
    initial = sim.initial_observation
    observations = sim.step(np.zeros((1, 8, 14), dtype=np.float32))
```

Inputs are NumPy arrays. Callers are responsible for converting tensors or other
framework objects into NumPy arrays before calling the SDK.

Observations are returned as NumPy arrays:

- `camera`: `np.uint8`, shape `(step, env, views, height, width, 3)`
- `qpos`, `qvel`, `ctrl`: `np.float32`
- `step_indices`: `np.int64`

`SimulationStream.step(...)` accepts actions as `(env, step, action_dim)`.
Returned observations remain step-major: `(step, env, ...)`.

Set `physics_randomization=True` to sample the default server-owned rack/tube
physics profile, or pass `"low"`, `"default"`, or `"high"`. Profiles own the
scene materialization details and vary masses, contact parameters, placement,
and robot joint/actuator dynamics without exposing rack/tube scene knobs to
clients. `scene_seed` is optional and only used to make physics randomization
reproducible.

## Installation

```bash
uv pip install git+https://github.com/dream3d-ai/sim-inference.git
```
