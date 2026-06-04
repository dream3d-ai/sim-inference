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
    substeps=1,
    scene_seed=10327,
    physics_randomization=True,
    rack_dynamic=True,
    tube_mode="rack-hole",
    tube_hole_index="random",
) as sim:
    initial = sim.initial_observation
    observations = sim.step(np.zeros((8, 1, 14), dtype=np.float32))
```

Inputs are NumPy arrays. Callers are responsible for converting tensors or other
framework objects into NumPy arrays before calling the SDK.

Observations are returned as NumPy arrays:

- `camera`: `np.uint8`, shape `(step, env, views, height, width, 3)`
- `qpos`, `qvel`, `ctrl`: `np.float32`
- `step_indices`: `np.int64`

`scene_seed` enables randomized rack/tube scene generation. Set
`physics_randomization=True` to sample server-owned rack/tube physics ranges
from the default named profile, or pass a profile name such as
`"low"`, `"default"`, or `"high"`. Those profiles vary masses, contact
parameters, and robot joint/actuator dynamics without changing rack/tube
sizes. The client also accepts the server physics parameters
directly on `start_sim`, including
`tube_radius`, `tube_half_length`, `tube_mass_kg`, rack/tube contact
`friction`/`solref`/`solimp`, table bounds, `rack_yaw_range`,
`tube_starts_in_rack_hole`, `tube_mode`, `tube_hole_index`,
`tube_table_spawn_height_m`, `tube_table_roll_range`, and
`tube_table_pitch_range`.

## Installation

```bash
uv pip install git+https://github.com/dream3d-ai/sim-inference.git
```
