# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "c5r-sim-inference",
#     "imageio[ffmpeg]",
#     "numpy",
#     "pillow",
#     "tqdm",
#     "typer",
# ]
#
# [tool.uv.sources]
# c5r-sim-inference = { git = "https://github.com/dream3d-ai/sim-inference.git" }
# ///

"""Run a C5R Flight simulation and write a tiled observation video."""

from __future__ import annotations

from collections.abc import Iterator
from enum import Enum
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import typer
from c5r_sim_inference import C5RSimClient
from c5r_sim_inference.types import Observation
from PIL import Image, ImageDraw
from tqdm import tqdm

DEFAULT_SERVER = "grpc://127.0.0.1:8815"
DEFAULT_TASK_ID = "task_11"
DEFAULT_VIEWS = ("overhead", "side", "wrist_left", "wrist_right")
TITLE_HEIGHT = 32

app = typer.Typer(
    help=(
        "Connect to a running C5R Flight simulation server and write a grid "
        "video from streamed observations."
    )
)


class ActionMode(str, Enum):
    """Action generation modes supported by the demo CLI."""

    zeros = "zeros"
    random = "random"


def iter_action_batches(
    *,
    frame_count: int,
    batch_size: int,
    env_count: int,
    action_dim: int,
    action_mode: ActionMode,
    action_scale: float,
    seed: int,
) -> Iterator[np.ndarray]:
    """Yield batched action arrays for each simulated frame after the initial state."""

    if frame_count < 1:
        raise ValueError("frame_count must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if env_count < 1:
        raise ValueError("env_count must be >= 1")
    if action_dim < 1:
        raise ValueError("action_dim must be >= 1")
    if action_scale < 0:
        raise ValueError("action_scale must be >= 0")

    remaining = frame_count - 1
    rng = np.random.default_rng(seed)
    while remaining > 0:
        rows = min(batch_size, remaining)
        if action_mode == "zeros":
            actions = np.zeros((rows, env_count, action_dim), dtype=np.float32)
        elif action_mode == "random":
            actions = rng.uniform(
                low=-action_scale,
                high=action_scale,
                size=(rows, env_count, action_dim),
            ).astype(np.float32)
        else:
            raise ValueError(f"unsupported action_mode: {action_mode!r}")
        yield actions
        remaining -= rows


def observation_grid_frame(
    observation: Observation,
    *,
    row: int,
    title_prefix: str,
) -> np.ndarray:
    """Render one row of a streamed observation as a titled view grid frame."""

    if observation.camera.ndim != 6:
        raise ValueError(
            "observation camera must have shape (step, env, views, height, width, channels)"
        )
    if not 0 <= row < observation.camera.shape[0]:
        raise IndexError(f"observation row {row} is out of bounds")
    if len(observation.view_names) != observation.camera.shape[2]:
        raise ValueError("observation view_names do not match camera view dimension")

    camera = observation.camera[row, 0]
    images = {
        view: np.ascontiguousarray(camera[view_index])
        for view_index, view in enumerate(observation.view_names)
    }
    step_index = int(observation.step_indices[row].item())
    title = f"{title_prefix} sim={observation.sim_id[:8]} step={step_index}"
    return draw_grid_views(images, title=title, views=observation.view_names)


def draw_grid_views(
    images: dict[str, np.ndarray],
    *,
    title: str,
    views: tuple[str, ...],
) -> np.ndarray:
    """Compose named RGB view images into a titled two-column grid."""

    if not views:
        raise ValueError("views must not be empty")
    missing = [view for view in views if view not in images]
    if missing:
        raise ValueError(f"missing images for views: {missing}")

    first = np.asarray(images[views[0]])
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError(f"{views[0]} image must have shape (height, width, 3)")
    if first.dtype != np.uint8:
        raise ValueError(f"{views[0]} image must have dtype uint8")
    first = np.ascontiguousarray(first)
    height, width = first.shape[:2]
    rows = max(2, (len(views) + 1) // 2)
    out = Image.new("RGB", (width * 2, TITLE_HEIGHT + height * rows), (0, 0, 0))
    draw = ImageDraw.Draw(out)
    draw.text((8, 8), title, fill=(255, 255, 255))

    for index, view in enumerate(views):
        image = np.asarray(images[view])
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{view} image must have shape (height, width, 3)")
        if image.dtype != np.uint8:
            raise ValueError(f"{view} image must have dtype uint8")
        image = np.ascontiguousarray(image)
        if image.shape[:2] != (height, width):
            raise ValueError("all view images must have the same height and width")
        x = (index % 2) * width
        y = TITLE_HEIGHT + (index // 2) * height
        out.paste(Image.fromarray(image), (x, y))
        draw.rectangle((x, y, x + 138, y + 22), fill=(0, 0, 0))
        draw.text((x + 6, y + 5), view, fill=(255, 255, 255))

    return np.asarray(out, dtype=np.uint8)


@app.command()
def main(
    server: str = typer.Option(DEFAULT_SERVER, help="Flight server URI."),
    task_id: str = typer.Option(DEFAULT_TASK_ID, help="Task id to start on the server."),
    output: Path = typer.Option(Path("sim.mp4"), help="Path for the output MP4 video."),
    views: str = typer.Option(
        ",".join(DEFAULT_VIEWS),
        help="Comma-separated camera views to render.",
    ),
    frames: int = typer.Option(
        64,
        min=1,
        help="Total number of video frames to write, including the initial observation.",
    ),
    batch_size: int = typer.Option(
        8,
        min=1,
        help="Number of action steps to request per Flight round trip.",
    ),
    state_dim: int = typer.Option(
        14,
        min=1,
        help="Length of the initial state vector.",
    ),
    action_dim: int = typer.Option(
        14,
        min=1,
        help="Width of each action vector.",
    ),
    width: int = typer.Option(
        320,
        min=1,
        help="Rendered camera width in pixels.",
    ),
    height: int = typer.Option(
        240,
        min=1,
        help="Rendered camera height in pixels.",
    ),
    substeps: int = typer.Option(
        1,
        min=1,
        help="Simulation substeps to advance for each action.",
    ),
    fps: float = typer.Option(
        10.0,
        min=0.0,
        help="Frames per second for the output video.",
    ),
    initial_state_value: float = typer.Option(
        0.0,
        help="Scalar value used to fill the initial state vector.",
    ),
    action_mode: ActionMode = typer.Option(
        ActionMode.zeros,
        help="Action generation mode for simulated steps.",
    ),
    action_scale: float = typer.Option(
        0.01,
        min=0.0,
        help="Scale applied to random actions.",
    ),
    seed: int = typer.Option(0, help="Random seed used when action mode is random."),
    title_prefix: str = typer.Option(
        "C5R Flight e2e",
        help="Text prefix drawn above each grid frame.",
    ),
) -> None:
    """Stream a C5R simulation from Flight and write the observation grid video."""

    output.parent.mkdir(parents=True, exist_ok=True)

    client = C5RSimClient(server)
    frames_written = 0
    try:
        # Setup Sim
        parsed_views = tuple(view.strip() for view in views.split(",") if view.strip())
        initial_state = np.full(
            (1, state_dim),
            fill_value=float(initial_state_value),
            dtype=np.float32,
        )
        with client.start_sim(
            task_id=task_id,
            initial_state=initial_state,
            views=parsed_views,
            width=width,
            height=height,
            substeps=substeps,
            action_dim=action_dim,
        ) as stream:
            typer.echo(f"started sim {stream.sim_id} from task {task_id!r}")

            with (
                imageio.get_writer(output, fps=fps, macro_block_size=1) as writer,
                tqdm(total=frames, unit="frame") as progress,
            ):
                # Write initial observation to video
                writer.append_data(
                    observation_grid_frame(
                        stream.initial_observation,
                        row=0,
                        title_prefix=title_prefix,
                    )
                )
                frames_written += 1
                progress.update(1)

                # Generate action batches and write to video
                for actions in iter_action_batches(
                    frame_count=frames,
                    batch_size=batch_size,
                    env_count=1,
                    action_dim=action_dim,
                    action_mode=action_mode,
                    action_scale=action_scale,
                    seed=seed,
                ):
                    observation = stream.step(actions)
                    for row in range(observation.camera.shape[0]):
                        writer.append_data(
                            observation_grid_frame(
                                observation,
                                row=row,
                                title_prefix=title_prefix,
                            )
                        )
                        frames_written += 1
                        progress.update(1)
            typer.echo(f"wrote {frames_written} grid frames to {output}")
    finally:
        client.close()


if __name__ == "__main__":
    app()
