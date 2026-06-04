from __future__ import annotations

import importlib.util
from pathlib import Path

from typer.testing import CliRunner


def _load_demo_module():
    path = Path(__file__).parents[1] / "examples" / "demo.py"
    spec = importlib.util.spec_from_file_location("c5r_demo", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_cli_uses_typer_and_exposes_existing_options() -> None:
    demo = _load_demo_module()

    assert not hasattr(demo, "run")
    assert not hasattr(demo, "_validate_image")
    assert demo.__doc__
    assert demo.ActionMode.__doc__
    assert demo.iter_action_batches.__doc__
    assert demo.observation_grid_frame.__doc__
    assert demo.draw_grid_views.__doc__
    assert demo.main.__doc__
    result = CliRunner().invoke(demo.app, ["--help"])

    assert result.exit_code == 0
    assert "--task-id" in result.output
    assert "--batch-size" in result.output
    assert "--action-mode" in result.output
    assert "Flight server URI." in result.output
    assert "Comma-separated camera" in result.output
    assert "views to render." in result.output
    assert "Action generation mode" in result.output
    assert "for simulated steps." in result.output
