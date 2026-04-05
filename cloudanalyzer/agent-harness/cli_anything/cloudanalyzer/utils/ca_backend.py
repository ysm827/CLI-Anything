"""CloudAnalyzer backend — direct Python import (no subprocess needed).

CloudAnalyzer is a Python package so we import and call its functions
directly rather than shelling out to a binary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def is_available() -> bool:
    """Check if CloudAnalyzer is importable."""
    try:
        import ca  # noqa: F401
        return True
    except ImportError:
        return False


def get_version() -> str:
    """Return the installed CloudAnalyzer version."""
    try:
        from importlib.metadata import version
        return version("cloudanalyzer")
    except Exception:
        return "unknown"


def evaluate(source: str, reference: str, **kwargs: Any) -> dict:
    """Run point cloud evaluation."""
    from ca.evaluate import evaluate as _evaluate
    return _evaluate(source, reference, **kwargs)


def compare(source: str, target: str, method: str = "gicp", **kwargs: Any) -> dict:
    """Run point cloud comparison with optional registration."""
    from ca.compare import run_compare
    return run_compare(source, target, method=method, **kwargs)


def diff(source: str, target: str, threshold: float | None = None) -> dict:
    """Run quick distance diff."""
    from ca.diff import run_diff
    return run_diff(source, target, threshold=threshold)


def evaluate_trajectory(estimated: str, reference: str, **kwargs: Any) -> dict:
    """Evaluate a trajectory against reference."""
    from ca.trajectory import evaluate_trajectory as _eval_traj
    return _eval_traj(estimated, reference, **kwargs)


def evaluate_ground(
    est_ground: str, est_nonground: str,
    ref_ground: str, ref_nonground: str, **kwargs: Any,
) -> dict:
    """Evaluate ground segmentation quality."""
    from ca.ground_evaluate import evaluate_ground_segmentation
    return evaluate_ground_segmentation(
        est_ground, est_nonground, ref_ground, ref_nonground, **kwargs,
    )


def run_check_suite(config_path: str) -> dict:
    """Run config-driven QA checks."""
    from ca.core import load_check_suite, run_check_suite as _run
    suite = load_check_suite(config_path)
    return _run(suite)


def render_check_scaffold(profile: str = "integrated") -> str:
    """Generate a starter config YAML."""
    from ca.core import render_check_scaffold as _render
    result = _render(profile=profile)
    return result.yaml_text


def baseline_decision(candidate_path: str, history_paths: list[str]) -> dict:
    """Decide promote / keep / reject for a baseline."""
    from ca.core import summarize_baseline_evolution
    candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    history = [
        json.loads(Path(p).read_text(encoding="utf-8")) for p in history_paths
    ]
    return summarize_baseline_evolution(candidate, history)


def baseline_save(summary_path: str, history_dir: str, **kwargs: Any) -> str:
    """Save a QA summary to the history directory."""
    from ca.baseline_history import save_baseline
    return save_baseline(summary_path, history_dir, **kwargs)


def baseline_list(history_dir: str) -> list[dict]:
    """List saved baselines."""
    from ca.baseline_history import list_baselines
    return list_baselines(history_dir)


def baseline_discover(history_dir: str) -> list[str]:
    """Discover history JSON paths."""
    from ca.baseline_history import discover_history
    return discover_history(history_dir)


def baseline_rotate(history_dir: str, keep: int = 10) -> list[str]:
    """Rotate old baselines."""
    from ca.baseline_history import rotate_history
    return rotate_history(history_dir, keep=keep)


def downsample(input_path: str, output_path: str, voxel_size: float) -> dict:
    """Voxel grid downsampling."""
    from ca.downsample import downsample as _ds
    return _ds(input_path, voxel_size, output_path)


def split(input_path: str, output_dir: str, grid_size: float, axis: str = "xy") -> dict:
    """Split a point cloud into grid tiles."""
    from ca.split import split as _split
    return _split(input_path, output_dir, grid_size, axis=axis)


def info(path: str) -> dict:
    """Get point cloud metadata."""
    from ca.info import get_info
    return get_info(path)
