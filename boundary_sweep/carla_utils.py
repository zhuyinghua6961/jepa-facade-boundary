"""Small helpers for finding and importing the local CARLA PythonAPI."""

from __future__ import annotations

import glob
import importlib
import os
import sys
from pathlib import Path


def discover_carla_root(explicit: str | None = None) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    for key in ("CARLA_ROOT",):
        if os.environ.get(key):
            candidates.append(Path(os.environ[key]).expanduser())
    here = Path(__file__).resolve()
    candidates.extend([
        Path.cwd() / "Carla-0.10.0-Linux-Shipping",
        Path.cwd() / "CARLA_0.10.0",
        here.parents[2] / "Carla-0.10.0-Linux-Shipping",
    ])
    for candidate in candidates:
        if (candidate / "PythonAPI").exists() or (candidate / "CarlaUE5.sh").exists():
            return candidate
    return None


def import_carla(carla_root: str | None = None):
    try:
        return importlib.import_module("carla")
    except ImportError:
        pass
    root = discover_carla_root(carla_root)
    if root is None:
        raise ImportError("CARLA PythonAPI not found; pass --carla-root")
    paths = [root / "PythonAPI", root / "PythonAPI" / "carla"]
    paths.extend(Path(p) for p in glob.glob(str(root / "PythonAPI" / "carla" / "dist" / "*.egg")))
    for path in paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return importlib.import_module("carla")


def transform_to_dict(transform) -> dict:
    return {
        "location": {k: float(getattr(transform.location, k)) for k in ("x", "y", "z")},
        "rotation": {k: float(getattr(transform.rotation, k)) for k in ("pitch", "yaw", "roll")},
    }


def transform_from_dict(carla, value: dict):
    loc, rot = value["location"], value["rotation"]
    return carla.Transform(
        carla.Location(x=float(loc["x"]), y=float(loc["y"]), z=float(loc["z"])),
        carla.Rotation(pitch=float(rot["pitch"]), yaw=float(rot["yaw"]), roll=float(rot["roll"])),
    )


def vector_dict(vector) -> dict:
    return {k: float(getattr(vector, k)) for k in ("x", "y", "z")}

