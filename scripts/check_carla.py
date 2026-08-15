#!/usr/bin/env python3
"""Read-only CARLA/Town10 and RGB-D capability check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from boundary_sweep.carla_utils import discover_carla_root, import_carla


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carla-root", default=None)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--map", default="Town10")
    ap.add_argument("--no-load", action="store_true", help="do not call load_world")
    args = ap.parse_args()
    root = discover_carla_root(args.carla_root)
    carla = import_carla(str(root) if root else args.carla_root)
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    server_version = getattr(client, "get_server_version", lambda: "unknown")()
    world = client.get_world()
    if not args.no_load and args.map not in world.get_map().name:
        world = client.load_world(args.map)
    bps = world.get_blueprint_library()
    checks = {name: bool(bps.filter(name)) for name in ("sensor.camera.rgb", "sensor.camera.depth")}
    settings = world.get_settings()
    result = {
        "carla_root": str(root) if root else None,
        "server_version": server_version,
        "client_version": getattr(carla, "__version__", "unknown"),
        "map": world.get_map().name,
        "requested_map": args.map,
        "town10_loaded": args.map in world.get_map().name,
        "synchronous_mode": bool(settings.synchronous_mode),
        "fixed_delta_seconds": settings.fixed_delta_seconds,
        "sensor_blueprints": checks,
    }
    print(json.dumps(result, indent=2, default=str))
    if not result["town10_loaded"] or not all(checks.values()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

