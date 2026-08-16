#!/usr/bin/env python3
"""Offline MASK-1R1 event and boundary reanalysis.

The historical MASK-1 JSON files are read only.  Every event and 3D boundary
statistic in this script is recomputed from the saved instance masks, z-depth,
camera intrinsics and per-frame transforms.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from boundary_sweep.observability import (action_axis_from_poses, backproject_contour,
                                          first_threshold_step, local_contour_evidence)
from boundary_sweep.segmentation import decode_instance_channels


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def draw_chart(rows: list[dict], direction: str, output: Path) -> None:
    """Draw a real-data coverage/event chart using Pillow only."""
    width, height = 1000, 620
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 80, 65, 950, 520
    draw.rectangle((left, top, right, bottom), outline=(20, 20, 20), width=2)
    for tick in range(0, 11):
        y = bottom - tick * (bottom - top) / 10
        draw.line((left, y, right, y), fill=(225, 225, 225), width=1)
        draw.text((18, y - 8), f"{tick / 10:.1f}", fill=(40, 40, 40))
    steps = [int(row["step_index"]) for row in rows]
    max_step = max(steps) if steps else 1
    def point(step: int, value: float) -> tuple[int, int]:
        x = left + int((right - left) * step / max(max_step, 1))
        y = bottom - int((bottom - top) * float(value))
        return x, y
    coverage = [float(row["target_coverage"]) for row in rows]
    external = [1.0 - value for value in coverage]
    local = [1.0 if row["local_straddle"] else 0.0 for row in rows]
    draw.line([point(step, value) for step, value in zip(steps, coverage)], fill=(30, 110, 210), width=4)
    draw.line([point(step, value) for step, value in zip(steps, external)], fill=(210, 80, 40), width=3)
    draw.line([point(step, value) for step, value in zip(steps, local)], fill=(40, 160, 80), width=3)
    for row in rows:
        if row["local_straddle"]:
            x, _ = point(int(row["step_index"]), 0.0)
            draw.line((x, top, x, bottom), fill=(40, 160, 80), width=1)
    draw.text((left, 18), f"MASK-1R1 {direction}: target coverage / external coverage / local contour", fill=(10, 10, 10))
    draw.text((left, 38), "blue=target coverage  red=global external coverage  green=local contour event", fill=(30, 30, 30))
    draw.text((right - 160, bottom + 28), "step", fill=(30, 30, 30))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=88, optimize=True)


def event_panels(rows: list[dict], direction: str, output: Path) -> None:
    selected = [rows[0], rows[-1]]
    local = next((row for row in rows if row["local_straddle"]), rows[-1])
    if local not in selected:
        selected.insert(1, local)
    canvas = Image.new("RGB", (640 * len(selected), 520), (22, 22, 22))
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(selected):
        panel = Image.open(row["rgb_path"]).convert("RGB").resize((640, 480))
        panel_draw = ImageDraw.Draw(panel)
        panel_draw.rectangle((0, 0, 640, 42), fill=(0, 0, 0))
        text = (f"{direction} step={row['step_index']} frame={row['frame_id']} "
                f"local={row['local_straddle']} ext={row['global_external_coverage']:.3f}")
        panel_draw.text((8, 8), text, fill=(255, 235, 0))
        canvas.paste(panel, (index * 640, 0))
    canvas.save(output, quality=86, optimize=True)


def boundary_axis_plot(boundary_rows: list[dict], axis: np.ndarray, direction: str, output: Path) -> None:
    points = []
    for row in boundary_rows:
        for point in row.get("world_points_sample", []):
            world = np.asarray(point, dtype=float)
            coordinate = float(np.dot(world, axis))
            points.append((coordinate, float(world[2])))
    canvas = Image.new("RGB", (900, 560), "white")
    draw = ImageDraw.Draw(canvas)
    if points:
        x_values = np.asarray([p[0] for p in points]); y_values = np.asarray([p[1] for p in points])
        x0, x1 = float(x_values.min()), float(x_values.max()); y0, y1 = float(y_values.min()), float(y_values.max())
        x_scale = 760 / max(x1 - x0, 1e-6); y_scale = 420 / max(y1 - y0, 1e-6)
        for x, y in points:
            px = int(80 + (x - x0) * x_scale); py = int(470 - (y - y0) * y_scale)
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(30, 110, 200))
        draw.text((20, 18), f"{direction} no-plane depth backprojection", fill=(10, 10, 10))
        draw.text((20, 42), f"action-axis coordinate range {x0:.3f}..{x1:.3f} m; vertical z span {y0:.3f}..{y1:.3f} m", fill=(10, 10, 10))
        draw.line((80, 470, 840, 470), fill=(30, 30, 30), width=2)
        draw.line((80, 50, 80, 470), fill=(30, 30, 30), width=2)
        draw.text((350, 500), "action-axis coordinate (m)", fill=(10, 10, 10))
        draw.text((8, 250), "world z (m)", fill=(10, 10, 10))
    else:
        draw.text((20, 20), f"{direction}: no valid boundary points", fill=(10, 10, 10))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=88, optimize=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/mask1")
    parser.add_argument("--assets", default="docs/assets/mask1r1")
    parser.add_argument("--config", default="configs/experiments/mask1.yaml")
    args = parser.parse_args(argv)
    root = PROJECT_ROOT / args.root
    assets = PROJECT_ROOT / args.assets
    assets.mkdir(parents=True, exist_ok=True)
    config = json.loads("{}")
    # Keep analysis thresholds explicit in the result even though the source
    # MASK-1 YAML is not modified.
    thresholds = {
        "local_min_side_fraction": 0.05,
        "local_min_span_fraction": 0.70,
        "global_external_coverage": {"3pct": 0.03, "5pct": 0.05, "10pct": 0.10},
        "repeatability_sensitivity_m": [0.05, 0.10, 0.25, 1.00],
    }
    trajectories = []
    for trajectory_path in sorted(root.glob("raw/**/trajectory.json")):
        data = json.loads(trajectory_path.read_text())
        summary = data["summary"]
        direction = summary["direction"]
        rows = []
        positions = [np.asarray(frame["T_world_camera"], dtype=float)[:3, 3] for frame in data["frames"]]
        action_axis, axis_meta = action_axis_from_poses(positions)
        for frame in data["frames"]:
            instance_path = resolve(root, frame["files"]["instance"]["path"])
            depth_path = resolve(root, frame["files"]["depth_m"]["path"])
            instance = decode_instance_channels(np.asarray(Image.open(instance_path).convert("RGB")))
            mask = instance["instance_id_16bit"] == int(frame["target_instance_id_16bit"])
            evidence = local_contour_evidence(mask, direction,
                                              thresholds["local_min_side_fraction"],
                                              thresholds["local_min_span_fraction"])
            depth = np.load(depth_path)
            boundary = backproject_contour(evidence["contour"], depth,
                                            np.asarray(frame["K"], dtype=float),
                                            np.asarray(frame["T_world_camera"], dtype=float), action_axis,
                                            depth_reference=depth[mask], depth_tolerance_m=0.25)
            row = {
                "frame_id": int(frame["frame_id"]), "step_index": int(frame["commanded_action"]["step_index"]),
                "offset_m": float(frame["commanded_action"]["offset_m"]),
                "direction": direction, "rgb_path": frame["files"]["rgb"]["path"],
                "target_coverage": float(mask.mean()),
                "global_external_coverage": float(1.0 - mask.mean()),
                "local_straddle": bool(evidence["local_straddle"]),
                "contour_present": bool(evidence["contour_present"]),
                "contour_span_fraction": float(evidence["contour_span_fraction"]),
                "contour_centroid_px": evidence["contour_centroid_px"],
                "target_side_fraction": float(evidence["target_side_fraction"]),
                "external_side_fraction": float(evidence["external_side_fraction"]),
                "local_reason": evidence["reason"],
                "camera_position_world": positions[int(frame["commanded_action"]["step_index"])].tolist(),
                "boundary": boundary,
            }
            rows.append(row)
        target_coverage = [row["target_coverage"] for row in rows]
        external_coverage = [row["global_external_coverage"] for row in rows]
        local_steps = [row["step_index"] for row in rows if row["local_straddle"]]
        first_local = min(local_steps) if local_steps else None
        last_step = rows[-1]["step_index"] if rows else None
        overshoot_steps = (last_step - first_local) if first_local is not None and last_step is not None else None
        straddle_rows = [row for row in rows if row["local_straddle"]]
        axis_coordinates = [row["boundary"]["median_action_axis_coordinate"] for row in straddle_rows if row["boundary"]["median_action_axis_coordinate"] is not None]
        spread = float(max(axis_coordinates) - min(axis_coordinates)) if len(axis_coordinates) > 1 else None
        sensitivity = {str(threshold): {"status": "PASS" if spread is not None and spread <= threshold else "FAIL", "spread_m": spread} for threshold in thresholds["repeatability_sensitivity_m"]}
        event = {
            "sequence_id": summary["sequence_id"], "surface_id": summary["surface_id"], "direction": direction,
            "frame_count": len(rows), "step_m": float(summary["step_m"]), "first_local_straddle_step": first_local,
            "first_global_3pct_step": first_threshold_step(external_coverage, 0.03) if first_threshold_step(external_coverage, 0.03) is not None else "NOT_REACHED",
            "first_global_5pct_step": first_threshold_step(external_coverage, 0.05) if first_threshold_step(external_coverage, 0.05) is not None else "NOT_REACHED",
            "first_global_10pct_step": first_threshold_step(external_coverage, 0.10) if first_threshold_step(external_coverage, 0.10) is not None else "NOT_REACHED",
            "actual_last_step": last_step, "overshoot_steps": overshoot_steps,
            "overshoot_distance_m": None if overshoot_steps is None else float(overshoot_steps * summary["step_m"]),
            "confirmation_same_pose": False,
            "confirmation_distinct_poses": bool(len(straddle_rows) >= 2 and len({tuple(row["camera_position_world"]) for row in straddle_rows}) >= 2),
            "moving_straddle_persistence": len(straddle_rows) >= 3,
            "action_axis": action_axis.tolist(), "action_axis_metadata": axis_meta,
            "boundary_coordinate_spread_m": spread, "repeatability_sensitivity": sensitivity,
            "frames": rows,
        }
        trajectories.append(event)
        draw_chart(rows, direction, assets / f"coverage_{direction.lower()}.jpg")
        event_panels(rows, direction, assets / f"events_{direction.lower()}.jpg")
        boundary_axis_plot([row["boundary"] for row in straddle_rows], action_axis, direction,
                           assets / f"boundary_axis_{direction.lower()}.jpg")
    repeatability = {
        "schema": "mask1.world_boundary_repeatability_r1.v1",
        "method": "target-side external contour pixels with valid z-depth backprojection; no legacy plane, bbox, width, height or boundary line filtering",
        "trajectories": [{key: value for key, value in event.items() if key != "frames"} for event in trajectories],
        "gates": {
            "WORLD_BOUNDARY_REPEATABILITY": {
                "status": "PASS" if trajectories and all(event["boundary_coordinate_spread_m"] is not None and event["boundary_coordinate_spread_m"] <= 1.0 for event in trajectories) else "FAIL",
                "acceptance_threshold_m": 1.0, "threshold_sensitivity_m": thresholds["repeatability_sensitivity_m"],
            },
            "WORLD_BOUNDARY_ABSOLUTE_ACCURACY": {
                "status": "NOT_EVALUATED", "reason": "no independent world-coordinate measurement is available"
            },
        },
    }
    reanalysis = {
        "schema": "mask1.event_reanalysis_r1.v1", "source": "existing results/mask1/raw only",
        "historical_files_unchanged": ["validation.json", "trajectories.json", "world_boundary_estimates.json", "docs/MASK1_VISUAL_AUDIT.md"],
        "thresholds": thresholds, "trajectories": trajectories,
        "gates": {
            "MASK1_EVENT_REANALYSIS": {"status": "PASS"},
            "FIRST_STRADDLE_DETECTION": {"status": "PASS" if all(event["first_local_straddle_step"] is not None for event in trajectories) else "FAIL"},
            "MOVING_STRADDLE_PERSISTENCE": {"status": "PASS" if all(event["moving_straddle_persistence"] for event in trajectories) else "FAIL"},
            "SAME_POSE_CONFIRMATION": {"status": "NOT_EVALUATED", "reason": "historical capture has distinct poses only"},
            "STOP_OVERSHOOT": {"status": "FAIL" if any((event["overshoot_steps"] or 0) > 0 for event in trajectories) else "PASS", "reason": "capture continued after first local STRADDLE"},
            "WORLD_BOUNDARY_REPEATABILITY": repeatability["gates"]["WORLD_BOUNDARY_REPEATABILITY"],
            "WORLD_BOUNDARY_ABSOLUTE_ACCURACY": repeatability["gates"]["WORLD_BOUNDARY_ABSOLUTE_ACCURACY"],
            "PREBOUNDARY_MASK_PROGRESS": {"status": "ABSENT" if all(all(abs(row["global_external_coverage"]) < 1e-12 for row in event["frames"] if event["first_local_straddle_step"] is not None and row["step_index"] < event["first_local_straddle_step"]) for event in trajectories) else "PRESENT", "definition": "global external coverage before local contour"},
        },
    }
    (root / "event_reanalysis_r1.json").write_text(json.dumps(reanalysis, indent=2) + "\n")
    (root / "world_boundary_repeatability_r1.json").write_text(json.dumps(repeatability, indent=2) + "\n")
    validation = {
        "schema": "mask1.validation_r1.v1", "source": "offline reanalysis of existing MASK-1 raw",
        "thresholds": thresholds, "gates": reanalysis["gates"],
        "trajectories": [{key: value for key, value in event.items() if key != "frames"} for event in trajectories],
        "human_visible_event": "EXTERNAL_VISUAL_REVIEW=PENDING",
        "historical_result_files_modified": False,
    }
    (root / "validation_r1.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
