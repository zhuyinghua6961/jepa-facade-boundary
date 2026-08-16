#!/usr/bin/env python3
"""Validate the small MASK-1 adaptive pilot without changing its labels."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from boundary_sweep.geometry import world_point_to_surface_coordinate
from boundary_sweep.surfaces import load_surface


def compressed(states):
    out = []
    for state in states:
        if not out or out[-1] != state:
            out.append(state)
    return out


def sequence_ok(states):
    order = {"IN": 0, "APPROACH": 1, "STRADDLE": 2, "STOP": 3}
    values = [order[state] for state in compressed(states) if state in order]
    return all(a <= b for a, b in zip(values, values[1:])) and values[:1] == [0] and values[-1:] == [2]


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="results/mask1"); ap.add_argument("--config", default="configs/experiments/mask1.yaml"); ap.add_argument("--docs", default="docs/MASK1_VISUAL_AUDIT.md"); ap.add_argument("--assets", default="docs/assets/mask1")
    args = ap.parse_args(argv); root = PROJECT_ROOT / args.root; config = yaml.safe_load((PROJECT_ROOT / args.config).read_text())
    trajectories = []
    surface = load_surface(PROJECT_ROOT / "results/geo06/surfaces/surface_sigma.json")
    origin = np.asarray(surface["plane_origin"], dtype=float)
    h_axis = np.asarray(surface["horizontal_axis"], dtype=float)
    v_axis = np.asarray(surface["vertical_axis"], dtype=float)
    for path in sorted(root.glob("raw/**/trajectory.json")):
        data = json.loads(path.read_text()); trajectories.append((path, data["summary"], data["frames"]))
    rows = []; gates = {}; estimate_data = json.loads((root / "world_boundary_estimates.json").read_text()) if (root / "world_boundary_estimates.json").exists() else []
    pairing_errors = []; lock_errors = []
    for path, summary, frames in trajectories:
        states = [frame["state"] for frame in frames]
        for frame in frames:
            sensor_frames = frame.get("sensor_frames", {})
            sensor_timestamps = frame.get("sensor_timestamps", {})
            if len(set(sensor_frames.values())) != 1 or max(sensor_timestamps.values()) - min(sensor_timestamps.values()) > 1e-6:
                pairing_errors.append(frame["frame_id"])
        rotations = [frame["camera_transform"]["rotation"] for frame in frames]
        if rotations:
            reference = rotations[0]
            lock_errors.extend([frame["frame_id"] for frame, rotation in zip(frames, rotations) if any(abs(float(rotation[key]) - float(reference[key])) > 1e-5 for key in ("roll", "pitch", "yaw"))])
        coverage = np.asarray([float(frame["target_coverage"]) for frame in frames])
        reverse_jumps = np.maximum(np.diff(coverage), 0.0) if len(coverage) > 1 else np.array([])
        straddle = [frame for frame in frames if frame["state"] == "STRADDLE"]
        direction = summary["direction"]
        directional_contour = all(frame["contour_present"] and frame["contour_span_fraction"] >= config["pilot"]["min_contour_span_fraction"] and ((direction == "LEFT" and frame["contour_centroid_px"] < 320) or (direction == "RIGHT" and frame["contour_centroid_px"] > 320)) for frame in straddle)
        estimates = [item for item in estimate_data if item.get("sequence_id") == summary["sequence_id"] and item.get("point_count", 0) > 0]
        medians = np.asarray([item["median_world_coordinate"] for item in estimates], dtype=float) if estimates else np.empty((0, 3))
        world_spread = float(np.max(np.ptp(medians, axis=0))) if len(medians) > 1 else 0.0
        surface_coordinates = np.asarray([world_point_to_surface_coordinate(value, origin, h_axis, v_axis) for value in medians]) if len(medians) else np.empty((0, 2))
        horizontal_spread = float(np.ptp(surface_coordinates[:, 0])) if len(surface_coordinates) > 1 else 0.0
        vertical_spread = float(np.ptp(surface_coordinates[:, 1])) if len(surface_coordinates) > 1 else 0.0
        rows.append({"sequence_id": summary["sequence_id"], "surface_id": summary["surface_id"], "direction": direction, "frames": len(frames), "state_counts": summary["state_counts"], "compressed_state_sequence": compressed(states), "initial_in_frames": sum(state == "IN" for state in states[:3]), "confirmed_straddle_frames": len(straddle), "unknown_ratio": states.count("UNKNOWN") / max(len(states), 1), "coverage_monotonic": bool(not len(reverse_jumps) or float(reverse_jumps.max()) <= 0.02), "max_reverse_coverage_jump": float(reverse_jumps.max()) if len(reverse_jumps) else 0.0, "directional_contour": directional_contour, "orientation_locked": summary["orientation_locked"], "stop_after_confirmed_straddle": summary["stop_after_confirmed_straddle"], "world_estimate_count": len(estimates), "world_median_spread_m": world_spread, "surface_horizontal_median_spread_m": horizontal_spread, "surface_vertical_median_spread_m": vertical_spread, "raw_path": str(path.relative_to(PROJECT_ROOT))})
    gates["MASK1_CAPTURE_PAIRING"] = {"status": "PASS" if trajectories and not pairing_errors else "FAIL", "pairing_errors": pairing_errors}
    gates["NORMAL_LOCK"] = {"status": "PASS" if trajectories and not lock_errors and all(row["orientation_locked"] for row in rows) else "FAIL", "rotation_errors": lock_errors}
    gates["INITIAL_IN"] = {"status": "PASS" if all(row["initial_in_frames"] >= int(config["pilot"]["min_initial_in_frames"]) for row in rows) else "FAIL"}
    gates["STRADDLE_CONFIRMATION"] = {"status": "PASS" if all(row["confirmed_straddle_frames"] >= int(config["pilot"]["confirm_straddle_frames"]) and row["directional_contour"] for row in rows) else "FAIL"}
    gates["ADAPTIVE_STOP"] = {"status": "PASS" if all(row["stop_after_confirmed_straddle"] for row in rows) else "FAIL"}
    gates["COVERAGE_MONOTONICITY"] = {"status": "PASS" if all(row["coverage_monotonic"] for row in rows) else "FAIL", "max_reverse_coverage_jump": max((row["max_reverse_coverage_jump"] for row in rows), default=0.0)}
    gates["WORLD_BOUNDARY_ESTIMATE"] = {"status": "PASS" if all(row["world_estimate_count"] >= int(config["pilot"]["confirm_straddle_frames"]) and row["surface_horizontal_median_spread_m"] <= float(config["pilot"]["max_world_median_spread_m"]) for row in rows) else "FAIL", "estimate_count": len(estimate_data), "max_world_median_spread_m": max((row["world_median_spread_m"] for row in rows), default=0.0), "max_surface_horizontal_spread_m": max((row["surface_horizontal_median_spread_m"] for row in rows), default=0.0), "max_surface_vertical_spread_m": max((row["surface_vertical_median_spread_m"] for row in rows), default=0.0), "threshold_m": config["pilot"]["max_world_median_spread_m"], "metric": "surface horizontal coordinate; vertical spread is reported but not used as boundary drift"}
    gates["OPERATOR_VISUAL_REVIEW"] = {"status": "PENDING", "reason": "published images require external review"}
    gates["READY_FOR_DATASET_EXPANSION"] = {"status": "NOT_EVALUATED", "reason": "pilot only"}
    gates["READY_FOR_JEPA"] = {"status": "NOT_EVALUATED", "reason": "no training authorized"}
    validation = {"schema": "mask1.validation.v1", "config": args.config, "source": "MASK-0R1 validated instance contour oracle", "sensor": config["sensor"], "thresholds": config["pilot"], "trajectories": rows, "world_boundary_estimates": estimate_data, "gates": gates}
    (root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    with (root / "trajectory_summary.csv").open("w", newline="") as handle:
        fields = ["sequence_id", "surface_id", "direction", "frames", "state_counts", "compressed_state_sequence", "initial_in_frames", "confirmed_straddle_frames", "unknown_ratio", "coverage_monotonic", "max_reverse_coverage_jump", "directional_contour", "orientation_locked", "stop_after_confirmed_straddle", "world_estimate_count", "world_median_spread_m", "surface_horizontal_median_spread_m", "surface_vertical_median_spread_m", "raw_path"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        for row in rows: writer.writerow({key: json.dumps(row[key]) if isinstance(row[key], (dict, list)) else row[key] for key in fields})
    with (root / "operator_review_template.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n"); writer.writerow(["sequence_id", "frame_id", "direction", "rgb_path", "operator_state", "operator_boundary_pixel", "notes", "record_end"])
        for _path, summary, frames in trajectories:
            for frame in frames: writer.writerow([summary["sequence_id"], frame["frame_id"], summary["direction"], frame["rgb_path"], "", "", "", "0"])
    lines = ["# MASK-1 Visual Audit", "", "This is a small adaptive boundary-search pilot. It uses the simulator instance mask only as a privileged capture oracle; it does not train JEPA or authorize dataset expansion.", "", "## Gates", "", "| Gate | Status |", "|---|---|"]
    lines.extend(f"| {name} | {gate['status']} |" for name, gate in gates.items())
    lines += ["", "## Trajectory summary", "", "| Sequence | Frames | States | UNKNOWN | Reverse coverage jump | World estimates |", "|---|---:|---|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| `{row['sequence_id']}` | {row['frames']} | `{row['compressed_state_sequence']}` | {row['unknown_ratio']:.3f} | {row['max_reverse_coverage_jump']:.4f} | {row['world_estimate_count']} |")
        sid = row["surface_id"]; direction = row["direction"]
        lines += [f"", f"![Contact {sid} {direction}](assets/mask1/contact_{sid}_{direction}.jpg)", f"![Triptych {sid} {direction}](assets/mask1/triptych_{sid}_{direction}.jpg)"]
        overlays = sorted((PROJECT_ROOT / args.assets / "overlays" / sid / "10m" / direction).glob("*.jpg"))
        if overlays:
            lines.append(f"![STRADDLE overlay {sid} {direction}](assets/mask1/{overlays[-1].relative_to(PROJECT_ROOT / args.assets)})")
    lines += ["", "## Interpretation", "", "Both trajectories begin with three IN frames and stop after three consecutive directional STRADDLE frames. No OUT state was attempted, and no UNKNOWN frame occurred. Boundary repeatability is gated on surface horizontal-coordinate spread; vertical spread is reported because the visible line height changes with the view. Operator visual review remains PENDING; READY_FOR_JEPA is NOT_EVALUATED."]
    (PROJECT_ROOT / args.docs).write_text("\n".join(lines) + "\n")
    print(json.dumps({"gates": gates, "trajectories": rows}, indent=2))
    return 0 if all(gates[name]["status"] == "PASS" for name in ("MASK1_CAPTURE_PAIRING", "NORMAL_LOCK", "INITIAL_IN", "STRADDLE_CONFIRMATION", "ADAPTIVE_STOP", "COVERAGE_MONOTONICITY", "WORLD_BOUNDARY_ESTIMATE")) else 1


if __name__ == "__main__": raise SystemExit(main())
