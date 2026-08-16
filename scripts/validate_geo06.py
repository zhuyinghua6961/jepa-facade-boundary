#!/usr/bin/env python3
"""Validate GEO-0.6 event coverage and build compact visual audit assets."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from boundary_sweep.geo06 import trajectory_metrics
from boundary_sweep.labels import render_overlay
from boundary_sweep.surfaces import load_surface


def _rgb_path(project_root, frame):
    path = project_root / frame["rgb_path"]
    if path.exists():
        return path
    return project_root / "results/geo06/raw" / frame["rgb_path"]


def _save_overlay(project_root, frame, output):
    png = output.with_suffix(".png")
    render_overlay(_rgb_path(project_root, frame), frame["labels"], png)
    Image.open(png).convert("RGB").save(output, quality=84, optimize=True)
    png.unlink(missing_ok=True)


def _contact_sheet(project_root, frames, output, title):
    count = min(12, len(frames)); indices = [round(i * (len(frames) - 1) / max(count - 1, 1)) for i in range(count)]
    tiles = []
    for index in indices:
        frame = frames[index]; image = Image.open(_rgb_path(project_root, frame)).convert("RGB").resize((320, 240)); draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 320, 38), fill=(0, 0, 0)); draw.text((5, 5), f"frame={frame['frame_id']} {frame['labels']['label']}", fill=(255, 235, 0)); draw.text((5, 21), f"offset={frame['commanded_action']['offset_m']:.1f} cov={frame['labels']['target_pixel_coverage']:.3f}", fill=(255, 235, 0)); tiles.append(image)
    sheet = Image.new("RGB", (320 * 4, 240 * 3), (20, 20, 20))
    for i, tile in enumerate(tiles): sheet.paste(tile, ((i % 4) * 320, (i // 4) * 240))
    sheet.save(output, quality=84, optimize=True)


def _triptych(project_root, frames, output):
    by_state = {}
    for frame in frames: by_state.setdefault(frame["labels"]["label"], frame)
    chosen = [by_state.get(state, frames[min(i, len(frames) - 1)]) for i, state in enumerate(("IN", "STRADDLE", "OUT"))]
    canvas = Image.new("RGB", (960, 480), (20, 20, 20))
    for i, frame in enumerate(chosen):
        image = Image.open(_rgb_path(project_root, frame)).convert("RGB").resize((320, 480)); draw = ImageDraw.Draw(image); draw.rectangle((0, 0, 320, 34), fill=(0, 0, 0)); draw.text((5, 5), f"{frame['labels']['label']} frame={frame['frame_id']}", fill=(255, 235, 0)); canvas.paste(image, (i * 320, 0))
    canvas.save(output, quality=84, optimize=True)


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default="results/geo06/raw"); parser.add_argument("--results", default="results/geo06"); parser.add_argument("--config", default="configs/experiments/geo06.yaml"); parser.add_argument("--assets", default="docs/assets/geo06"); parser.add_argument("--docs", default="docs/GEO06_VISUAL_AUDIT.md")
    args = parser.parse_args(argv); root = Path(args.root); result_root = Path(args.results); assets = Path(args.assets); assets.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text()); surfaces_manifest = json.loads((result_root / "surface_manifest.json").read_text()); surfaces = {p.stem: load_surface(p) for p in (result_root / "surfaces").glob("*.json")}
    trajectories = []
    for path in sorted(root.glob("**/trajectory.json")):
        data = json.loads(path.read_text()); frames = data.get("frames_data", data.get("frames", [])); trajectories.append((path, data, frames))
    trajectory_rows = []; frame_rows = []; image_rows = []
    for path, data, frames in trajectories:
        metrics = trajectory_metrics(frames); plan = data["plan"]; tid = data["sequence_id"].replace("/", "_")
        metrics.update({"sequence_id": data["sequence_id"], "surface_id": data["surface_id"], "direction": data["direction"], "active_boundary": data["active_boundary"], "plan": plan})
        trajectory_rows.append(metrics)
        for frame in frames:
            label = frame["labels"]; frame_rows.append({"frame_id": frame["frame_id"], "sequence_id": frame["sequence_id"], "surface_id": frame["facade_id"], "direction": frame["commanded_action"]["direction"], "active_boundary": frame["labels"]["active_boundary"], "step_index": frame["commanded_action"]["step_index"], "offset_m": frame["commanded_action"]["offset_m"], "label": label["label"], "target_pixel_coverage": label["target_pixel_coverage"], "occlusion_visibility_ratio": label["occlusion_visibility_ratio"], "boundary_pixel_line": json.dumps(label["boundary"]["boundary_pixel_line"]), "rgb_path": frame["rgb_path"]})
        contact = assets / f"contact_{tid}.jpg"; _contact_sheet(PROJECT_ROOT, frames, contact, tid); image_rows.append((tid, "contact", contact))
        triptych = assets / f"triptych_{tid}.jpg"; _triptych(PROJECT_ROOT, frames, triptych); image_rows.append((tid, "triptych", triptych))
        straddles = [frame for frame in frames if frame["labels"]["label"] == "STRADDLE"]
        straddle = straddles[len(straddles) // 2] if straddles else frames[len(frames) // 2]
        straddle_path = assets / f"straddle_{tid}.jpg"; _save_overlay(PROJECT_ROOT, straddle, straddle_path); image_rows.append((tid, "straddle", straddle_path))
        unknowns = [frame for frame in frames if frame["labels"]["label"] == "UNKNOWN"]
        failure = unknowns[0] if unknowns else frames[-1]
        failure_path = assets / f"failure_{tid}.jpg"; _save_overlay(PROJECT_ROOT, failure, failure_path); image_rows.append((tid, "failure", failure_path))
        overview_path = assets / f"overview_{data['surface_id']}.jpg"
        if not overview_path.exists(): _save_overlay(PROJECT_ROOT, frames[0], overview_path); image_rows.append((data["surface_id"], "overview", overview_path))
    rejected = [row for row in surfaces_manifest.get("rejected_candidates", []) if row.get("rgb_path")]
    if rejected:
        row = rejected[0]; source = PROJECT_ROOT / row["rgb_path"]; rejected_path = assets / "rejected_candidate_example.jpg"; shutil.copyfile(source, rejected_path); image_rows.append(("rejected", "rejected", rejected_path))

    pairing_ok = True; pairing_errors = []
    for frame in frame_rows:
        base = _rgb_path(PROJECT_ROOT, frame); depth = base.with_name(base.name.replace("_rgb.png", "_depth_m.npy")); metadata = base.with_name(base.name.replace("_rgb.png", ".json"))
        if not (base.exists() and depth.exists() and metadata.exists()): pairing_ok = False; pairing_errors.append(frame["rgb_path"])
    selected_items = surfaces_manifest.get("selected_surfaces", [])
    scene_ok = len(selected_items) >= 2 and len(surfaces_manifest.get("rejected_candidates", [])) > 0 and not surfaces_manifest.get("rejected_surfaces")
    physical_ok = len(selected_items) >= 2 and all(item.get("plane_fit_max_residual_m", 999) <= config["validation"]["plane_fit_max_residual_m"] and item.get("depth_alignment_median_abs_error_m", 999) <= 0.1 and set(surfaces[item["surface_id"]]["physical_boundary"]) == {"LEFT", "RIGHT", "TOP", "BOTTOM"} for item in selected_items)
    visibility_ok = all(row["unknown_ratio"] <= config["sweep"]["max_unknown_ratio"] and row["straddle_boundary_inside"] for row in trajectory_rows)
    ordering_ok = all(row["monotonic_ignoring_unknown"] and row["max_reverse_overlap_jump"] <= config["sweep"]["max_reverse_overlap_jump"] for row in trajectory_rows)
    coverage_ok = all(row["event_coverage"] for row in trajectory_rows)
    gates = {
        "SCENE_SUITABILITY": {"status": "PASS" if scene_ok else "FAIL", "selected_surface_count": len(surfaces_manifest.get("selected_surfaces", [])), "rejected_candidate_count": len(surfaces_manifest.get("rejected_candidates", []))},
        "SENSOR_PAIRING": {"status": "PASS" if pairing_ok and len(frame_rows) > 0 else "FAIL", "frame_count": len(frame_rows), "pairing_errors": pairing_errors[:20]},
        "PHYSICAL_BOUNDARY_GT": {"status": "PASS" if physical_ok else "FAIL", "surfaces": selected_items, "rejected_surfaces": surfaces_manifest.get("rejected_surfaces", [])},
        "BOUNDARY_VISIBILITY": {"status": "PASS" if visibility_ok else "FAIL", "max_unknown_ratio": max((row["unknown_ratio"] for row in trajectory_rows), default=1.0), "straddle_boundary_inside": all(row["straddle_boundary_inside"] for row in trajectory_rows)},
        "TRAJECTORY_ORDERING": {"status": "PASS" if ordering_ok else "FAIL", "all_monotonic": ordering_ok, "max_reverse_overlap_jump": max((row["max_reverse_overlap_jump"] for row in trajectory_rows), default=1.0)},
        "EVENT_COVERAGE": {"status": "PASS" if coverage_ok else "FAIL", "complete_trajectories": sum(row["event_coverage"] for row in trajectory_rows), "required_trajectories": len(trajectory_rows)},
        "OPERATOR_VISUAL_REVIEW": {"status": "PENDING", "reason": "external review of published images is required"},
    }
    auto_pass = all(gates[name]["status"] == "PASS" for name in ("SCENE_SUITABILITY", "SENSOR_PAIRING", "PHYSICAL_BOUNDARY_GT", "BOUNDARY_VISIBILITY", "TRAJECTORY_ORDERING", "EVENT_COVERAGE"))
    gates["READY_FOR_DATASET_EXPANSION"] = {"status": "PASS" if auto_pass else "FAIL"}
    gates["READY_FOR_JEPA"] = {"status": "NOT_EVALUATED", "reason": "GEO-0.6 is a data-feasibility audit; no JEPA training is authorized"}
    validation = {"schema": "geo06.validation.v1", "config": args.config, "map": surfaces_manifest.get("map"), "sensor": {"width": 640, "height": 480, "horizontal_fov_deg": 90.0, "fixed_delta_seconds": 0.05, "depth_metric": "z-depth"}, "thresholds": config["sweep"], "gates": gates, "trajectories": trajectory_rows, "image_assets": [str(path) for _, _, path in image_rows]}
    (result_root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    with (result_root / "trajectory_summary.csv").open("w", newline="") as f:
        fields = ["sequence_id", "surface_id", "direction", "active_boundary", "frames", "state_counts", "compressed_state_sequence", "unknown_ratio", "monotonic_ignoring_unknown", "outward_overlap_spearman", "max_reverse_overlap_jump", "event_coverage"]; w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for row in trajectory_rows: w.writerow({key: json.dumps(row[key]) if isinstance(row[key], (dict, list)) else row[key] for key in fields})
    with (result_root / "frame_manifest.csv").open("w", newline="") as f:
        fields = list(frame_rows[0]) if frame_rows else ["frame_id"]; w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(frame_rows)
    with (result_root / "operator_review_template.csv").open("w", newline="") as f:
        fields = ["frame_id", "sequence_id", "surface_id", "direction", "step_index", "rgb_path", "operator_state", "operator_boundary_pixel", "operator_notes"]; w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n"); w.writeheader()
        for row in frame_rows: w.writerow({**{key: row.get(key, "") for key in fields[:6]}, "operator_state": "", "operator_boundary_pixel": "", "operator_notes": ""})
    lines = ["# GEO-0.6 Visual Audit", "", "This is a data-feasibility audit for later modeling. Operator visual review is **PENDING** and `READY_FOR_JEPA` is **NOT_EVALUATED**.", "", "## Overview", ""]
    for surface in sorted({row["surface_id"] for row in trajectory_rows}): lines.append(f"![{surface}](assets/geo06/overview_{surface}.jpg)")
    lines += ["", "## Gate Results", "", "| Gate | Status |", "|---|---|"] + [f"| {name} | {gate['status']} |" for name, gate in gates.items()] + ["", "## Trajectory Evidence", "", "| Sequence | States | UNKNOWN | Spearman | Reverse jump | Events |", "|---|---|---:|---:|---:|---|"]
    for row in trajectory_rows:
        lines.append(f"| `{row['sequence_id']}` | `{row['compressed_state_sequence']}` | {row['unknown_ratio']:.3f} | {row['outward_overlap_spearman']:.3f} | {row['max_reverse_overlap_jump']:.3f} | {row['event_coverage']} |")
        tid = row["sequence_id"].replace("/", "_"); lines.extend(["", f"![Contact {tid}](assets/geo06/contact_{tid}.jpg)", f"![Triptych {tid}](assets/geo06/triptych_{tid}.jpg)", f"![STRADDLE {tid}](assets/geo06/straddle_{tid}.jpg)", f"![Failure or UNKNOWN {tid}](assets/geo06/failure_{tid}.jpg)"])
    lines += ["", "## Representative Frames", "", "| Sequence | Frame ID | Step | Offset (m) | Label | Coverage | Occlusion ratio | Boundary in image | Reason |", "|---|---:|---:|---:|---|---:|---:|---|---|"]
    for _path, data, frames in trajectories:
        chosen = []
        for state in ("IN", "STRADDLE", "OUT", "UNKNOWN"):
            chosen_frame = next((frame for frame in frames if frame["labels"]["label"] == state), None)
            if chosen_frame is not None: chosen.append(chosen_frame)
        for frame in chosen:
            label = frame["labels"]; reason = "depth/occlusion or boundary evidence incomplete" if label["label"] == "UNKNOWN" else "independent geometry label"
            lines.append(f"| `{data['sequence_id']}` | {frame['frame_id']} | {frame['commanded_action']['step_index']} | {frame['commanded_action']['offset_m']:.2f} | {label['label']} | {label['target_pixel_coverage']:.3f} | {label['occlusion_visibility_ratio']:.3f} | {label['boundary'].get('boundary_in_image')} | {reason} |")
    lines += ["", "## Rejected Candidate", "", "![Rejected candidate](assets/geo06/rejected_candidate_example.jpg)", "", "Rejected candidate reasons and raycast statistics are recorded in `results/geo06/surface_manifest.json`. Raw RGB-D remains outside Git tracking.", ""]
    Path(args.docs).write_text("\n".join(lines))
    print(json.dumps({"gates": gates, "trajectory_count": len(trajectory_rows), "frame_count": len(frame_rows), "assets": len(image_rows)}, indent=2))
    return 0 if gates["READY_FOR_DATASET_EXPANSION"]["status"] == "PASS" else 1


if __name__ == "__main__": raise SystemExit(main())
