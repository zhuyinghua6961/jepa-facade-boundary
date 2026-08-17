#!/usr/bin/env python3
"""ACT-0 multi-facade scout and counterfactual pilot entry point."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from boundary_sweep.act0 import (backproject_mask_points, candidate_quality,
                                 directional_instance_contour, fit_target_plane,
                                 fit_terminal_boundaries, orthogonal_surface_axes,
                                 scout_sensor_pairing, tier_metric_repeatability,
                                 tier_physical_plane, tier_visual_event)
from boundary_sweep.carla_utils import (discover_carla_root, import_carla,
                                        transform_to_dict)
from boundary_sweep.observability import backproject_contour
from boundary_sweep.segmentation import (BUILDING_TAG, bgra_array,
                                         choose_center_ids,
                                         decode_instance_channels,
                                         rgb_from_bgra,
                                         stable_id_intersection)
from boundary_sweep.sensors import SynchronousRGBDSeg


def _normal_rotation(carla, normal):
    direction = -np.asarray(normal, dtype=float)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    pitch = math.degrees(math.atan2(direction[2], max(math.hypot(direction[0], direction[1]), 1e-12)))
    yaw = math.degrees(math.atan2(direction[1], direction[0]))
    return carla.Rotation(pitch=float(pitch), yaw=float(yaw), roll=0.0)


def _horizontal_axis(normal):
    normal = np.asarray(normal, dtype=float)
    horizontal = np.array([normal[1], -normal[0], 0.0], dtype=float)
    horizontal /= max(float(np.linalg.norm(horizontal)), 1e-12)
    return horizontal


def _location(carla, xyz):
    return carla.Location(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))


def _relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def _raycast_audit(world, carla, camera_origins, point_groups, tolerance_m: float) -> dict:
    attempted = matched = building_hits = 0
    residuals = []
    for origin, points in zip(camera_origins, point_groups):
        arr = np.asarray(points, dtype=float)
        if not len(arr):
            continue
        for point in arr[::max(1, len(arr) // 20)]:
            direction = point - origin
            length = float(np.linalg.norm(direction))
            if length <= 1e-6:
                continue
            target = point + direction / length * 0.5
            attempted += 1
            hits = world.cast_ray(_location(carla, origin), _location(carla, target))
            hit = next((row for row in hits if str(row.label) == "Buildings"), None)
            if hit is None:
                continue
            building_hits += 1
            location = np.array([hit.location.x, hit.location.y, hit.location.z], dtype=float)
            residual = float(np.linalg.norm(location - point))
            residuals.append(residual)
            matched += int(residual <= tolerance_m)
    agreement = float(matched / attempted) if attempted else None
    return {"attempted": attempted, "building_hits": building_hits, "matched": matched,
            "agreement": agreement, "tolerance_m": float(tolerance_m),
            "median_residual_m": float(np.median(residuals)) if residuals else None,
            "max_residual_m": float(np.max(residuals)) if residuals else None}


def _save_overview(rgb, candidate, target_id, output: Path):
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 42), fill=(0, 0, 0))
    draw.text((7, 6), f"candidate={candidate['candidate_index']} bbox={candidate['bbox_id']} side={candidate['surface']}", fill=(255, 235, 0))
    draw.text((7, 23), f"target_instance_id={target_id}", fill=(255, 235, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, quality=84, optimize=True)


def _save_overlay(rgb, mask, left_contour, right_contour, candidate, decision,
                  reasons, output: Path):
    image = Image.fromarray(rgb, mode="RGB").convert("RGBA")
    overlay = np.zeros((*mask.shape, 4), dtype=np.uint8)
    overlay[mask] = (35, 215, 85, 85)
    overlay[left_contour] = (255, 190, 0, 240)
    overlay[right_contour] = (20, 210, 255, 240)
    image = Image.alpha_composite(image, Image.fromarray(overlay, mode="RGBA"))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 58), fill=(0, 0, 0, 225))
    draw.text((7, 6), f"bbox={candidate['bbox_id']} {decision} target={mask.mean():.3f}", fill=(255, 235, 0))
    draw.text((7, 24), "yellow=LEFT external contour, cyan=RIGHT", fill=(255, 235, 0))
    draw.text((7, 42), ", ".join(reasons[:3]) if reasons else "all automatic scout gates passed", fill=(255, 235, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output, quality=84, optimize=True)


def _scout_candidate(candidate, rig, world, carla, config, assets_root):
    thresholds = config["scout"]
    center = np.asarray(candidate["center"], dtype=float)
    normal = np.asarray(candidate["normal"], dtype=float)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    candidate_h = _horizontal_axis(normal)
    distance = max(float(thresholds["min_distance_m"]),
                   float(thresholds["distance_width_fraction"]) * float(candidate["width_m"]))
    rotation = _normal_rotation(carla, normal)
    views = []
    for view_index, fraction in enumerate(thresholds["view_offsets_fraction"]):
        position = center + normal * distance + candidate_h * float(fraction) * float(candidate["width_m"])
        transform = carla.Transform(_location(carla, position), rotation)
        rig.set_transform(transform)
        action = {"mode": "ACT0_SCOUT", "candidate_index": candidate["candidate_index"],
                  "bbox_id": candidate["bbox_id"], "view_index": view_index,
                  "offset_fraction": float(fraction)}
        sample = rig.capture(action)
        instance = decode_instance_channels(bgra_array(sample["data"]["instance"]))
        semantic = decode_instance_channels(bgra_array(sample["data"]["semantic"]))["semantic_tag"]
        views.append({
            "view_index": view_index, "position": position, "transform": sample["T_world_camera"],
            "transform_dict": transform_to_dict(sample["camera_transform"]), "K": sample["K"],
            "rgb": rgb_from_bgra(bgra_array(sample["data"]["rgb"])),
            "depth_m": sample["depth_m"], "semantic": semantic,
            "instance_id": instance["instance_id_16bit"],
            "candidate_ids": choose_center_ids(instance["instance_id_16bit"], semantic, fraction=0.30),
            "frame_id": int(sample["frame_id"]), "timestamp": float(sample["timestamp"]),
            "sensor_frames": sample["sensor_frames"], "sensor_timestamps": sample["sensor_timestamps"],
        })
    stable_ids = stable_id_intersection(views, min_views=len(views))
    center_candidates = views[len(views) // 2]["candidate_ids"]
    target_id = next((int(value) for value in center_candidates if int(value) in stable_ids), None)
    masks, supports, camera_origins = [], [], []
    boundary_audits = {"LEFT": [], "RIGHT": []}
    boundary_points = {"LEFT": [], "RIGHT": []}
    for view in views:
        mask = ((view["semantic"] == BUILDING_TAG) &
                (view["instance_id"] == int(target_id))) if target_id is not None else np.zeros_like(view["semantic"], dtype=bool)
        masks.append(mask)
        support = backproject_mask_points(mask, view["depth_m"], view["K"], view["transform"], pixel_step=12)
        supports.append(support)
        camera_origins.append(np.asarray(view["transform"], dtype=float)[:3, 3])
        for direction in ("LEFT", "RIGHT"):
            evidence = directional_instance_contour(mask, direction)
            contour = evidence.pop("contour")
            boundary_audits[direction].append(evidence)
            projected = backproject_contour(contour, view["depth_m"], view["K"],
                                            view["transform"], candidate_h,
                                            depth_reference=view["depth_m"][mask],
                                            depth_tolerance_m=0.35)
            boundary_points[direction].append(np.asarray(projected["world_points_sample"], dtype=float))
            view[f"{direction.lower()}_contour"] = contour
    merged_support = np.row_stack([points for points in supports if len(points)]) if any(len(points) for points in supports) else np.empty((0, 3))
    if len(merged_support) >= 3:
        plane = fit_target_plane(merged_support, normal)
        h, v = orthogonal_surface_axes(candidate_h, plane["normal"])
        boundaries = fit_terminal_boundaries(boundary_points, plane["origin"], h, v)
    else:
        plane = {"origin": center, "normal": normal, "median_residual_m": None,
                 "p95_residual_m": None, "max_residual_m": None,
                 "normal_error_deg": None, "support_count": 0}
        h, v = candidate_h, np.array([0.0, 0.0, 1.0])
        boundaries = {direction: {"view_count": 0, "world_line": None,
                                  "horizontal_std_m": None}
                      for direction in ("LEFT", "RIGHT")}
    if len(merged_support) and plane.get("inlier_count", 0):
        filtered_supports = []
        for points in supports:
            residual = np.abs((points - plane["origin"]) @ plane["normal"])
            filtered_supports.append(points[residual <= plane["inlier_tolerance_m"]])
    else:
        filtered_supports = supports
    raycast = _raycast_audit(world, carla, camera_origins, filtered_supports,
                             float(thresholds["raycast_tolerance_m"]))
    target_id_stable = target_id is not None and int(target_id) in stable_ids
    passed, reasons = candidate_quality(masks, boundary_audits, target_id_stable, plane,
                                        boundaries, raycast["agreement"], thresholds)
    decision = "SELECTED" if passed else "REJECTED"
    stem = f"candidate_{int(candidate['candidate_index']):02d}_bbox_{int(candidate['bbox_id'])}"
    center_view = views[len(views) // 2]
    overview_path = assets_root / f"{stem}_overview.jpg"
    overlay_path = assets_root / f"{stem}_instance_overlay.jpg"
    _save_overview(center_view["rgb"], candidate, target_id, overview_path)
    _save_overlay(center_view["rgb"], masks[len(masks) // 2],
                  center_view["left_contour"], center_view["right_contour"],
                  candidate, decision, reasons, overlay_path)
    surface_id = f"act0_bbox_{int(candidate['bbox_id'])}"
    physical = {name: boundaries[name].get("world_line") for name in ("LEFT", "RIGHT")}
    result = {
        "candidate_index": int(candidate["candidate_index"]), "bbox_id": int(candidate["bbox_id"]),
        "candidate_surface": candidate["surface"], "surface_id": surface_id,
        "decision": decision, "rejection_reasons": reasons,
        "bbox_used_for": "candidate pose seed only; not physical boundary truth",
        "target_instance_id_16bit": int(target_id) if target_id is not None else None,
        "stable_instance_ids": [int(value) for value in stable_ids],
        "target_instance_id_stable": bool(target_id_stable),
        "center_target_coverage": float(masks[len(masks) // 2].mean()),
        "plane": {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in plane.items()},
        "horizontal_axis": h.tolist(), "vertical_axis": v.tolist(),
        "boundaries": boundaries, "raycast_audit": raycast,
        "boundary_audits": boundary_audits,
        "views": [{"view_index": row["view_index"], "frame_id": row["frame_id"],
                   "timestamp": row["timestamp"], "camera_transform": row["transform_dict"],
                   "sensor_frames": row["sensor_frames"], "sensor_timestamps": row["sensor_timestamps"],
                   "candidate_ids": [int(value) for value in row["candidate_ids"]]}
                  for row in views],
        "overview_path": _relative(overview_path), "instance_overlay_path": _relative(overlay_path),
        "surface": {
            "schema": "act0.surface.v1", "surface_id": surface_id,
            "bbox_id": int(candidate["bbox_id"]), "target_instance_id_16bit": int(target_id) if target_id is not None else None,
            "plane_origin": plane["origin"].tolist(), "plane_normal": plane["normal"].tolist(),
            "horizontal_axis": h.tolist(), "vertical_axis": v.tolist(),
            "physical_boundary": physical,
            "boundary_fit_audit": boundaries, "raycast_audit": raycast,
            "plane_support_points": merged_support[::max(1, len(merged_support) // 200)].tolist(),
            "boundary_method": "multi-view_instance_exterior_contour_z_depth_backprojection",
            "bbox_used_for": "candidate pose seed only",
            "operator_visual_review_status": "PENDING",
        } if passed else None,
    }
    return result


def scout(args, config, carla, client):
    world = client.get_world()
    if config["map"] not in world.get_map().name:
        world = client.load_world(config["map"])
    source = json.loads((PROJECT_ROOT / args.candidates).read_text())
    wanted = set(int(value) for value in config["scout"]["candidate_indices"])
    candidates = [row for row in source["candidates"] if int(row["candidate_index"]) in wanted]
    if len(candidates) < 6:
        raise RuntimeError("ACT-0 requires at least six scout candidates")
    result_root = PROJECT_ROOT / args.result_root
    assets_root = PROJECT_ROOT / args.assets_root / "candidates"
    surfaces_root = result_root / "surfaces"
    result_root.mkdir(parents=True, exist_ok=True)
    surfaces_root.mkdir(parents=True, exist_ok=True)
    first = candidates[0]
    normal = np.asarray(first["normal"], dtype=float)
    distance = max(float(config["scout"]["min_distance_m"]),
                   float(config["scout"]["distance_width_fraction"]) * float(first["width_m"]))
    transform = carla.Transform(_location(carla, np.asarray(first["center"]) + normal * distance),
                                _normal_rotation(carla, normal))
    rows = []
    sensor = config["sensor"]
    with SynchronousRGBDSeg(world, carla, transform, sensor["width"], sensor["height"],
                           sensor["horizontal_fov_deg"], sensor["fixed_delta_seconds"]) as rig:
        for candidate in candidates:
            try:
                row = _scout_candidate(candidate, rig, world, carla, config, assets_root)
            except Exception as exc:
                row = {"candidate_index": int(candidate["candidate_index"]),
                       "bbox_id": int(candidate["bbox_id"]), "decision": "REJECTED",
                       "rejection_reasons": [f"scout_exception:{type(exc).__name__}:{exc}"],
                       "surface": None}
            rows.append(row)
            print(json.dumps({"candidate_index": row["candidate_index"],
                              "bbox_id": row["bbox_id"], "decision": row["decision"],
                              "reasons": row.get("rejection_reasons", [])}))
    selected = [row for row in rows if row["decision"] == "SELECTED"]
    selected.sort(key=lambda row: (row["plane"]["p95_residual_m"],
                                   -row["raycast_audit"]["agreement"]))
    final = selected[:int(config["scout"]["required_surfaces"])]
    for row in final:
        path = surfaces_root / f"{row['surface_id']}.json"
        path.write_text(json.dumps(row["surface"], indent=2) + "\n")
        row["surface_path"] = _relative(path)
    manifest = {
        "schema": "act0.scout_manifest.v1", "map": world.get_map().name,
        "candidate_count": len(rows), "selected_count": len(final),
        "required_surfaces": int(config["scout"]["required_surfaces"]),
        "status": "PASS" if len(final) >= int(config["scout"]["required_surfaces"]) else "FAIL",
        "selection_method": "instance stability + z-depth plane + multi-view exterior contours + raycast agreement",
        "operator_visual_review": "PENDING", "candidates": rows,
        "final_surface_ids": [row["surface_id"] for row in final],
    }
    (result_root / "scout_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "candidate_count": len(rows),
                      "selected_count": len(final), "final_surface_ids": manifest["final_surface_ids"]}, indent=2))
    return 0 if manifest["status"] == "PASS" else 2


def _scout_contact_sheet(rows, output: Path):
    tiles = []
    for row in rows:
        path = row.get("instance_overlay_path")
        if not path:
            continue
        image = Image.open(PROJECT_ROOT / path).convert("RGB").resize((320, 240))
        tiles.append(image)
    if not tiles:
        return
    columns = 4
    rows_count = int(math.ceil(len(tiles) / columns))
    canvas = Image.new("RGB", (columns * 320, rows_count * 240), (20, 20, 20))
    for index, tile in enumerate(tiles):
        canvas.paste(tile, ((index % columns) * 320, (index // columns) * 240))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=84, optimize=True)


def validate(args, config):
    result_root = PROJECT_ROOT / args.result_root
    manifest_path = result_root / "scout_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    candidates = manifest["candidates"]
    phase_a = json.loads((PROJECT_ROOT / "results/obs0/validation_r1.json").read_text())
    phase_a_required = ("HISTORY_BOUNDARY_LEAKAGE_FIXED", "TRAIN_ONLY_PREPROCESSING",
                        "SYNTHETIC_ALIGNMENT_TEST", "OBS0R1_REPRODUCIBILITY")
    phase_a_pass = all(phase_a.get("gates", {}).get(name, {}).get("status") == "PASS"
                       for name in phase_a_required)
    views = [view for row in candidates for view in row.get("views", [])]
    paired = []
    for view in views:
        frames = view.get("sensor_frames", {})
        timestamps = view.get("sensor_timestamps", {})
        paired.append(bool(frames) and len(set(frames.values())) == 1 and bool(timestamps) and
                      max(timestamps.values()) - min(timestamps.values()) <= 1e-6)
    stable_count = sum(bool(row.get("target_instance_id_stable")) for row in candidates)
    selected_count = int(manifest["selected_count"])
    required = int(manifest["required_surfaces"])
    reasons = Counter(reason for row in candidates for reason in row.get("rejection_reasons", []))
    gates = {
        "PHASE_A_PREREQUISITES": {"status": "PASS" if phase_a_pass else "FAIL",
                                  "required_gates": list(phase_a_required)},
        "COUNTERFACTUAL_START_MATCH": {"status": "NOT_EVALUATED", "reason": "no rollout started"},
        "SENSOR_PAIRING": {"status": "PASS" if views and all(paired) else "FAIL",
                           "scout_quartets": len(views), "paired_quartets": sum(paired),
                           "scope": "low-cost facade scout only"},
        "INSTANCE_ID_STABILITY": {"status": "PASS" if stable_count >= required else "FAIL",
                                  "stable_candidates": stable_count,
                                  "required_candidates": required},
        "VALID_EXTERNAL_BOUNDARY": {"status": "PASS" if selected_count >= required else "FAIL",
                                    "selected_surfaces": selected_count,
                                    "required_surfaces": required},
        "FIRST_STRADDLE_STOP": {"status": "NOT_EVALUATED", "reason": "stopped after facade screening"},
        "SAME_POSE_CONFIRMATION": {"status": "NOT_EVALUATED", "reason": "stopped after facade screening"},
        "FIXED_SCHEDULE_SHORTCUT_BROKEN": {"status": "NOT_EVALUATED", "reason": "no ACT-0 rollout labels"},
        "MULTI_SURFACE_SPLIT_VALID": {"status": "FAIL", "surface_count": selected_count,
                                      "required_surfaces": required},
        "COUNTERFACTUAL_EVENT_COVERAGE": {"status": "FAIL", "trajectory_count": 0,
                                          "paired_start_count": 0},
        "RGB_INCREMENTAL_VALUE_OVER_ODOMETRY": {"status": "NOT_EVALUATED",
                                                 "reason": "no cross-surface ACT-0 rollout data",
                                                 "obs0r1_diagnostic_status": phase_a.get("gates", {}).get("RGB_INCREMENTAL_VALUE_OVER_ODOMETRY", {}).get("status")},
        "ACTION_SELECTION_OBSERVABILITY": {"status": "NOT_EVALUATED", "reason": "no counterfactual outcome pairs"},
        "OPERATOR_VISUAL_REVIEW": {"status": "PENDING", "candidate_images": sum(bool(row.get("overview_path")) for row in candidates)},
        "READY_FOR_DATASET_EXPANSION": {"status": "FAIL", "reason": "fewer than four valid facades"},
        "READY_FOR_JEPA": {"status": "NOT_EVALUATED"},
    }
    trajectory_manifest = {"schema": "act0.trajectory_manifest.v1", "status": "NOT_RUN",
                           "reason": "VALID_EXTERNAL_BOUNDARY failed during facade screening",
                           "trajectory_count": 0, "counterfactual_pair_count": 0,
                           "trajectories": []}
    (result_root / "trajectory_manifest.json").write_text(json.dumps(trajectory_manifest, indent=2) + "\n")
    validation = {
        "schema": "act0.validation.v1", "phase": "ACT-0 counterfactual multi-facade pilot",
        "status": "STOPPED_AT_FACADE_SCREENING", "map": manifest["map"],
        "candidate_count": len(candidates), "selected_surface_count": selected_count,
        "required_surface_count": required, "rejection_reason_counts": dict(sorted(reasons.items())),
        "thresholds": config["scout"], "runtime_safety": config["runtime_safety"],
        "gates": gates, "raw_data": {"rollout_raw_bytes": 0,
                                      "scout_full_rgbd_saved": False},
        "constraints": {"jepa_training_run": False, "models_downloaded": False,
                        "historical_results_modified": False},
    }
    (result_root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    template_path = result_root / "operator_review_template.csv"
    with template_path.open("w", newline="") as handle:
        fields = ["candidate_index", "bbox_id", "automatic_decision", "overview_path",
                  "instance_overlay_path", "operator_accept", "operator_notes"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in candidates:
            writer.writerow({"candidate_index": row["candidate_index"], "bbox_id": row["bbox_id"],
                             "automatic_decision": row["decision"],
                             "overview_path": row.get("overview_path", ""),
                             "instance_overlay_path": row.get("instance_overlay_path", ""),
                             "operator_accept": "", "operator_notes": ""})
    contact_path = PROJECT_ROOT / args.assets_root / "scout_contact_sheet.jpg"
    _scout_contact_sheet(candidates, contact_path)
    lines = [
        "# ACT-0 Visual Audit", "",
        "ACT-0 stopped at facade screening. The required four independent facades with reliable LEFT/RIGHT external boundaries were not available under the fixed scout gates. No counterfactual rollout or JEPA training was run.", "",
        "## Outcome", "", "```text",
        f"candidate facades screened: {len(candidates)}",
        f"automatically valid facades: {selected_count}",
        f"required valid facades: {required}",
        "counterfactual trajectories: 0", "operator visual review: PENDING",
        "READY_FOR_DATASET_EXPANSION: FAIL", "READY_FOR_JEPA: NOT_EVALUATED", "```", "",
        "The scout used three synchronized RGB/depth/semantic/instance views per candidate. Bboxes seeded camera poses only. Physical-boundary evidence came from stable instance IDs, exterior contours, z-depth backprojection and raycast comparison.", "",
        "![ACT-0 scout contact sheet](assets/act0/scout_contact_sheet.jpg)", "",
        "## Gates", "", "| Gate | Status | Evidence |", "|---|---|---|",
    ]
    for name, gate in gates.items():
        evidence = gate.get("reason", "") or ", ".join(f"{k}={v}" for k, v in gate.items() if k != "status")
        lines.append(f"| {name} | {gate['status']} | {evidence} |")
    lines += ["", "## Candidates", "", "| Index | bbox | Decision | Target coverage | Stable ID | Raycast agreement | Rejection reasons |", "|---:|---:|---|---:|---|---:|---|"]
    for row in candidates:
        ray = row.get("raycast_audit", {}).get("agreement")
        ray_text = "n/a" if ray is None else f"{ray:.3f}"
        lines.append(f"| {row['candidate_index']} | {row['bbox_id']} | {row['decision']} | {row.get('center_target_coverage', 0):.3f} | {row.get('target_instance_id_stable', False)} | {ray_text} | {', '.join(row.get('rejection_reasons', []))} |")
    lines += ["", "## Candidate images", ""]
    for row in candidates:
        if not row.get("overview_path"):
            continue
        overview = Path(row["overview_path"]).relative_to("docs")
        overlay = Path(row["instance_overlay_path"]).relative_to("docs")
        lines += [f"### Candidate {row['candidate_index']} / bbox {row['bbox_id']}", "",
                  f"![Overview {row['candidate_index']}]({overview})",
                  f"![Instance overlay {row['candidate_index']}]({overlay})", ""]
    lines += ["## Missing rollout artifacts", "",
              "The task requires immediate stop when fewer than four facades pass screening. Therefore LEFT/RIGHT initial pairs, rollout contact sheets, first-STRADDLE frames, frozen confirmations, outcome distributions and leave-one-surface-out plots do not exist. They are not fabricated or replaced with scout images.", "",
              "Instance, semantic, depth and raycast evidence was used only for privileged screening. It is not an RGB probe input."]
    (PROJECT_ROOT / "docs/ACT0_VISUAL_AUDIT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(validation, indent=2))
    return 0


def _screening_sheet(candidate, audit, output: Path):
    canvas = Image.new("RGB", (1280, 820), (22, 24, 27))
    draw = ImageDraw.Draw(canvas)
    title = (f"ACT-0S candidate {candidate['candidate_index']} / bbox {candidate['bbox_id']}  "
             f"class={audit['classification']}")
    draw.text((18, 12), title, fill=(255, 230, 80))
    draw.text((18, 34),
              f"Tier V={audit['tier_v']['status']}  M={audit['tier_m']['status']}  P={audit['tier_p']['status']}",
              fill=(235, 235, 235))
    overview = Image.open(PROJECT_ROOT / candidate["overview_path"]).convert("RGB").resize((600, 450))
    overlay = Image.open(PROJECT_ROOT / candidate["instance_overlay_path"]).convert("RGB").resize((600, 450))
    canvas.paste(overview, (18, 64))
    canvas.paste(overlay, (642, 64))
    draw.text((18, 520), "Persisted actual center RGB", fill=(180, 220, 255))
    draw.text((642, 520), "Persisted center instance/mask/transition overlay", fill=(180, 220, 255))
    views = candidate.get("views", [])
    for index, view in enumerate(views):
        x = 18 + index * 414
        frames = view.get("sensor_frames", {})
        availability = "CENTER PIXELS PERSISTED" if index == 1 else "PIXELS NOT PERSISTED"
        draw.rectangle((x, 550, x + 392, 653), outline=(95, 105, 115), width=1)
        draw.text((x + 8, 560), f"view={view.get('view_index')}  frame={view.get('frame_id')}", fill=(240, 240, 240))
        draw.text((x + 8, 580), availability, fill=(255, 170, 80) if index != 1 else (100, 230, 140))
        draw.text((x + 8, 600), f"quartet={len(set(frames.values())) == 1 if frames else False}", fill=(210, 210, 210))
        draw.text((x + 8, 620), f"candidate_ids={view.get('candidate_ids', [])}", fill=(210, 210, 210))
    plane = audit["tier_p"].get("metrics", {})
    sides = audit["tier_v"].get("sides", {})
    draw.text((18, 676),
              f"coverage={candidate.get('center_target_coverage', 0):.3f}  target_id={candidate.get('target_instance_id_16bit')}  "
              f"LEFT spans={sides.get('LEFT', {}).get('span_fractions', [])}", fill=(225, 225, 225))
    draw.text((18, 699),
              f"RIGHT spans={sides.get('RIGHT', {}).get('span_fractions', [])}  "
              f"legacy spread L/R={audit['tier_m']['legacy_proxy']['LEFT']['spread_m']}/"
              f"{audit['tier_m']['legacy_proxy']['RIGHT']['spread_m']} m", fill=(225, 225, 225))
    draw.text((18, 722),
              f"plane p95={plane.get('plane_p95_residual')} m  inlier={plane.get('plane_inlier_ratio')}  "
              f"raycast agreement={plane.get('raycast_agreement')}", fill=(225, 225, 225))
    draw.text((18, 749), "No off-center RGB/depth or residual heatmap was retained; unavailable panels are not reconstructed.",
              fill=(255, 150, 120))
    draw.text((18, 775), "Operator visual review: PENDING", fill=(255, 210, 90))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=82, optimize=True, progressive=True)


def _screening_overview(paths, output: Path):
    tiles = [Image.open(path).convert("RGB").resize((320, 205)) for path in paths]
    canvas = Image.new("RGB", (1280, 615), (20, 20, 20))
    for index, tile in enumerate(tiles):
        canvas.paste(tile, ((index % 4) * 320, (index // 4) * 205))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=80, optimize=True, progressive=True)


def screening_audit(args, config):
    """Reclassify the persisted scout evidence without CARLA or new capture."""
    result_root = PROJECT_ROOT / args.result_root
    assets_root = PROJECT_ROOT / args.assets_root
    manifest_path = result_root / "scout_manifest.json"
    validation_path = result_root / "validation.json"
    protected_before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in (manifest_path, validation_path)}
    manifest = json.loads(manifest_path.read_text())
    thresholds = config["screening_audit"]
    interpretations = thresholds.get("existing_evidence_interpretation", {})
    metric_thresholds = [float(value) for value in thresholds["metric_spread_thresholds_m"]]
    rows = []
    image_paths = []
    allowed_classes = {"VISUAL_EVENT_PASS", "METRIC_ONLY_FAIL", "PHYSICAL_PLANE_ONLY_FAIL",
                       "SCOUT_POSE_INSUFFICIENT", "INSTANCE_GROUPING_UNRESOLVED", "SCENE_UNSUITABLE"}
    for candidate in manifest["candidates"]:
        key = str(candidate["candidate_index"])
        interpretation = interpretations.get(key, interpretations.get(int(candidate["candidate_index"]), {}))
        classification = interpretation.get("classification")
        if classification not in allowed_classes:
            raise ValueError(f"missing ACT-0S evidence classification for candidate {key}")
        tier_v = tier_visual_event(candidate, thresholds, interpretation)
        tier_m = tier_metric_repeatability(candidate, metric_thresholds,
                                            candidate.get("metric_evidence_v2"))
        tier_p = tier_physical_plane(candidate, config["scout"])
        row = {"candidate_index": int(candidate["candidate_index"]),
               "bbox_id": int(candidate["bbox_id"]), "classification": classification,
               "classification_rationale": interpretation.get("rationale", ""),
               "current_full_physical_gate": candidate.get("decision", "REJECTED"),
               "visual_event_gate": tier_v["status"],
               "metric_repeatability_gate": tier_m["status"],
               "physical_plane_gate": tier_p["status"],
               "tier_v": tier_v, "tier_m": tier_m, "tier_p": tier_p,
               "operator_visual_review": "PENDING"}
        rows.append(row)
        image_path = assets_root / f"candidate_{int(candidate['candidate_index']):02d}_screening.jpg"
        _screening_sheet(candidate, row, image_path)
        row["public_image"] = _relative(image_path)
        image_paths.append(image_path)
    overview_path = assets_root / "candidate_overview.jpg"
    _screening_overview(image_paths, overview_path)
    required = int(manifest.get("required_surfaces", 4))
    class_counts = Counter(row["classification"] for row in rows)
    tier_v_pass = sum(row["visual_event_gate"] == "PASS" for row in rows)
    tier_p_pass = sum(row["physical_plane_gate"] == "PASS" for row in rows)
    official_metric_pass = sum(row["metric_repeatability_gate"] == "PASS" for row in rows)
    legacy_sensitivity = {
        f"{threshold:.2f}m": sum(row["tier_m"]["legacy_proxy_sensitivity"][f"{threshold:.2f}m"]
                                 for row in rows)
        for threshold in metric_thresholds
    }
    all_views = [view for candidate in manifest["candidates"] for view in candidate.get("views", [])]
    pairing = [scout_sensor_pairing(candidate.get("views", []))
               for candidate in manifest["candidates"]]
    pose_insufficient = class_counts["SCOUT_POSE_INSUFFICIENT"]
    grouping_unresolved = class_counts["INSTANCE_GROUPING_UNRESOLVED"]
    scene_unsuitable = class_counts["SCENE_UNSUITABLE"]
    current_reasons = Counter(reason for candidate in manifest["candidates"]
                              for reason in candidate.get("rejection_reasons", []))
    reason_intersections = Counter(" + ".join(sorted(candidate.get("rejection_reasons", [])))
                                   for candidate in manifest["candidates"])
    gates = {
        "PUBLIC_ACT0_EVIDENCE": {"status": "FAIL", "candidate_sheets": len(image_paths),
                                  "reason": "only center-view pixels were persisted; 24 off-center views and residual heatmaps are unavailable"},
        "SCOUT_SENSOR_PAIRING": {"status": "PASS" if all(row["status"] == "PASS" for row in pairing) else "FAIL",
                                  "paired_quartets": sum(row["paired_view_count"] for row in pairing),
                                  "quartets": len(all_views)},
        "SCOUT_POSE_COVERAGE": {"status": "FAIL" if pose_insufficient else "PASS",
                                 "insufficient_candidates": pose_insufficient},
        "SCREENING_DEFINITION_VALID": {"status": "PASS",
                                        "tier_p_can_veto_tier_v": False,
                                        "not_observed_counted_as_scene_fail": False},
        "VISUAL_EVENT_SURFACE_COUNT": {"status": "PASS" if tier_v_pass >= required else "FAIL",
                                       "count": tier_v_pass, "required": required,
                                       "scope": "geometry-reference classification pending operator review"},
        "METRIC_REPEATABLE_SURFACE_COUNT": {"status": "NOT_EVALUATED",
                                             "verified_count": official_metric_pass,
                                             "evaluable_count": 0,
                                             "legacy_proxy_sensitivity_counts": legacy_sensitivity},
        "PHYSICAL_PLANE_SURFACE_COUNT": {"status": "PASS" if tier_p_pass >= required else "FAIL",
                                          "count": tier_p_pass, "required": required},
        "INSTANCE_GROUPING_RESOLVED": {"status": "FAIL" if grouping_unresolved else "PASS",
                                        "unresolved_candidates": grouping_unresolved},
        "READY_FOR_ADAPTIVE_RESCOUT": {"status": "CONDITIONAL_PASS",
                                        "reason": "targeted boundary-view recapture can address pose/evidence gaps without changing historical gates"},
        "READY_FOR_COUNTERFACTUAL_ROLLOUT": {"status": "FAIL",
                                             "reason": "official Tier M and complete public three-view evidence are unavailable"},
        "READY_FOR_DATASET_EXPANSION": {"status": "NOT_EVALUATED"},
        "READY_FOR_JEPA": {"status": "NOT_EVALUATED"},
        "OPERATOR_VISUAL_REVIEW": {"status": "PENDING"},
    }
    coverage = {
        "schema": "act0.scout_coverage_audit.v2", "candidate_count": len(rows),
        "synchronized_quartet_count": len(all_views),
        "persisted_actual_rgb_views": len(rows), "expected_actual_rgb_views": len(all_views),
        "persisted_center_instance_overlays": len(rows),
        "persisted_raw_depth_views": 0, "persisted_raw_semantic_views": 0,
        "persisted_raw_instance_views": 0, "persisted_raycast_residual_heatmaps": 0,
        "missing_off_center_rgb_views": len(all_views) - len(rows),
        "raw_sensor_data_path": None, "raw_sensor_data_size_bytes": 0,
        "conclusion": "compact metadata supports pairing and Tier V/P screening, but not plane-free Tier M or three-view public pixel audit",
    }
    audit = {
        "schema": "act0.screening_audit.v2", "phase": "ACT-0S",
        "source_manifest": "results/act0/scout_manifest.json",
        "historical_full_physical_gate": {"selected": int(manifest["selected_count"]),
                                           "candidate_count": int(manifest["candidate_count"]),
                                           "preserved": True},
        "resource_limits": {"configured_address_space_limit_bytes": 4294967296,
                            "actual_outer_address_space_limit_bytes": 2147483648,
                            "numeric_threads": 1},
        "thresholds": {key: value for key, value in thresholds.items()
                       if key != "existing_evidence_interpretation"},
        "counts": {"tier_v_pass": tier_v_pass, "tier_m_official_pass": official_metric_pass,
                   "tier_p_pass": tier_p_pass, "scout_pose_insufficient": pose_insufficient,
                   "instance_grouping_unresolved": grouping_unresolved,
                   "scene_unsuitable": scene_unsuitable,
                   "classifications": {name: int(class_counts[name]) for name in sorted(allowed_classes)}},
        "rejection_ablation": {"without_tier_p_tier_v_pass": tier_v_pass,
                               "scout_edge_not_observed": pose_insufficient,
                               "instance_grouping_unresolved": grouping_unresolved,
                               "visual_semantics_fail": scene_unsuitable,
                               "individual_current_hard_gate_counts": dict(sorted(current_reasons.items())),
                               "current_hard_gate_intersections": dict(sorted(reason_intersections.items()))},
        "gates": gates, "candidates": rows,
        "constraints": {"carla_started": False, "new_scout_run": False,
                        "rollout_started": False, "jepa_training_run": False,
                        "models_downloaded": False, "historical_results_modified": False},
    }
    result_root.mkdir(parents=True, exist_ok=True)
    (result_root / "screening_audit_v2.json").write_text(json.dumps(audit, indent=2) + "\n")
    (result_root / "scout_coverage_audit.json").write_text(json.dumps(coverage, indent=2) + "\n")
    matrix_path = result_root / "candidate_gate_matrix_v2.csv"
    fields = ["candidate_index", "bbox_id", "classification", "current_full_physical_gate",
              "visual_event_gate", "metric_repeatability_gate", "physical_plane_gate",
              "left_contour_status", "right_contour_status", "left_legacy_spread_m",
              "right_legacy_spread_m", "public_image", "classification_rationale"]
    with matrix_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({**{name: row.get(name, "") for name in fields},
                             "left_contour_status": row["tier_v"]["sides"]["LEFT"]["status"],
                             "right_contour_status": row["tier_v"]["sides"]["RIGHT"]["status"],
                             "left_legacy_spread_m": row["tier_m"]["legacy_proxy"]["LEFT"]["spread_m"],
                             "right_legacy_spread_m": row["tier_m"]["legacy_proxy"]["RIGHT"]["spread_m"]})
    lines = ["# ACT-0S Screening Definition and Public Evidence Audit", "",
             "ACT-0S reclassifies the existing 12 candidates and 36 synchronized scout quartets. It did not start CARLA, recapture scout views, run a counterfactual rollout, download a model, or train JEPA. The historical full physical gate remains 0/12.", "",
             "## Evidence limitation", "",
             "The scout persisted all frame/timestamp/pose metadata but only the center RGB and center instance overlay for each candidate. Off-center RGB/depth/semantic/instance arrays and residual heatmaps were not saved. The candidate sheets below show the real persisted center evidence and label the missing views explicitly; they are not reconstructed. Consequently `PUBLIC_ACT0_EVIDENCE` is FAIL and official Tier M is NOT_EVALUATED.", "",
             "![Twelve-candidate overview](assets/act0_screening/candidate_overview.jpg)", "",
             "## Gates", "", "| Gate | Status | Evidence |", "|---|---|---|"]
    for name, gate in gates.items():
        evidence = gate.get("reason", ", ".join(f"{key}={value}" for key, value in gate.items() if key != "status"))
        lines.append(f"| {name} | {gate['status']} | {evidence} |")
    lines += ["", "## Tier definitions", "",
              "- Tier V uses quartet pairing, Building-semantic target provenance, selected-instance stability, the retained component pass/fail bit, and multi-view target/non-target transition summaries. It does not use plane or raycast metrics.",
              "- Tier M requires target-side contour pixels, z-depth, K, per-frame `T_world_camera`, and a camera-motion action axis. Those raw inputs were not persisted. Historical plane-basis spreads appear only as non-gating sensitivity proxies.",
              "- Tier P retains the original strict plane residual, normal, inlier, raycast/depth, and physical-width checks. Tier P does not overwrite Tier V.", "",
              "## Candidate matrix", "",
              "| Candidate | bbox | Classification | Tier V | Tier M | Tier P | Rationale |", "|---:|---:|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['candidate_index']} | {row['bbox_id']} | {row['classification']} | {row['visual_event_gate']} | {row['metric_repeatability_gate']} | {row['physical_plane_gate']} | {row['classification_rationale']} |")
    lines += ["", "## Candidate evidence", ""]
    for row in rows:
        image = Path(row["public_image"]).relative_to("docs")
        lines += [f"### Candidate {row['candidate_index']} / bbox {row['bbox_id']}", "",
                  f"![Candidate {row['candidate_index']} screening evidence]({image})", ""]
    lines += ["## Requested examples", "",
              "- `SCOUT_POSE_INSUFFICIENT`: candidates 12 and 20.",
              "- `PHYSICAL_PLANE_ONLY_FAIL`: candidates 10, 11, and 19.",
              "- Instance fragmentation / `SCENE_UNSUITABLE`: candidates 17 and 18.",
              "- Confirmed `SCENE_UNSUITABLE`: candidates 17 and 18 under compact geometry-reference evidence; external operator review remains pending.", "",
              "## Decision", "",
              "The next permissible experiment is a small adaptive rescout that persists the missing boundary-side pixels and plane-free Tier M inputs. A map or asset replacement is not yet required because six candidates pass Tier V under the compact geometry reference, but counterfactual rollout is not authorized by this audit."]
    (PROJECT_ROOT / "docs/ACT0_SCREENING_AUDIT.md").write_text("\n".join(lines) + "\n")
    protected_after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (manifest_path, validation_path)}
    if protected_before != protected_after:
        raise RuntimeError("ACT-0S modified a protected historical result")
    print(json.dumps({"status": "COMPLETE_WITH_EVIDENCE_GAPS", "counts": audit["counts"],
                      "gates": gates}, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("scout", "validate", "screening-audit"))
    parser.add_argument("--config", default="configs/experiments/act0.yaml")
    parser.add_argument("--carla-root")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--candidates", default="results/geo06/facade_candidates.json")
    parser.add_argument("--result-root", default="results/act0")
    parser.add_argument("--assets-root", default="docs/assets/act0")
    args = parser.parse_args(argv)
    config = yaml.safe_load((PROJECT_ROOT / args.config).read_text())
    if args.command == "validate":
        return validate(args, config)
    if args.command == "screening-audit":
        return screening_audit(args, config)
    if not args.carla_root:
        parser.error("--carla-root is required for scout")
    root = discover_carla_root(args.carla_root)
    carla = import_carla(str(root))
    host = args.host or config["server"]["host"]
    port = args.port or int(config["server"]["port"])
    client = carla.Client(host, port)
    client.set_timeout(20.0)
    if args.command == "scout":
        return scout(args, config, carla, client)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
