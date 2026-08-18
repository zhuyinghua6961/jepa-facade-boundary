#!/usr/bin/env python3
"""AVS-0 bounded capture and cross-surface endpoint-RGB audit."""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from boundary_sweep.active_view import (evaluate_preregistered_gates,
                                         paired_policy_rows, policy_summary,
                                         stratified_start_bootstrap,
                                         surface_leave_one_out)
from boundary_sweep.act0r import (boundary_type_consensus,
                                  config_outcome_override_audit,
                                  official_tier_m,
                                  physical_termination_pixel_gate,
                                  tier_v_from_pixel_frames,
                                  verify_manifest_hashes)
from boundary_sweep.carla_utils import import_carla, transform_from_matrix
from boundary_sweep.observability import fixed_length_descriptor, rgb_descriptor
from boundary_sweep.segmentation import (BUILDING_TAG, bgra_array,
                                         decode_instance_channels, rgb_from_bgra)
from boundary_sweep.sensors import SynchronousRGBDSeg
from check_sensor_health import (TraceWriter, _act0r2_center_boundary_present,
                                 _attempt_capture, _cf0_offset_pose,
                                 _cf0_sample_metric)
from run_probe0 import eligible_truth


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def candidate_pose(candidate: dict) -> np.ndarray:
    normal = np.asarray(candidate["normal"], dtype=float)
    normal /= np.linalg.norm(normal)
    axis = np.asarray(candidate["action_axis"], dtype=float)
    axis /= np.linalg.norm(axis)
    matrix = np.eye(4, dtype=float)
    matrix[:3, 0] = -normal
    matrix[:3, 1] = axis
    matrix[:3, 2] = np.cross(matrix[:3, 0], matrix[:3, 1])
    matrix[:3, 3] = np.asarray(candidate["center"], dtype=float) + normal * float(
        candidate["distance_m"])
    return matrix


def save_frame(rig, sample: dict, raw_root: Path, stem: str, candidate_id: int,
               bbox_id: int, target_id: int, role: str, start_id: str | None,
               direction: str, offset_m: float, metric: dict) -> dict:
    metadata = rig.save(sample, raw_root, stem)
    metadata.update({
        "experiment": "AVS-0", "candidate_index": int(candidate_id),
        "surface_id": f"candidate_{int(candidate_id)}", "bbox_id": int(bbox_id),
        "target_instance_id": int(target_id), "capture_role": role,
        "role_used_as_scientific_label": False, "start_id": start_id,
        "direction": direction, "signed_offset_m_gt_audit_only": float(offset_m),
        "pixel_metric": metric,
        "metadata_path": str(raw_root / f"{stem}.json"),
    })
    (raw_root / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def capture_at(rig, carla, center: np.ndarray, axis: np.ndarray, signed_offset: float,
               action: dict, trace: TraceWriter) -> tuple[dict, dict]:
    pose = _cf0_offset_pose(center, axis, signed_offset)
    rig.set_transform(transform_from_matrix(carla, pose))
    settle = rig.settle(3)
    sample = _attempt_capture(rig, action, trace)
    return sample, settle


def frame_metrics(sample: dict, target_id: int, axis: np.ndarray,
                  act0r_config: dict) -> dict[str, dict]:
    return {direction: _cf0_sample_metric(sample, direction, target_id, axis, act0r_config)
            for direction in ("LEFT", "RIGHT")}


def resolve_center_target_instance(sample: dict) -> dict:
    """Resolve the current-session target ID from the optical-center Building pixel."""
    instance_rgb = rgb_from_bgra(bgra_array(sample["data"]["instance"]))
    decoded = decode_instance_channels(instance_rgb)
    semantic = decoded["semantic_tag"]
    instance = decoded["instance_id_16bit"]
    center_y, center_x = semantic.shape[0] // 2, semantic.shape[1] // 2
    if int(semantic[center_y, center_x]) == int(BUILDING_TAG):
        target_id = int(instance[center_y, center_x])
        method = "optical_center_building_pixel"
    else:
        values, counts = np.unique(instance[semantic == BUILDING_TAG], return_counts=True)
        if not len(values):
            return {"status": "FAIL", "reason": "no Building instance pixels at CENTER"}
        target_id = int(values[int(np.argmax(counts))])
        method = "largest_center_view_building_instance"
    coverage = float(np.mean((semantic == BUILDING_TAG) & (instance == target_id)))
    return {"status": "PASS", "target_instance_id": target_id,
            "method": method, "center_view_coverage": coverage}


def no_external_boundary(metrics: dict[str, dict], act0r_config: dict) -> bool:
    return all(not _act0r2_center_boundary_present(metrics[direction], act0r_config)
               for direction in ("LEFT", "RIGHT"))


def sensor_quartet_paired(entry: dict) -> bool:
    frames = entry.get("sensor_frames", {})
    timestamps = entry.get("sensor_timestamps", {})
    return (set(frames) == {"rgb", "depth", "semantic", "instance"} and
            len(set(frames.values())) == 1 and set(timestamps) == set(frames) and
            max(timestamps.values()) - min(timestamps.values()) <= 1e-6)


def qualify_candidate(rig, carla, candidate_id: int, candidate: dict, config: dict,
                      act0r_config: dict, raw_root: Path, trace: TraceWriter,
                      saved_counter: list[int]) -> dict:
    center = candidate_pose(candidate)
    axis = np.asarray(candidate["action_axis"], dtype=float)
    axis /= np.linalg.norm(axis)
    configured_target_id = int(candidate["target_instance_id"])
    result = {"candidate_index": int(candidate_id), "bbox_id": int(candidate["bbox_id"]),
              "configured_target_instance_id_provenance_only": configured_target_id,
              "status": "FAIL", "directions": {}}
    evidence = []
    reference_samples = {}
    center_sample, center_settle = capture_at(rig, carla, center, axis, 0.0, {
        "experiment": "AVS-0", "phase": "qualification", "candidate_index": candidate_id,
        "capture_role": "CENTER", "role_used_as_scientific_label": False}, trace)
    target_resolution = resolve_center_target_instance(center_sample)
    if target_resolution["status"] != "PASS":
        result["reason"] = target_resolution["reason"]
        result["target_resolution"] = target_resolution
        return result
    target_id = int(target_resolution["target_instance_id"])
    result["resolved_target_instance_id"] = target_id
    result["target_resolution"] = target_resolution
    reference_samples["CENTER"] = {
        "sample": center_sample, "settle": center_settle,
        "metrics": frame_metrics(center_sample, target_id, axis, act0r_config)}
    for role, offset in (("LEFT_1M", -1.0), ("RIGHT_1M", 1.0)):
        sample, settle = capture_at(rig, carla, center, axis, offset, {
            "experiment": "AVS-0", "phase": "qualification", "candidate_index": candidate_id,
            "capture_role": role, "role_used_as_scientific_label": False}, trace)
        metrics = frame_metrics(sample, target_id, axis, act0r_config)
        reference_samples[role] = {"sample": sample, "metrics": metrics, "settle": settle}
    center_coverage = reference_samples["CENTER"]["metrics"]["LEFT"]["target_coverage"]
    endpoint_absent = all(no_external_boundary(reference_samples[role]["metrics"], act0r_config)
                          for role in ("LEFT_1M", "RIGHT_1M"))
    center_absent = no_external_boundary(reference_samples["CENTER"]["metrics"], act0r_config)
    for role, offset in (("CENTER", 0.0), ("LEFT_1M", -1.0), ("RIGHT_1M", 1.0)):
        if saved_counter[0] >= int(config["qualification"]["maximum_saved_quartets"]):
            raise RuntimeError("AVS-0 qualification saved-frame limit reached")
        item = reference_samples[role]
        metadata = save_frame(rig, item["sample"], raw_root,
                              f"qual_candidate_{candidate_id:02d}_{role.lower()}",
                              candidate_id, candidate["bbox_id"], target_id, role, None,
                              "NONE", offset, item["metrics"])
        metadata["settle"] = item["settle"]
        (raw_root / Path(metadata["metadata_path"]).name).write_text(
            json.dumps(metadata, indent=2) + "\n")
        evidence.append(metadata); saved_counter[0] += 1
    result.update({"center_target_coverage": center_coverage,
                   "center_boundary_absent": center_absent,
                   "bilateral_probe_endpoints_preboundary": endpoint_absent,
                   "reference_frame_ids": {role: int(item["sample"]["frame_id"])
                                           for role, item in reference_samples.items()}})
    if (center_coverage < float(config["qualification"]["minimum_center_target_coverage"]) or
            not center_absent or not endpoint_absent):
        result["reason"] = "center or 1.0 m endpoint fails preboundary safety gate"
        result["evidence_frames"] = evidence
        return result
    live_frames = 3
    for direction in ("LEFT", "RIGHT"):
        sign = -1.0 if direction == "LEFT" else 1.0
        repeated_metrics, repeated_meta = [], []
        event_offset = None
        distance = float(config["qualification"]["probe_distance_m"])
        while distance <= float(config["qualification"]["maximum_search_distance_m"]) + 1e-9:
            if live_frames >= int(config["qualification"]["maximum_live_frames_per_candidate"]):
                break
            sample, settle = capture_at(rig, carla, center, axis, sign * distance, {
                "experiment": "AVS-0", "phase": "qualification_search",
                "candidate_index": candidate_id, "direction": direction,
                "distance_m_gt_audit_only": distance,
                "role_used_as_scientific_label": False}, trace)
            live_frames += 1
            metric = _cf0_sample_metric(sample, direction, target_id, axis, act0r_config)
            boundary_type = metric.get("boundary_classification", {}).get(
                "boundary_type", "UNRESOLVED")
            if (metric.get("contour_present") and boundary_type != "UNRESOLVED" and
                    float(metric.get("span_over_target_bbox_height", 0.0)) >=
                    float(act0r_config["contour"]["tier_v_min_span_over_target_bbox"])):
                event_offset = sign * distance
                repeated = [(sample, settle, metric)]
                for repeat in range(1, int(config["qualification"]["repeated_event_frames"])):
                    repeat_sample, repeat_settle = capture_at(
                        rig, carla, center, axis, event_offset, {
                            "experiment": "AVS-0", "phase": "qualification_repeat",
                            "candidate_index": candidate_id, "direction": direction,
                            "repeat_index": repeat, "role_used_as_scientific_label": False}, trace)
                    live_frames += 1
                    repeated.append((repeat_sample, repeat_settle, _cf0_sample_metric(
                        repeat_sample, direction, target_id, axis, act0r_config)))
                for repeat, (repeat_sample, repeat_settle, repeat_metric) in enumerate(repeated):
                    if saved_counter[0] >= int(config["qualification"]["maximum_saved_quartets"]):
                        raise RuntimeError("AVS-0 qualification saved-frame limit reached")
                    stem = f"qual_candidate_{candidate_id:02d}_{direction.lower()}_event_{repeat}"
                    metadata = save_frame(rig, repeat_sample, raw_root, stem, candidate_id,
                                          candidate["bbox_id"], target_id,
                                          "BOUNDARY_REPEAT", None, direction, event_offset,
                                          repeat_metric)
                    metadata["settle"] = repeat_settle
                    (raw_root / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")
                    repeated_meta.append(metadata); repeated_metrics.append(repeat_metric)
                    evidence.append(metadata); saved_counter[0] += 1
                break
            distance += float(config["qualification"]["coarse_step_m"])
        classifications = [row.get("boundary_classification", {}) for row in repeated_metrics]
        consensus = boundary_type_consensus(
            classifications, int(config["qualification"]["minimum_consensus_frames"]))
        tier_v = tier_v_from_pixel_frames(
            repeated_metrics, act0r_config["contour"],
            int(config["qualification"]["minimum_consensus_frames"]))
        tier_m = official_tier_m(
            [row.get("tier_m_frame", {}) for row in repeated_metrics],
            act0r_config["tier_m"]["sensitivity_thresholds_m"],
            act0r_config["tier_m"]["gate_spread_m"])
        physical = (consensus["status"] == "PASS" and
                    consensus["boundary_type"] == "PHYSICAL_TERMINATION")
        result["directions"][direction] = {
            "status": "PASS" if physical and tier_v["status"] == "PASS" and
                      tier_m["status"] == "PASS" else "FAIL",
            "event_signed_offset_m": event_offset, "boundary_type": consensus,
            "tier_v": tier_v, "official_tier_m": tier_m,
            "boundary_action_axis_coordinate_m": (float(np.median([
                row["tier_m_frame"]["action_axis_median_m"] for row in repeated_metrics
                if row.get("tier_m_frame", {}).get("action_axis_median_m") is not None]))
                if repeated_metrics else None),
            "frame_ids": [int(row["frame_id"]) for row in repeated_meta],
        }
    result["live_frame_count"] = live_frames
    result["evidence_frames"] = evidence
    result["status"] = "PASS" if len(result["directions"]) == 2 and all(
        row["status"] == "PASS" for row in result["directions"].values()) else "FAIL"
    result["reason"] = ("bilateral physical termination and safety gates pass" if
                        result["status"] == "PASS" else
                        "bilateral physical termination or repeatability gate failed")
    return result


def capture_surface_starts(rig, carla, qualification: dict, candidate: dict,
                           config: dict, act0r_config: dict, raw_root: Path,
                           trace: TraceWriter, saved_counter: list[int]) -> tuple[list[dict], list[dict]]:
    candidate_id = int(qualification["candidate_index"])
    center = candidate_pose(candidate)
    axis = np.asarray(candidate["action_axis"], dtype=float); axis /= np.linalg.norm(axis)
    center_coordinate = float(np.dot(center[:3, 3], axis))
    left_coordinate = float(qualification["directions"]["LEFT"][
        "boundary_action_axis_coordinate_m"])
    right_coordinate = float(qualification["directions"]["RIGHT"][
        "boundary_action_axis_coordinate_m"])
    lower_boundary, upper_boundary = sorted((left_coordinate, right_coordinate))
    margin = float(config["capture"]["probe_distance_m"] +
                   config["capture"]["boundary_safety_margin_m"])
    left_event_offset = float(qualification["directions"]["LEFT"][
        "event_signed_offset_m"])
    right_event_offset = float(qualification["directions"]["RIGHT"][
        "event_signed_offset_m"])
    lower = left_event_offset + margin
    upper = right_event_offset - margin
    if lower >= upper:
        return [], [{"reason": "no shared bilateral preboundary interval"}]
    rng = np.random.default_rng(int(config["seed"]) + candidate_id)
    offsets = rng.uniform(lower, upper, int(config["capture"]["maximum_start_attempts_per_surface"]))
    starts, rejected = [], []
    target_id = int(qualification["resolved_target_instance_id"])
    for offset in offsets.tolist():
        if len(starts) >= int(config["capture"]["starts_per_new_surface"]):
            break
        camera_coordinate = center_coordinate + offset
        left_cost = camera_coordinate - lower_boundary
        right_cost = upper_boundary - camera_coordinate
        if min(left_cost, right_cost) <= float(config["capture"]["probe_distance_m"]):
            rejected.append({"offset_m": offset, "reason": "probe endpoint crosses world boundary"})
            continue
        if abs(left_cost - right_cost) < float(config["capture"]["minimum_non_tie_distance_m"]):
            rejected.append({"offset_m": offset, "reason": "near-boundary direction tie"})
            continue
        samples = {}
        safe = True
        for role, delta, direction in (("START", 0.0, "START"),
                                       ("LEFT_1M", -1.0, "LEFT"),
                                       ("RIGHT_1M", 1.0, "RIGHT")):
            sample, settle = capture_at(rig, carla, center, axis, offset + delta, {
                "experiment": "AVS-0", "phase": "formal_probe",
                "candidate_index": candidate_id, "capture_role": role,
                "role_used_as_scientific_label": False}, trace)
            metrics = frame_metrics(sample, target_id, axis, act0r_config)
            samples[role] = (sample, settle, metrics, offset + delta, direction)
            safe = safe and no_external_boundary(metrics, act0r_config)
        if not safe:
            rejected.append({"offset_m": offset, "reason": "pixel boundary appears before probe endpoint",
                             "frame_ids": [int(row[0]["frame_id"]) for row in samples.values()]})
            continue
        start_id = f"candidate_{candidate_id}_start_{len(starts):02d}"
        frame_ids, entries = {}, []
        for role in ("START", "LEFT_1M", "RIGHT_1M"):
            if saved_counter[0] >= int(config["capture"]["maximum_saved_quartets"]):
                raise RuntimeError("AVS-0 formal saved-frame limit reached")
            sample, settle, metrics, signed, direction = samples[role]
            stem = f"{start_id}_{role.lower()}"
            metadata = save_frame(rig, sample, raw_root, stem, candidate_id,
                                  candidate["bbox_id"], target_id, role, start_id,
                                  direction, signed, metrics)
            metadata["settle"] = settle
            (raw_root / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")
            entries.append(metadata); frame_ids[role] = int(sample["frame_id"])
            saved_counter[0] += 1
        starts.append({
            "surface_id": f"candidate_{candidate_id}", "start_id": start_id,
            "frame_ids": frame_ids, "near_direction_gt": "LEFT" if left_cost < right_cost else "RIGHT",
            "left_cost_m_gt_audit_only": left_cost, "right_cost_m_gt_audit_only": right_cost,
            "wrong_action_regret_m": abs(left_cost - right_cost),
            "signed_offset_m_gt_audit_only": offset, "frames": entries,
        })
    return starts, rejected


def run_capture(args, config: dict, act0r_config: dict) -> int:
    raw_root, result_root = resolve(args.raw), resolve(args.results)
    if raw_root.exists() and any(raw_root.rglob("*")):
        raise RuntimeError("AVS-0 raw output is non-empty")
    raw_root.mkdir(parents=True, exist_ok=True); result_root.mkdir(parents=True, exist_ok=True)
    carla = import_carla(args.carla_root)
    client = carla.Client(args.host or config["server"]["host"],
                          int(args.port or config["server"]["port"]))
    client.set_timeout(float(config["server"]["timeout_s"]))
    world = client.get_world()
    if config["map"] not in world.get_map().name:
        raise RuntimeError(f"AVS-0 requires {config['map']}, got {world.get_map().name}")
    first_candidate = config["candidates"][int(config["candidate_order"][0])]
    initial = candidate_pose(first_candidate)
    trace = TraceWriter(raw_root / "timing_trace.jsonl",
                        int(config["resources"]["python_rss_watchdog_bytes"]), args.carla_pid)
    sensor = config["sensor"]
    rig = SynchronousRGBDSeg(world, carla, transform_from_matrix(carla, initial),
                            sensor["width"], sensor["height"], sensor["horizontal_fov_deg"],
                            sensor["fixed_delta_seconds"], trace_hook=trace)
    qualifications, selected, formal_starts = [], [], []
    qualification_saved, formal_saved = [0], [0]
    try:
        warmup = rig.warmup(5, 3, 12)
        for candidate_id in config["candidate_order"]:
            candidate = config["candidates"][int(candidate_id)]
            outcome = qualify_candidate(rig, carla, int(candidate_id), candidate, config,
                                        act0r_config, raw_root, trace, qualification_saved)
            qualifications.append(outcome)
            print(json.dumps({"phase": "AVS0_QUALIFICATION", "candidate": candidate_id,
                              "status": outcome["status"], "reason": outcome["reason"]}), flush=True)
            if outcome["status"] != "PASS":
                continue
            starts, rejected = capture_surface_starts(
                rig, carla, outcome, candidate, config, act0r_config, raw_root, trace, formal_saved)
            outcome["formal_valid_start_count"] = len(starts)
            outcome["formal_rejected_attempts"] = rejected
            if len(starts) != int(config["capture"]["starts_per_new_surface"]):
                outcome["status"] = "FAIL"
                outcome["reason"] = "fewer than eight shared non-tie preboundary starts"
                continue
            selected.append(int(candidate_id)); formal_starts.extend(starts)
            if len(selected) == 2:
                break
        trace.assert_rss()
    finally:
        rig.close()
    resources = trace.summary(); resources.pop("gpu_processes_at_end", None)
    manifest = {
        "schema": "avs0.capture_manifest.v1", "experiment": "AVS-0",
        "status": "CAPTURE_COMPLETE" if len(selected) == 2 else "STOPPED_INSUFFICIENT_SURFACES",
        "candidate_order": config["candidate_order"], "selected_new_candidates": selected,
        "qualifications": qualifications, "formal_starts": formal_starts,
        "qualification_saved_quartets": qualification_saved[0],
        "formal_saved_quartets": formal_saved[0], "warmup": warmup,
        "resources": {**resources,
                      "configured_python_address_space_limit_bytes": config["resources"]["python_address_space_limit_bytes"],
                      "actual_python_address_space_limit_bytes": resource.getrlimit(resource.RLIMIT_AS)[0],
                      "configured_carla_address_space_limit_bytes": config["resources"]["carla_address_space_limit_bytes"],
                      "numeric_threads": cv2.getNumThreads()},
        "constraints": config["constraints"],
    }
    (result_root / "capture_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return postprocess(args, config, act0r_config)


def rgb_descriptor_for_entry(entry: dict, length: int) -> list[float]:
    image = np.asarray(Image.open(resolve(entry["files"]["rgb"]["path"])).convert("RGB"))
    return fixed_length_descriptor(rgb_descriptor(image), length).tolist()


def candidate1_records(config: dict) -> tuple[list[dict], list[dict]]:
    validation = json.loads(resolve("results/cf0/validation.json").read_text())
    manifest = json.loads(resolve("results/cf0/capture_manifest.json").read_text())
    truth = eligible_truth(validation)[:int(config["capture"]["starts_per_new_surface"])]
    entries = {int(row["frame_id"]): row for row in manifest["frames"]}
    starts = {row["start_id"]: row for row in manifest["starts"]}
    records, evidence = [], {}
    for target in truth:
        source = starts[target["start_id"]]
        shared = entries[int(source["shared_start_frame_id"])]
        evidence[int(shared["frame_id"])] = shared
        for direction in ("LEFT", "RIGHT"):
            endpoint = entries[int(source["branch_frame_ids"][direction][1])]
            evidence[int(endpoint["frame_id"])] = endpoint
            records.append({
                "surface_id": "candidate_1", "start_id": f"candidate_1_{target['start_id']}",
                "direction": direction, "relative_distance_m": 1.0,
                "relative_delta_m": 0.5,
                "descriptor": rgb_descriptor_for_entry(endpoint, config["evaluation"]["descriptor_length"]),
                "target": int(target["target"]), "near_direction": target["near_direction"],
                "wrong_action_regret_m": target["wrong_action_regret_m"],
                "current_frame_id_gt_audit_only": int(endpoint["frame_id"]),
                "rgb_path": endpoint["files"]["rgb"]["path"],
                "shared_rgb_path": shared["files"]["rgb"]["path"],
            })
    return records, list(evidence.values())


def new_surface_records(manifest: dict, config: dict) -> tuple[list[dict], list[dict]]:
    records, evidence = [], []
    for start in manifest["formal_starts"]:
        by_role = {row["capture_role"]: row for row in start["frames"]}
        evidence.extend(start["frames"])
        target = int(start["near_direction_gt"] == "RIGHT")
        for direction, role in (("LEFT", "LEFT_1M"), ("RIGHT", "RIGHT_1M")):
            endpoint = by_role[role]
            records.append({
                "surface_id": start["surface_id"], "start_id": start["start_id"],
                "direction": direction, "relative_distance_m": 1.0,
                "relative_delta_m": 0.5,
                "descriptor": rgb_descriptor_for_entry(endpoint, config["evaluation"]["descriptor_length"]),
                "target": target, "near_direction": start["near_direction_gt"],
                "wrong_action_regret_m": start["wrong_action_regret_m"],
                "current_frame_id_gt_audit_only": int(endpoint["frame_id"]),
                "rgb_path": endpoint["files"]["rgb"]["path"],
                "shared_rgb_path": by_role["START"]["files"]["rgb"]["path"],
            })
    return records, evidence


def draw_surface_sheet(surface_id: str, records: list[dict], output: Path) -> None:
    by_start = {}
    for row in records:
        by_start.setdefault(row["start_id"], {})[row["direction"]] = row
    canvas = Image.new("RGB", (960, len(by_start) * 125 + 35), (18, 18, 18))
    draw = ImageDraw.Draw(canvas); draw.text((8, 8), f"AVS-0 {surface_id}", fill=(255, 230, 70))
    for index, (start_id, pair) in enumerate(sorted(by_start.items())):
        y = 35 + index * 125
        left, right = pair["LEFT"], pair["RIGHT"]
        for panel, (path, label) in enumerate(((left["shared_rgb_path"], "START"),
                                               (left["rgb_path"], "LEFT 1.0m"),
                                               (right["rgb_path"], "RIGHT 1.0m"))):
            image = Image.open(resolve(path)).convert("RGB"); image.thumbnail((300, 110))
            x = panel * 320; canvas.paste(image, (x, y))
            draw.text((x + 4, y + 2), label, fill=(255, 230, 70))
        draw.text((805, y + 94), f"GT {left['near_direction']}", fill=(240, 240, 240))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=80, optimize=True, progressive=True)


def draw_policy_chart(summary: dict, output: Path) -> None:
    canvas = Image.new("RGB", (1100, 560), "white"); draw = ImageDraw.Draw(canvas)
    draw.text((25, 20), "AVS-0 fixed probes versus per-start oracle", fill=(0, 0, 0))
    for index, name in enumerate(("FIXED_LEFT", "FIXED_RIGHT", "RANDOM", "ORACLE_PER_START")):
        value = summary["accuracy"][name]; y = 100 + index * 100
        draw.text((25, y), name, fill=(0, 0, 0)); draw.rectangle((270, y, 270 + int(700 * value), y + 40), fill=(55, 120, 210))
        draw.text((985, y + 10), f"{value:.3f}", fill=(0, 0, 0))
    canvas.save(output, quality=88, optimize=True, progressive=True)


def draw_preference_chart(summary: dict, output: Path) -> None:
    counts = summary["unique_best_action_counts"]
    canvas = Image.new("RGB", (900, 480), "white"); draw = ImageDraw.Draw(canvas)
    draw.text((25, 20), "AVS-0 unique optimal probe action distribution", fill=(0, 0, 0))
    maximum = max(counts.values()) or 1
    for index, name in enumerate(("LEFT", "RIGHT", "TIE")):
        value = counts[name]; y = 110 + index * 105
        draw.text((25, y), name, fill=(0, 0, 0)); draw.rectangle((180, y, 180 + int(600 * value / maximum), y + 42), fill=(70, 155, 95))
        draw.text((800, y + 10), str(value), fill=(0, 0, 0))
    canvas.save(output, quality=88, optimize=True, progressive=True)


def compact_qualification(row: dict) -> dict:
    directions = {}
    for direction, value in row.get("directions", {}).items():
        tier_m = value.get("official_tier_m", {})
        tier_v = value.get("tier_v", {})
        directions[direction] = {
            "status": value.get("status"),
            "event_signed_offset_m": value.get("event_signed_offset_m"),
            "boundary_type": value.get("boundary_type"),
            "tier_v": {key: tier_v.get(key) for key in
                       ("status", "frame_count", "pass_count", "minimum_pass_frames", "thresholds")},
            "official_tier_m": {key: tier_m.get(key) for key in
                                ("status", "frame_count", "spread_m", "standard_deviation_m",
                                 "gate_threshold_m", "sensitivity", "absolute_accuracy",
                                 "uses_plane", "uses_bbox", "uses_legacy_boundary")},
            "boundary_action_axis_coordinate_m_gt_audit_only": value.get(
                "boundary_action_axis_coordinate_m"),
            "frame_ids_gt_audit_only": value.get("frame_ids", []),
        }
    return {
        key: row.get(key) for key in (
            "candidate_index", "bbox_id", "configured_target_instance_id_provenance_only",
            "resolved_target_instance_id", "status", "reason", "target_resolution",
            "center_target_coverage", "center_boundary_absent",
            "bilateral_probe_endpoints_preboundary", "reference_frame_ids", "live_frame_count",
            "formal_valid_start_count")
    } | {"directions": directions}


def postprocess(args, config: dict, act0r_config: dict) -> int:
    result_root, assets = resolve(args.results), resolve(args.assets)
    manifest = json.loads((result_root / "capture_manifest.json").read_text())
    override = config_outcome_override_audit(config)
    selected = manifest.get("selected_new_candidates", [])
    if len(selected) < 2:
        gates = {
            "CANDIDATE_ORDER_PRESERVED": {"status": "PASS"},
            "PHYSICAL_BOUNDARY_AND_SAFETY": {"status": "FAIL", "qualified_new_surfaces": len(selected)},
            "ACTIVE_VIEW_SELECTION_HEADROOM": {"status": "FAIL", "reason": "fewer than two new surfaces qualified"},
            "READY_FOR_POLICY_PILOT": {"status": "FAIL"},
            "READY_FOR_JEPA": {"status": "NOT_EVALUATED"},
        }
        validation = {"schema": "avs0.validation.v1", "experiment": "AVS-0",
                      "run_status": "STOPPED_INSUFFICIENT_SURFACES",
                      "candidate_qualification": [compact_qualification(row) for row in
                                                  manifest["qualifications"]], "gates": gates,
                      "constraints": config["constraints"]}
        (result_root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        write_docs(validation, resolve(args.docs))
        return 2
    candidate1, old_evidence = candidate1_records(config)
    new_records, new_evidence = new_surface_records(manifest, config)
    records = candidate1 + new_records
    prediction, folds = surface_leave_one_out(
        records, config["evaluation"]["pca_components"], config["evaluation"]["ridge_alpha"])
    paired = paired_policy_rows(records, prediction)
    summary = policy_summary(paired)
    bootstrap = stratified_start_bootstrap(
        paired, config["evaluation"]["bootstrap_samples"], config["seed"] + 701)
    starts_per_surface = {surface: sum(row["surface_id"] == surface for row in paired)
                          for surface in sorted({row["surface_id"] for row in paired})}
    preregistered = evaluate_preregistered_gates(
        summary, starts_per_surface, bootstrap, config["gates"])
    qualification_evidence = [entry for row in manifest["qualifications"]
                              for entry in row.get("evidence_frames", [])]
    all_evidence = old_evidence + new_evidence + qualification_evidence
    raw_hash = verify_manifest_hashes(all_evidence, PROJECT_ROOT)
    new_pairing = all(sensor_quartet_paired(entry)
                      for entry in new_evidence + qualification_evidence)
    gates = {
        "CANDIDATE_ORDER_PRESERVED": {"status": "PASS", "selected": selected,
                                      "preregistered_order": config["candidate_order"]},
        "PHYSICAL_BOUNDARY_AND_SAFETY": {"status": "PASS", "qualified_new_surfaces": 2},
        "SENSOR_QUADRUPLET_PAIRING": {
            "status": "PASS" if new_pairing else "FAIL",
            "new_quartet_count": len(new_evidence + qualification_evidence),
            "all_sensor_frame_ids_equal": new_pairing,
            "maximum_timestamp_delta_s": 1e-6,
        },
        "RAW_HASH_AUDIT": raw_hash,
        "SURFACE_LEAVE_ONE_OUT": {"status": "PASS", "folds": folds},
        "ACTIVE_VIEW_SELECTION_HEADROOM": preregistered,
        "READY_FOR_POLICY_PILOT": {"status": "CONDITIONAL_PASS" if preregistered["status"] == "PASS" else "FAIL"},
        "READY_FOR_JEPA": {"status": "NOT_EVALUATED"},
    }
    prediction_rows = []
    for row, probability in zip(records, prediction.tolist()):
        prediction_rows.append({
            "surface_id": row["surface_id"], "start_id": row["start_id"],
            "probe_direction": row["direction"], "near_direction_gt": row["near_direction"],
            "right_probability": probability,
            "predicted_direction": "RIGHT" if probability >= 0.5 else "LEFT",
            "correct": int((probability >= 0.5) == bool(row["target"])),
        })
    write_csv(result_root / "predictions.csv", prediction_rows)
    write_csv(result_root / "policy_comparison.csv", paired)
    (result_root / "bootstrap.json").write_text(json.dumps({"schema": "avs0.bootstrap.v1", **bootstrap}, indent=2) + "\n")
    assets.mkdir(parents=True, exist_ok=True)
    for surface in starts_per_surface:
        draw_surface_sheet(surface, [row for row in records if row["surface_id"] == surface],
                           assets / f"{surface}_probe_contact_sheet.jpg")
    draw_policy_chart(summary, assets / "fixed_vs_oracle.jpg")
    draw_preference_chart(summary, assets / "action_preference_distribution.jpg")
    per_surface = {surface: policy_summary([
        row for row in paired if row["surface_id"] == surface])
        for surface in starts_per_surface}
    source_manifest = {
        "schema": "avs0.source_manifest.v1",
        "candidate_1_source": "frozen CF-0 raw",
        "new_surface_source": "bounded AVS-0 synchronous sensor quartets",
        "selected_new_candidates": selected,
        "evidence_frame_count": len(all_evidence),
        "hash_audit": raw_hash,
        "new_sensor_quartet_pairing": gates["SENSOR_QUADRUPLET_PAIRING"],
        "server_raw_path": "results/avs0/raw",
        "server_raw_size_bytes": sum(path.stat().st_size for path in resolve(
            "results/avs0/raw").rglob("*") if path.is_file()),
        "raw_uploaded": False,
    }
    (result_root / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2) + "\n")
    (result_root / "surface_qualification.json").write_text(json.dumps({
        "schema": "avs0.surface_qualification.v1",
        "candidate_order": config["candidate_order"],
        "selected_new_candidates": selected,
        "qualifications": [compact_qualification(row) for row in manifest["qualifications"]],
    }, indent=2) + "\n")
    validation = {
        "schema": "avs0.validation.v1", "experiment": "AVS-0", "run_status": "COMPLETE",
        "candidate_qualification": [compact_qualification(row) for row in
                                    manifest["qualifications"]],
        "selected_new_candidates": selected, "valid_starts_per_surface": starts_per_surface,
        "frozen_model": {"name": "E1 static endpoint RGB plus action",
                         "descriptor_dimension": config["evaluation"]["descriptor_length"],
                         "pca_max": config["evaluation"]["pca_components"],
                         "ridge_alpha": config["evaluation"]["ridge_alpha"],
                         "preprocessing": "training surfaces only"},
        "policy_summary": summary, "surface_policy_summary": per_surface,
        "bootstrap_95_ci": bootstrap,
        "preregistered_thresholds": config["gates"], "gates": gates,
        "resources": {**manifest["resources"], "model_artifacts_saved": False,
                      "raw_path": "results/avs0/raw",
                      "raw_size_bytes": source_manifest["server_raw_size_bytes"]},
        "constraints": config["constraints"], "config_outcome_override_audit": override,
    }
    (result_root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    write_docs(validation, resolve(args.docs))
    print(json.dumps({"gates": {name: value["status"] for name, value in gates.items()},
                      "policy_summary": summary, "bootstrap": bootstrap}, indent=2))
    return 0 if preregistered["status"] == "PASS" else 1


def write_docs(validation: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if validation["run_status"] != "COMPLETE":
        lines = ["# AVS-0 Active View Selection Audit", "",
                 "AVS-0 stopped before model evaluation because fewer than two new surfaces",
                 "passed the preregistered physical-boundary and probe-safety qualification.",
                 "Candidate order was 7, 8, 10, 19. No policy or JEPA was trained.", "",
                 "## Qualification", ""]
        for row in validation["candidate_qualification"]:
            lines.append(f"- candidate {row['candidate_index']}: {row['status']} - {row['reason']}")
        output.write_text("\n".join(lines) + "\n")
        return
    summary = validation["policy_summary"]
    lines = ["# AVS-0 Active View Selection Audit", "",
             "AVS-0 is a cross-surface feasibility audit, not a trained active policy and not",
             "a JEPA experiment. New surfaces were qualified in fixed order before endpoint",
             "RGB results were computed. E1 uses only the 64-D endpoint RGB descriptor and",
             "probe action. PCA and ridge are fitted on training surfaces only.", "",
             "## Results", "",
             "| Policy | Accuracy |", "| --- | ---: |"]
    for name, value in summary["accuracy"].items():
        lines.append(f"| {name} | {value:.6f} |")
    lines.extend(["", f"- Best fixed: {summary['best_fixed_policy']} ({summary['best_fixed_accuracy']:.6f})",
                  f"- Oracle minus best fixed: {summary['oracle_minus_best_fixed_accuracy']:.6f}",
                  f"- Bootstrap 95% CI: [{validation['bootstrap_95_ci']['lower']:.6f}, {validation['bootstrap_95_ci']['upper']:.6f}]",
                  f"- ACTIVE_VIEW_SELECTION_HEADROOM: {validation['gates']['ACTIVE_VIEW_SELECTION_HEADROOM']['status']}",
                  f"- SENSOR_QUADRUPLET_PAIRING: {validation['gates']['SENSOR_QUADRUPLET_PAIRING']['status']}",
                  f"- READY_FOR_POLICY_PILOT: {validation['gates']['READY_FOR_POLICY_PILOT']['status']}",
                  "- READY_FOR_JEPA: NOT_EVALUATED", ""])
    lines.extend(["## Per-Surface Policy Accuracy", "",
                  "| Surface | Fixed LEFT | Fixed RIGHT | Oracle | Oracle-best fixed |",
                  "| --- | ---: | ---: | ---: | ---: |"])
    for surface, value in validation["surface_policy_summary"].items():
        accuracy = value["accuracy"]
        lines.append(
            f"| {surface} | {accuracy['FIXED_LEFT']:.6f} | "
            f"{accuracy['FIXED_RIGHT']:.6f} | {accuracy['ORACLE_PER_START']:.6f} | "
            f"{value['oracle_minus_best_fixed_accuracy']:.6f} |")
    lines.extend(["", "## Qualification", "",
                  "Candidates were evaluated in the fixed order 7, 8, 10, 19. Candidate 7",
                  "and candidate 8 were the first two surfaces to pass bilateral physical",
                  "termination, Tier V, plane-free Tier M and 1.0 m endpoint safety gates.",
                  "Instance IDs were resolved from the current-session CENTER semantic/instance",
                  "pixels; historical IDs were retained only as provenance.", "",
                  "The historical candidate-1 fixed-RIGHT value of 1.0 came from the earlier",
                  "13-start within-surface grouped evaluation. AVS-0 holds candidate 1 out and",
                  "trains E1 only on candidates 7 and 8, using eight frozen candidate-1 starts;",
                  "its candidate-1 fixed-RIGHT accuracy is therefore 0.125 and is not the same",
                  "estimand. This distinction is why the old 1.0 is not reused as an AVS-0 score.", "",
                  "## Public Evidence", ""])
    for surface in validation["valid_starts_per_surface"]:
        lines.append(f"![{surface} probes](assets/avs0/{surface}_probe_contact_sheet.jpg)")
    lines.extend(["", "![Fixed policies and oracle](assets/avs0/fixed_vs_oracle.jpg)", "",
                  "![Action preference distribution](assets/avs0/action_preference_distribution.jpg)"])
    output.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("capture", "postprocess"))
    parser.add_argument("--config", default="configs/experiments/avs0.yaml")
    parser.add_argument("--act0r-config", default="configs/experiments/act0r.yaml")
    parser.add_argument("--carla-root")
    parser.add_argument("--host"); parser.add_argument("--port", type=int)
    parser.add_argument("--carla-pid", type=int)
    parser.add_argument("--results", default="results/avs0")
    parser.add_argument("--raw", default="results/avs0/raw")
    parser.add_argument("--assets", default="docs/assets/avs0")
    parser.add_argument("--docs", default="docs/AVS0_ACTIVE_VIEW_SELECTION_AUDIT.md")
    args = parser.parse_args()
    config = yaml.safe_load(resolve(args.config).read_text())
    act0r_config = yaml.safe_load(resolve(args.act0r_config).read_text())
    if config_outcome_override_audit(config)["status"] != "PASS":
        raise RuntimeError("AVS-0 configuration contains an outcome override")
    cv2.setNumThreads(1)
    return run_capture(args, config, act0r_config) if args.mode == "capture" else postprocess(
        args, config, act0r_config)


if __name__ == "__main__":
    raise SystemExit(main())
