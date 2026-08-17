#!/usr/bin/env python3
"""Offline physical-boundary audit of the persisted ACT-0R1 sensor quartets."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from boundary_sweep.act0r import (
    action_axis_from_transforms,
    boundary_bilateral_samples,
    boundary_type_consensus,
    classify_boundary_pixels,
    config_outcome_override_audit,
    contour_action_axis_coordinate,
    contour_span_metrics,
    official_tier_m,
    pose_repeatability,
    select_repeated_pose_group,
    sha256_file,
    tier_v_from_pixel_frames,
    verify_manifest_hashes,
)
from boundary_sweep.segmentation import (
    decode_instance_channels,
    largest_connected_component_ratio,
    semantic_instance_consistency,
)


def _open_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(PROJECT_ROOT / path).convert("RGB"))


def _bbox(mask: np.ndarray) -> dict:
    ys, xs = np.where(mask)
    if not len(ys):
        return {"x_min": None, "y_min": None, "x_max": None, "y_max": None,
                "width_px": 0, "height_px": 0}
    return {
        "x_min": int(xs.min()), "y_min": int(ys.min()),
        "x_max": int(xs.max()), "y_max": int(ys.max()),
        "width_px": int(xs.max() - xs.min() + 1),
        "height_px": int(ys.max() - ys.min() + 1),
    }


def _raw_audit(frames: list[dict], raw_root: Path) -> tuple[dict, list[dict]]:
    manifest = verify_manifest_hashes(frames, PROJECT_ROOT)
    expected_paths = set()
    size_mismatches = []
    rows = []
    metadata_mismatches = []
    for frame in frames:
        for name, item in frame["files"].items():
            path = PROJECT_ROOT / item["path"]
            expected_paths.add(path.resolve())
            actual_size = path.stat().st_size if path.exists() else None
            actual_hash = sha256_file(path) if path.exists() else None
            if actual_size != int(item["size_bytes"]):
                size_mismatches.append({
                    "frame_id": frame["frame_id"], "name": name,
                    "expected": int(item["size_bytes"]), "actual": actual_size,
                })
            rows.append({
                "frame_id": frame["frame_id"], "file_type": name,
                "path": item["path"], "size_bytes": actual_size,
                "expected_sha256": item["sha256"],
                "actual_sha256": actual_hash,
                "status": "PASS" if actual_hash == item["sha256"] else "FAIL",
            })
        metadata_path = PROJECT_ROOT / frame["metadata_path"]
        expected_paths.add(metadata_path.resolve())
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else None
        if metadata != frame:
            metadata_mismatches.append({
                "frame_id": frame["frame_id"], "path": frame["metadata_path"]})
        rows.append({
            "frame_id": frame["frame_id"], "file_type": "metadata",
            "path": frame["metadata_path"],
            "size_bytes": metadata_path.stat().st_size if metadata_path.exists() else None,
            "expected_sha256": "",
            "actual_sha256": sha256_file(metadata_path) if metadata_path.exists() else None,
            "status": "PASS" if metadata == frame else "FAIL",
        })
    actual_paths = {path.resolve() for path in raw_root.glob("*") if path.is_file()}
    unexpected = sorted(str(path.relative_to(PROJECT_ROOT))
                        for path in actual_paths - expected_paths)
    absent = sorted(str(path.relative_to(PROJECT_ROOT))
                    for path in expected_paths - actual_paths)
    passed = (
        manifest["status"] == "PASS" and not size_mismatches and
        not metadata_mismatches and not unexpected and not absent and
        len(frames) == 8 and len(rows) == 56
    )
    result = {
        "status": "PASS" if passed else "FAIL",
        "frame_count": len(frames),
        "checked_file_count": len(rows),
        "manifest_hashed_file_count": manifest["checked_file_count"],
        "metadata_file_count": sum(row["file_type"] == "metadata" for row in rows),
        "raw_file_count": len(actual_paths),
        "raw_file_bytes": int(sum(path.stat().st_size for path in actual_paths)),
        "missing": manifest["missing"] + absent,
        "hash_mismatches": manifest["mismatches"],
        "size_mismatches": size_mismatches,
        "metadata_payload_mismatches": metadata_mismatches,
        "unexpected_files": unexpected,
    }
    return result, rows


def _pairing_audit(frames: list[dict]) -> dict:
    failures = []
    for frame in frames:
        sensor_frames = frame.get("sensor_frames", {})
        timestamps = frame.get("sensor_timestamps", {})
        lengths = frame.get("raw_byte_lengths", {})
        expected_length = (
            int(frame["sensor_config"]["width"]) *
            int(frame["sensor_config"]["height"]) * 4
        )
        valid = (
            set(sensor_frames) == {"rgb", "depth", "semantic", "instance"} and
            len(set(sensor_frames.values())) == 1 and
            next(iter(sensor_frames.values())) == frame["frame_id"] and
            set(timestamps) == set(sensor_frames) and
            max(timestamps.values()) - min(timestamps.values()) <= 1e-6 and
            all(int(value) == expected_length for value in lengths.values())
        )
        if not valid:
            failures.append(frame["frame_id"])
    return {
        "status": "PASS" if len(frames) == 8 and not failures else "FAIL",
        "frame_count": len(frames),
        "paired_frame_count": len(frames) - len(failures),
        "failed_frame_ids": failures,
        "timestamp_tolerance_s": 1e-6,
    }


def _colorize(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.uint32)
    return np.stack([
        ((source * 37 + 41) % 223 + 24).astype(np.uint8),
        ((source * 73 + 19) % 223 + 24).astype(np.uint8),
        ((source * 109 + 7) % 223 + 24).astype(np.uint8),
    ], axis=-1)


def _overlay(rgb: np.ndarray, mask: np.ndarray, contour: np.ndarray,
             samples: dict) -> Image.Image:
    image = np.asarray(rgb, dtype=np.float32).copy()
    image[mask] = image[mask] * 0.55 + np.array([30, 220, 70]) * 0.45
    image[contour] = np.array([255, 30, 30])
    output = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(output)
    step = max(1, len(samples["y"]) // 28)
    for index in range(0, len(samples["y"]), step):
        y = int(samples["y"][index])
        tx = int(samples["target_x"][index])
        ex = int(samples["external_x"][index])
        draw.ellipse((tx - 2, y - 2, tx + 2, y + 2), fill=(0, 255, 255))
        draw.ellipse((ex - 2, y - 2, ex + 2, y + 2), fill=(255, 230, 0))
    return output


def _panel(image: Image.Image, title: str) -> Image.Image:
    resized = image.convert("RGB").resize((320, 240))
    panel = Image.new("RGB", (320, 268), (18, 18, 18))
    panel.paste(resized, (0, 28))
    ImageDraw.Draw(panel).text((6, 7), title, fill=(255, 235, 90))
    return panel


def _all_roles_sheet(evidence: list[dict], output: Path) -> None:
    canvas = Image.new("RGB", (1280, 536), (18, 18, 18))
    for index, row in enumerate(evidence):
        panel = _panel(
            _overlay(row["rgb"], row["mask"], row["contour"], row["samples"]),
            (f"plan={row['plan_role']} frame={row['frame_id']} "
             f"cov={row['metrics']['target_coverage']:.3f} "
             f"span={row['metrics']['span_over_target_bbox_height']:.3f}"),
        )
        x, y = (index % 4) * 320, (index // 4) * 268
        canvas.paste(panel, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=84, optimize=True, progressive=True)


def _depth_evidence(row: dict) -> Image.Image:
    depth = row["depth"]
    valid = np.isfinite(depth) & (depth > 0.05) & (depth < 999.0)
    low, high = np.percentile(depth[valid], [2, 98]) if np.any(valid) else (0.0, 1.0)
    normalized = np.clip((depth - low) / max(float(high - low), 1e-6), 0, 1)
    image = np.stack([
        (255 * normalized).astype(np.uint8),
        (255 * (1 - np.abs(normalized - 0.5) * 2)).astype(np.uint8),
        (255 * (1 - normalized)).astype(np.uint8),
    ], axis=-1)
    image[~valid] = 0
    output = Image.fromarray(image)
    draw = ImageDraw.Draw(output)
    samples = row["samples"]
    step = max(1, len(samples["y"]) // 32)
    for index in range(0, len(samples["y"]), step):
        y = int(samples["y"][index])
        tx = int(samples["target_x"][index])
        ex = int(samples["external_x"][index])
        color = (20, 180, 255) if bool(samples["valid_depth"][index]) else (150, 150, 150)
        draw.line((tx, y, ex, y), fill=color, width=2)
    return output


def _straddle_evidence_sheet(row: dict, output: Path) -> None:
    classification = row["metrics"]["boundary_classification"]
    semantic = _colorize(row["semantic"])
    instance = _colorize(row["instance"])
    semantic[row["contour"]] = [255, 30, 30]
    instance[row["contour"]] = [255, 30, 30]
    panels = [
        _panel(_overlay(row["rgb"], row["mask"], row["contour"], row["samples"]),
               "RGB: mask/contour + target/external probes"),
        _panel(_depth_evidence(row),
               f"depth ext-target median={classification['external_minus_target_depth_median_m']:.3f}m"),
        _panel(Image.fromarray(semantic),
               f"semantic ext Building={classification['external_building_fraction']:.3f}"),
        _panel(Image.fromarray(instance),
               f"instance ext non-target={classification['external_non_target_instance_fraction']:.3f}"),
    ]
    canvas = Image.new("RGB", (1280, 308), (18, 18, 18))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * 320, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8),
              (f"frame={row['frame_id']} computed={classification['boundary_type']} "
               f"valid_pairs={classification['valid_depth_pair_count']}"),
              fill=(255, 235, 90))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=84, optimize=True, progressive=True)


def _consensus_sheet(rows: list[dict], output: Path) -> None:
    canvas = Image.new("RGB", (1280, 536), (18, 18, 18))
    for index, row in enumerate(rows):
        classification = row["metrics"]["boundary_classification"]
        panel = _panel(
            _overlay(row["rgb"], row["mask"], row["contour"], row["samples"]),
            f"frame={row['frame_id']} computed={classification['boundary_type']}",
        )
        canvas.paste(panel.resize((640, 268)),
                     ((index % 2) * 640, (index // 2) * 268))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=84, optimize=True, progressive=True)


def _write_frame_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "frame_id", "plan_role", "target_pixels", "target_coverage",
        "bbox_x_min", "bbox_y_min", "bbox_x_max", "bbox_y_max",
        "bbox_width_px", "bbox_height_px", "largest_component_ratio",
        "target_semantic_building_fraction", "semantic_camera_agreement",
        "contour_pixel_count", "span_over_image_height",
        "span_over_target_bbox_height", "target_side_fraction",
        "external_side_fraction", "boundary_type", "bilateral_sample_count",
        "valid_depth_pair_count", "external_depth_delta_median_m",
        "external_building_fraction", "external_non_target_instance_fraction",
        "action_axis_median_m", "action_axis_mad_m",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            bbox = row["mask_bbox"]
            classification = row["boundary_classification"]
            tier_m = row["tier_m_frame"]
            writer.writerow({
                **{key: row.get(key) for key in fields},
                "bbox_x_min": bbox["x_min"], "bbox_y_min": bbox["y_min"],
                "bbox_x_max": bbox["x_max"], "bbox_y_max": bbox["y_max"],
                "bbox_width_px": bbox["width_px"], "bbox_height_px": bbox["height_px"],
                "boundary_type": classification["boundary_type"],
                "bilateral_sample_count": classification.get("bilateral_sample_count"),
                "valid_depth_pair_count": classification.get("valid_depth_pair_count"),
                "external_depth_delta_median_m": classification.get(
                    "external_minus_target_depth_median_m"),
                "external_building_fraction": classification.get(
                    "external_building_fraction"),
                "external_non_target_instance_fraction": classification.get(
                    "external_non_target_instance_fraction"),
                "action_axis_median_m": tier_m["action_axis_median_m"],
                "action_axis_mad_m": tier_m["action_axis_mad_m"],
            })


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", default="results/act0r1/validation.json")
    parser.add_argument("--raw-root", default="results/act0r1/raw")
    parser.add_argument("--config", default="configs/experiments/act0r.yaml")
    parser.add_argument("--cap0-config", default="configs/experiments/cap0.yaml")
    parser.add_argument("--checkpoint", default="results/act0r/search_plan_checkpoint.json")
    parser.add_argument("--output", default="results/act0r1/offline_boundary_audit.json")
    parser.add_argument("--frame-csv", default="results/act0r1/offline_frame_metrics.csv")
    parser.add_argument("--hash-csv", default="results/act0r1/offline_raw_hash_audit.csv")
    parser.add_argument("--assets", default="docs/assets/act0r1")
    args = parser.parse_args(argv)

    validation_path = PROJECT_ROOT / args.validation
    validation = json.loads(validation_path.read_text())
    frames = validation["frames"]
    config = yaml.safe_load((PROJECT_ROOT / args.config).read_text())
    cap0_config = yaml.safe_load((PROJECT_ROOT / args.cap0_config).read_text())
    offline = config["offline_audit"]
    raw_root = PROJECT_ROOT / args.raw_root
    checkpoint_path = PROJECT_ROOT / args.checkpoint

    raw_audit, hash_rows = _raw_audit(frames, raw_root)
    pairing = _pairing_audit(frames)
    checkpoint_expected = cap0_config["checkpoint"]["sha256"]
    checkpoint_actual = sha256_file(checkpoint_path)
    checkpoint_audit = {
        "status": "PASS" if checkpoint_actual == checkpoint_expected else "FAIL",
        "path": args.checkpoint,
        "expected_sha256": checkpoint_expected,
        "actual_sha256": checkpoint_actual,
    }
    config_audit = config_outcome_override_audit(config)
    transforms = [np.asarray(frame["T_world_camera"], dtype=float) for frame in frames]
    action_axis = action_axis_from_transforms(transforms)

    frame_metrics = []
    evidence = []
    target_valid_rows = []
    for frame in frames:
        rgb = _open_rgb(frame["files"]["rgb"]["path"])
        semantic_rgb = _open_rgb(frame["files"]["semantic"]["path"])
        instance_rgb = _open_rgb(frame["files"]["instance"]["path"])
        semantic = decode_instance_channels(semantic_rgb)["semantic_tag"]
        instance_channels = decode_instance_channels(instance_rgb)
        instance = instance_channels["instance_id_16bit"]
        instance_semantic = instance_channels["semantic_tag"]
        depth = np.load(PROJECT_ROOT / frame["files"]["depth_m"]["path"])
        mask = instance == 39220
        target_pixels = int(mask.sum())
        semantic_fraction = float(np.mean(semantic[mask] == 3)) if target_pixels else 0.0
        instance_semantic_fraction = float(np.mean(
            instance_semantic[mask] == 3)) if target_pixels else 0.0
        consistency = semantic_instance_consistency(semantic_rgb, instance_rgb)
        target_valid = (
            target_pixels > 0 and
            semantic_fraction >= float(offline["target_semantic_min_fraction"]) and
            instance_semantic_fraction >= float(offline["target_semantic_min_fraction"])
        )
        target_valid_rows.append(target_valid)
        span = contour_span_metrics(mask, "LEFT")
        contour = span.pop("contour")
        samples = boundary_bilateral_samples(
            mask, contour, depth, semantic, instance, "LEFT",
            config["boundary_classification"]["side_probe_offset_px"])
        classification = classify_boundary_pixels(
            mask, contour, depth, semantic, instance, 39220, "LEFT",
            config["boundary_classification"])
        tier_m_frame = contour_action_axis_coordinate(
            contour, depth, np.asarray(frame["K"], dtype=float),
            np.asarray(frame["T_world_camera"], dtype=float), action_axis["axis"])
        tier_m_frame["world_points_sample"] = tier_m_frame["world_points_sample"][:24]
        metric = {
            "frame_id": int(frame["frame_id"]),
            "plan_role": frame["capture_role"],
            "plan_role_used_as_scientific_label": False,
            "target_instance_id": 39220,
            "target_pixels": target_pixels,
            "target_coverage": float(mask.mean()),
            "mask_bbox": _bbox(mask),
            "largest_component_ratio": largest_connected_component_ratio(mask),
            "target_semantic_building_fraction": semantic_fraction,
            "target_instance_camera_building_fraction": instance_semantic_fraction,
            "semantic_camera_agreement": consistency["agreement"],
            "target_mask_pixel_valid": bool(target_valid),
            **span,
            "boundary_classification": classification,
            "tier_m_frame": tier_m_frame,
        }
        frame_metrics.append(metric)
        evidence.append({
            "frame_id": int(frame["frame_id"]),
            "plan_role": frame["capture_role"],
            "rgb": rgb, "semantic": semantic, "instance": instance,
            "depth": depth, "mask": mask, "contour": contour,
            "samples": samples, "metrics": metric,
        })

    repeated_group = select_repeated_pose_group(
        transforms, int(offline["repeated_pose_min_frames"]),
        config["repeatability"]["max_position_error_m"],
        config["repeatability"]["max_rotation_error_deg"])
    repeated_indices = repeated_group["indices"]
    repeated_group["frame_ids"] = [frames[index]["frame_id"] for index in repeated_indices]
    repeated_group["plan_roles_for_provenance_only"] = [
        frames[index]["capture_role"] for index in repeated_indices]
    repeated_metrics = [frame_metrics[index] for index in repeated_indices]
    repeated_evidence = [evidence[index] for index in repeated_indices]
    consensus = boundary_type_consensus(
        [row["boundary_classification"] for row in repeated_metrics],
        int(offline["boundary_consensus_min_frames"]))
    tier_v = tier_v_from_pixel_frames(
        repeated_metrics, config["contour"],
        int(offline["tier_v_min_pass_frames"]))
    tier_m = official_tier_m(
        [row["tier_m_frame"] for row in repeated_metrics],
        config["tier_m"]["sensitivity_thresholds_m"],
        config["tier_m"]["gate_spread_m"])
    same_pose = pose_repeatability(
        [transforms[index] for index in repeated_indices],
        config["repeatability"]["max_position_error_m"],
        config["repeatability"]["max_rotation_error_deg"])

    target_gate = {
        "status": "PASS" if len(target_valid_rows) == 8 and all(target_valid_rows) else "FAIL",
        "valid_frame_count": int(sum(target_valid_rows)),
        "frame_count": len(target_valid_rows),
        "target_instance_id": 39220,
        "building_fraction_threshold": float(offline["target_semantic_min_fraction"]),
    }
    role_independence = {
        "status": "PASS" if (
            repeated_group["uses_role_labels"] is False and
            tier_v["uses_role_labels"] is False) else "FAIL",
        "capture_role_usage": "provenance and display only",
        "frozen_group_selection": "T_world_camera clustering",
        "classification_inputs": "semantic pixels, instance pixels and z-depth",
    }
    resolved_gate = {
        "status": consensus["status"],
        "boundary_type": consensus["boundary_type"],
        "consensus_count": consensus["consensus_count"],
        "frame_count": consensus["frame_count"],
    }
    raw_gate = dict(raw_audit)
    raw_gate["checkpoint_status"] = checkpoint_audit["status"]
    raw_gate["status"] = (
        "PASS" if raw_audit["status"] == "PASS" and
        checkpoint_audit["status"] == "PASS" else "FAIL")
    prerequisites = (
        raw_gate["status"] == "PASS" and pairing["status"] == "PASS" and
        target_gate["status"] == "PASS" and role_independence["status"] == "PASS" and
        consensus["boundary_type"] == "PHYSICAL_TERMINATION" and
        tier_v["status"] == "PASS" and tier_m["status"] == "PASS" and
        same_pose["status"] == "PASS"
    )
    gates = {
        "RAW_HASH_AUDIT": raw_gate,
        "SENSOR_PAIRING": pairing,
        "TARGET_MASK_PIXEL_VALID": target_gate,
        "ROLE_LABEL_INDEPENDENCE": role_independence,
        "LEFT_BOUNDARY_TYPE_RESOLVED": resolved_gate,
        "TIER_V": tier_v,
        "OFFICIAL_TIER_M": tier_m,
        "SAME_POSE_CONFIRMATION": same_pose,
        "EXTERNAL_VISUAL_REVIEW": {"status": "PENDING"},
        "READY_FOR_CANDIDATE1_RIGHT": {
            "status": "CONDITIONAL_PASS" if prerequisites else "FAIL",
            "required_boundary_type": "PHYSICAL_TERMINATION",
        },
        "READY_FOR_COUNTERFACTUAL_ROLLOUT": {"status": "NOT_EVALUATED"},
        "READY_FOR_JEPA": {"status": "NOT_EVALUATED"},
    }

    cap0_probe = json.loads((PROJECT_ROOT / "results/cap0/as2_probe.json").read_text())
    historical_act0r = json.loads((PROJECT_ROOT / "results/act0r/validation.json").read_text())
    result = {
        "schema": "act0r1.offline_boundary_audit.v1",
        "source": {
            "validation": args.validation,
            "validation_sha256": sha256_file(validation_path),
            "raw_root": args.raw_root,
            "raw_root_file_count": raw_audit["raw_file_count"],
            "raw_root_file_bytes": raw_audit["raw_file_bytes"],
            "checkpoint": checkpoint_audit,
        },
        "config": args.config,
        "config_outcome_override_audit": config_audit,
        "thresholds": {
            "contour": config["contour"],
            "boundary_classification": config["boundary_classification"],
            "tier_m": config["tier_m"],
            "repeatability": config["repeatability"],
            "offline_audit": offline,
        },
        "target_instance_id": 39220,
        "direction": "LEFT",
        "role_label_policy": {
            "capture_roles_are_plan_pose_names_only": True,
            "capture_roles_used_as_ground_truth": False,
        },
        "actual_action_axis": action_axis,
        "repeated_pose_group": repeated_group,
        "boundary_consensus": consensus,
        "frame_metrics": frame_metrics,
        "fault_attribution": {
            "TWO_GIB_ADDRESS_SPACE_FAILURE": {
                "status": "CONFIRMED",
                "evidence": {
                    "probe_status": cap0_probe.get("status"),
                    "outer_exit_code": cap0_probe.get("outer_exit_code"),
                    "actual_outer_address_space_limit_bytes": cap0_probe.get(
                        "actual_outer_address_space_limit_bytes"),
                    "four_gib_reference": cap0_probe.get("four_gib_reference"),
                },
            },
            "HISTORICAL_TRIANGLE_ARTIFACT_ROOT_CAUSE": {
                "status": "LIKELY_BUT_NOT_UNIQUELY_PROVEN",
                "evidence": {
                    "historical_act0r_rgb_integrity": historical_act0r.get(
                        "gates", {}).get("SENSOR_QUADRUPLET_PAIRING", {}).get(
                        "available_rgb_visual_integrity"),
                    "fresh_act0r1_rgb_integrity": validation.get(
                        "gates", {}).get("RGB_VISUAL_INTEGRITY", {}).get("status"),
                    "controlled_single_variable_reproduction": False,
                },
            },
        },
        "gates": gates,
        "constraints": {
            "carla_started": False,
            "new_capture_run": False,
            "rollout_run": False,
            "jepa_training_run": False,
            "legacy_plane_used": False,
            "legacy_bbox_used": False,
            "manual_boundary_used": False,
            "historical_results_modified": False,
        },
    }

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    _write_frame_csv(frame_metrics, PROJECT_ROOT / args.frame_csv)
    with (PROJECT_ROOT / args.hash_csv).open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(hash_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(hash_rows)

    assets = PROJECT_ROOT / args.assets
    _all_roles_sheet(evidence, assets / "offline_all_roles_mask_contour.jpg")
    for row in repeated_evidence:
        _straddle_evidence_sheet(
            row, assets / f"offline_straddle_{row['frame_id']}_evidence.jpg")
    _consensus_sheet(
        repeated_evidence, assets / "offline_straddle_consensus.jpg")
    print(json.dumps({
        "boundary_consensus": consensus,
        "tier_v": tier_v["status"],
        "official_tier_m": {
            "status": tier_m["status"], "spread_m": tier_m["spread_m"]},
        "same_pose": same_pose,
        "ready_for_candidate1_right": gates["READY_FOR_CANDIDATE1_RIGHT"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
