"""ACT-0R pixel-derived boundary evidence and plane-free repeatability."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .act0 import directional_instance_contour
from .geometry import camera_to_world, pixel_depth_to_camera_point


FORBIDDEN_OUTCOME_CONFIG_KEYS = {
    "existing_evidence_interpretation",
    "tier_v_status",
    "candidate_classification",
    "classification",
    "expected_boundary_type",
    "force_pass",
    "override",
}

BOUNDARY_TYPES = {
    "PHYSICAL_TERMINATION",
    "FOREGROUND_OCCLUSION",
    "INTERNAL_INSTANCE_SEAM",
    "UNRESOLVED",
}


def config_outcome_override_audit(config: Mapping) -> dict:
    """Reject configuration keys that can directly prescribe scientific outcomes."""
    found = []

    def visit(value, path=""):
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if str(key).lower() in FORBIDDEN_OUTCOME_CONFIG_KEYS:
                    found.append(child_path)
                visit(child, child_path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(config)
    return {"status": "PASS" if not found else "FAIL", "forbidden_paths": found,
            "forbidden_keys": sorted(FORBIDDEN_OUTCOME_CONFIG_KEYS)}


def contour_span_metrics(target_mask: np.ndarray, direction: str) -> dict:
    """Measure directional exterior contour span against image and target bbox."""
    target = np.asarray(target_mask, dtype=bool)
    evidence = directional_instance_contour(target, direction)
    contour = evidence.pop("contour")
    target_y, target_x = np.where(target)
    contour_y, contour_x = np.where(contour)
    bbox_height = int(target_y.max() - target_y.min() + 1) if len(target_y) else 0
    bbox_width = int(target_x.max() - target_x.min() + 1) if len(target_x) else 0
    span = int(len(np.unique(contour_y))) if len(contour_y) else 0
    evidence.update({
        "contour": contour,
        "contour_pixel_count": int(len(contour_y)),
        "contour_span_px": span,
        "target_bbox_height_px": bbox_height,
        "target_bbox_width_px": bbox_width,
        "span_over_image_height": float(span / max(target.shape[0], 1)),
        "span_over_target_bbox_height": float(span / max(bbox_height, 1)),
    })
    return evidence


def _side_samples(contour: np.ndarray, target_mask: np.ndarray, direction: str,
                  offset_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.where(np.asarray(contour, dtype=bool))
    if direction == "LEFT":
        target_x, external_x = xs + offset_px, xs - offset_px
    elif direction == "RIGHT":
        target_x, external_x = xs - offset_px, xs + offset_px
    else:
        raise ValueError("direction must be LEFT or RIGHT")
    valid = ((target_x >= 0) & (target_x < target_mask.shape[1]) &
             (external_x >= 0) & (external_x < target_mask.shape[1]))
    return ys[valid], target_x[valid], external_x[valid]


def classify_boundary_pixels(target_mask: np.ndarray, contour: np.ndarray,
                             depth_m: np.ndarray, semantic: np.ndarray,
                             instance: np.ndarray, target_instance_id: int,
                             direction: str, thresholds: Mapping) -> dict:
    """Classify one observed target/non-target boundary from bilateral pixels."""
    target = np.asarray(target_mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=float)
    semantic = np.asarray(semantic)
    instance = np.asarray(instance)
    offset = int(thresholds["side_probe_offset_px"])
    ys, target_x, external_x = _side_samples(contour, target, direction, offset)
    if not len(ys):
        return {"boundary_type": "UNRESOLVED", "reason": "no bilateral samples",
                "bilateral_sample_count": 0}
    target_membership = target[ys, target_x]
    external_membership = ~target[ys, external_x]
    bilateral = target_membership & external_membership
    ys, target_x, external_x = ys[bilateral], target_x[bilateral], external_x[bilateral]
    target_depth, external_depth = depth[ys, target_x], depth[ys, external_x]
    valid_depth = (np.isfinite(target_depth) & np.isfinite(external_depth) &
                   (target_depth > 0.05) & (external_depth > 0.05) &
                   (target_depth < 999.0) & (external_depth < 999.0))
    depth_delta = external_depth[valid_depth] - target_depth[valid_depth]
    ext_semantic = semantic[ys, external_x]
    ext_instance = instance[ys, external_x]
    count = int(len(ys))
    valid_count = int(valid_depth.sum())
    building_tag = int(thresholds["building_semantic_tag"])
    external_building_fraction = float(np.mean(ext_semantic == building_tag)) if count else 0.0
    external_non_target_fraction = float(np.mean(ext_instance != int(target_instance_id))) if count else 0.0
    median_delta = float(np.median(depth_delta)) if valid_count else None
    closer_fraction = float(np.mean(depth_delta < -float(thresholds["depth_margin_m"]))) if valid_count else None
    same_depth_fraction = float(np.mean(np.abs(depth_delta) <= float(thresholds["same_depth_tolerance_m"]))) if valid_count else None
    metrics = {
        "bilateral_sample_count": count,
        "valid_depth_pair_count": valid_count,
        "external_building_fraction": external_building_fraction,
        "external_non_target_instance_fraction": external_non_target_fraction,
        "external_minus_target_depth_median_m": median_delta,
        "external_closer_fraction": closer_fraction,
        "same_depth_fraction": same_depth_fraction,
    }
    if count < int(thresholds["min_bilateral_samples"]) or valid_count < int(thresholds["min_depth_pairs"]):
        boundary_type, reason = "UNRESOLVED", "insufficient bilateral depth evidence"
    elif median_delta is not None and (median_delta <= -float(thresholds["depth_margin_m"]) or
                                       closer_fraction >= float(thresholds["occlusion_closer_fraction"])):
        boundary_type, reason = "FOREGROUND_OCCLUSION", "external side is measurably closer"
    elif (external_building_fraction >= float(thresholds["semantic_majority_fraction"]) and
          external_non_target_fraction >= float(thresholds["semantic_majority_fraction"]) and
          same_depth_fraction >= float(thresholds["same_depth_majority_fraction"])):
        boundary_type, reason = "INTERNAL_INSTANCE_SEAM", "same-depth Building pixels change instance ID"
    elif (median_delta is not None and
          (median_delta >= float(thresholds["depth_margin_m"]) or
           (external_building_fraction < float(thresholds["semantic_majority_fraction"]) and
            closer_fraction <= float(thresholds["termination_max_closer_fraction"])))):
        boundary_type, reason = "PHYSICAL_TERMINATION", "target ends before non-occluding exterior region"
    else:
        boundary_type, reason = "UNRESOLVED", "bilateral semantics/depth do not prove termination"
    return {"boundary_type": boundary_type, "reason": reason, **metrics}


def contour_action_axis_coordinate(contour: np.ndarray, depth_m: np.ndarray,
                                   K: np.ndarray, T_world_camera: np.ndarray,
                                   action_axis: Sequence[float]) -> dict:
    """Backproject target contour pixels and summarize a camera-motion axis coordinate."""
    axis = np.asarray(action_axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-9:
        raise ValueError("action axis must be non-zero")
    axis /= norm
    ys, xs = np.where(np.asarray(contour, dtype=bool))
    coordinates = []
    world_points = []
    depth = np.asarray(depth_m, dtype=float)
    for y, x in zip(ys.tolist(), xs.tolist()):
        value = float(depth[y, x])
        if not np.isfinite(value) or value <= 0.05 or value >= 999.0:
            continue
        camera = pixel_depth_to_camera_point((x, y), value, K, mode="z-depth")
        world = camera_to_world(camera, T_world_camera)
        world_points.append(world)
        coordinates.append(float(np.dot(world, axis)))
    if not coordinates:
        return {"valid_contour_point_count": 0, "action_axis_median_m": None,
                "action_axis_mad_m": None, "world_points_sample": []}
    values = np.asarray(coordinates, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sample = np.asarray(world_points)[::max(1, len(world_points) // 100)]
    return {"valid_contour_point_count": int(len(values)),
            "action_axis_median_m": median, "action_axis_mad_m": mad,
            "world_points_sample": sample.tolist()}


def official_tier_m(frame_metrics: Sequence[Mapping], thresholds_m: Sequence[float],
                    gate_threshold_m: float) -> dict:
    """Aggregate plane-free per-frame action-axis boundary coordinates."""
    valid = [row for row in frame_metrics
             if row.get("action_axis_median_m") is not None and
             int(row.get("valid_contour_point_count", 0)) > 0]
    values = np.asarray([row["action_axis_median_m"] for row in valid], dtype=float)
    spread = float(np.ptp(values)) if len(values) else None
    standard_deviation = float(np.std(values)) if len(values) else None
    sensitivity = {f"{float(threshold):.2f}m": bool(spread is not None and spread <= float(threshold))
                   for threshold in thresholds_m}
    passed = len(values) >= 4 and spread is not None and spread <= float(gate_threshold_m)
    return {"status": "PASS" if passed else "FAIL", "frame_count": len(valid),
            "spread_m": spread, "standard_deviation_m": standard_deviation,
            "gate_threshold_m": float(gate_threshold_m), "sensitivity": sensitivity,
            "absolute_accuracy": "NOT_EVALUATED",
            "uses_plane": False, "uses_bbox": False, "uses_legacy_boundary": False,
            "frames": [dict(row) for row in valid]}


def pose_repeatability(transforms: Sequence[np.ndarray], max_position_error_m: float,
                       max_rotation_error_deg: float) -> dict:
    matrices = [np.asarray(value, dtype=float) for value in transforms]
    if len(matrices) < 4:
        return {"status": "FAIL", "reason": "STRADDLE plus three repeats required",
                "frame_count": len(matrices)}
    reference = matrices[0]
    position_errors, rotation_errors = [], []
    for matrix in matrices[1:]:
        position_errors.append(float(np.linalg.norm(matrix[:3, 3] - reference[:3, 3])))
        rotation = reference[:3, :3].T @ matrix[:3, :3]
        cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
        rotation_errors.append(float(math.degrees(math.acos(cosine))))
    max_position = max(position_errors, default=0.0)
    max_rotation = max(rotation_errors, default=0.0)
    passed = max_position <= float(max_position_error_m) and max_rotation <= float(max_rotation_error_deg)
    return {"status": "PASS" if passed else "FAIL", "frame_count": len(matrices),
            "max_position_error_m": max_position,
            "max_rotation_error_deg": max_rotation,
            "position_threshold_m": float(max_position_error_m),
            "rotation_threshold_deg": float(max_rotation_error_deg)}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest_hashes(entries: Sequence[Mapping], project_root: str | Path = ".") -> dict:
    root = Path(project_root)
    checked, mismatches, missing = 0, [], []
    for entry in entries:
        for name, item in entry.get("files", {}).items():
            path = root / item["path"]
            if not path.exists():
                missing.append({"frame_id": entry.get("frame_id"), "name": name,
                                "path": item["path"]})
                continue
            checked += 1
            actual = sha256_file(path)
            if actual != item.get("sha256"):
                mismatches.append({"frame_id": entry.get("frame_id"), "name": name,
                                   "expected": item.get("sha256"), "actual": actual})
    return {"status": "PASS" if not missing and not mismatches and checked else "FAIL",
            "checked_file_count": checked, "missing": missing, "mismatches": mismatches}
