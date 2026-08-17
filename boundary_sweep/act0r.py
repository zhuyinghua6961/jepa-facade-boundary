"""ACT-0R pixel-derived boundary evidence and plane-free repeatability."""

from __future__ import annotations

from collections import Counter
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


def boundary_bilateral_samples(target_mask: np.ndarray, contour: np.ndarray,
                               depth_m: np.ndarray, semantic: np.ndarray,
                               instance: np.ndarray, direction: str,
                               offset_px: int) -> dict:
    """Return target/external sample pairs at a directional pixel contour."""
    target = np.asarray(target_mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=float)
    semantic = np.asarray(semantic)
    instance = np.asarray(instance)
    ys, target_x, external_x = _side_samples(
        contour, target, direction, int(offset_px))
    bilateral = target[ys, target_x] & ~target[ys, external_x]
    ys = ys[bilateral]
    target_x = target_x[bilateral]
    external_x = external_x[bilateral]
    target_depth = depth[ys, target_x]
    external_depth = depth[ys, external_x]
    valid_depth = (
        np.isfinite(target_depth) & np.isfinite(external_depth) &
        (target_depth > 0.05) & (external_depth > 0.05) &
        (target_depth < 999.0) & (external_depth < 999.0)
    )
    return {
        "y": ys,
        "target_x": target_x,
        "external_x": external_x,
        "target_depth_m": target_depth,
        "external_depth_m": external_depth,
        "valid_depth": valid_depth,
        "external_semantic": semantic[ys, external_x],
        "external_instance": instance[ys, external_x],
    }


def _value_histogram(values: np.ndarray, limit: int = 8) -> list[dict]:
    unique, counts = np.unique(np.asarray(values), return_counts=True)
    order = np.argsort(counts)[::-1][:int(limit)]
    return [{"value": int(unique[index]), "count": int(counts[index])}
            for index in order]


def classify_boundary_pixels(target_mask: np.ndarray, contour: np.ndarray,
                             depth_m: np.ndarray, semantic: np.ndarray,
                             instance: np.ndarray, target_instance_id: int,
                             direction: str, thresholds: Mapping) -> dict:
    """Classify one observed target/non-target boundary from bilateral pixels."""
    offset = int(thresholds["side_probe_offset_px"])
    samples = boundary_bilateral_samples(
        target_mask, contour, depth_m, semantic, instance, direction, offset)
    ys = samples["y"]
    if not len(ys):
        return {"boundary_type": "UNRESOLVED", "reason": "no bilateral samples",
                "bilateral_sample_count": 0}
    target_x = samples["target_x"]
    external_x = samples["external_x"]
    target_depth = samples["target_depth_m"]
    external_depth = samples["external_depth_m"]
    valid_depth = samples["valid_depth"]
    depth_delta = external_depth[valid_depth] - target_depth[valid_depth]
    ext_semantic = samples["external_semantic"]
    ext_instance = samples["external_instance"]
    count = int(len(ys))
    valid_count = int(valid_depth.sum())
    building_tag = int(thresholds["building_semantic_tag"])
    external_building_fraction = float(np.mean(ext_semantic == building_tag)) if count else 0.0
    external_non_target_fraction = float(np.mean(ext_instance != int(target_instance_id))) if count else 0.0
    median_delta = float(np.median(depth_delta)) if valid_count else None
    closer_fraction = float(np.mean(depth_delta < -float(thresholds["depth_margin_m"]))) if valid_count else None
    same_depth_fraction = float(np.mean(np.abs(depth_delta) <= float(thresholds["same_depth_tolerance_m"]))) if valid_count else None
    sample_step = max(1, count // 32)
    sample_pixels = []
    for index in range(0, count, sample_step):
        valid = bool(valid_depth[index])
        sample_pixels.append({
            "y": int(ys[index]),
            "target_x": int(target_x[index]),
            "external_x": int(external_x[index]),
            "target_depth_m": float(target_depth[index]),
            "external_depth_m": float(external_depth[index]),
            "depth_delta_m": (float(external_depth[index] - target_depth[index])
                              if valid else None),
            "external_semantic": int(ext_semantic[index]),
            "external_instance": int(ext_instance[index]),
            "valid_depth_pair": valid,
        })
    metrics = {
        "bilateral_sample_count": count,
        "valid_depth_pair_count": valid_count,
        "external_building_fraction": external_building_fraction,
        "external_non_target_instance_fraction": external_non_target_fraction,
        "external_minus_target_depth_median_m": median_delta,
        "external_closer_fraction": closer_fraction,
        "same_depth_fraction": same_depth_fraction,
        "target_depth_median_m": (float(np.median(target_depth[valid_depth]))
                                  if valid_count else None),
        "external_depth_median_m": (float(np.median(external_depth[valid_depth]))
                                    if valid_count else None),
        "depth_delta_mad_m": (float(np.median(np.abs(
            depth_delta - np.median(depth_delta)))) if valid_count else None),
        "external_semantic_histogram": _value_histogram(ext_semantic),
        "external_instance_histogram": _value_histogram(ext_instance),
        "sample_pixels": sample_pixels,
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


def boundary_type_consensus(frame_results: Sequence[Mapping],
                            minimum_agreement: int = 3) -> dict:
    """Resolve a boundary type only when enough independent frames agree."""
    types = [str(row.get("boundary_type", "UNRESOLVED"))
             for row in frame_results]
    counts = Counter(value if value in BOUNDARY_TYPES else "UNRESOLVED"
                     for value in types)
    winner, count = counts.most_common(1)[0] if counts else ("UNRESOLVED", 0)
    resolved = count >= int(minimum_agreement) and winner != "UNRESOLVED"
    return {
        "status": "PASS" if resolved else "FAIL",
        "boundary_type": winner if resolved else "UNRESOLVED",
        "consensus_count": int(count),
        "frame_count": len(types),
        "minimum_agreement": int(minimum_agreement),
        "counts": {name: int(counts.get(name, 0)) for name in sorted(BOUNDARY_TYPES)},
    }


def tier_v_from_pixel_frames(frame_metrics: Sequence[Mapping],
                             thresholds: Mapping,
                             minimum_pass_frames: int = 3) -> dict:
    """Evaluate repeated-frame Tier V directly from directional mask pixels."""
    rows = []
    for index, frame in enumerate(frame_metrics):
        passed = (
            bool(frame.get("contour_present")) and
            float(frame.get("span_over_target_bbox_height", 0.0)) >=
            float(thresholds["tier_v_min_span_over_target_bbox"]) and
            float(frame.get("target_side_fraction", 0.0)) >=
            float(thresholds["min_target_side_fraction"]) and
            float(frame.get("external_side_fraction", 0.0)) >=
            float(thresholds["min_external_side_fraction"])
        )
        rows.append({"frame_index": index, "pass": bool(passed)})
    pass_count = sum(row["pass"] for row in rows)
    passed = len(rows) >= 4 and pass_count >= int(minimum_pass_frames)
    return {
        "status": "PASS" if passed else "FAIL",
        "frame_count": len(rows),
        "pass_count": int(pass_count),
        "minimum_pass_frames": int(minimum_pass_frames),
        "thresholds": {
            "span_over_target_bbox_height": float(
                thresholds["tier_v_min_span_over_target_bbox"]),
            "target_side_fraction": float(thresholds["min_target_side_fraction"]),
            "external_side_fraction": float(thresholds["min_external_side_fraction"]),
        },
        "frames": rows,
        "uses_role_labels": False,
        "uses_plane_or_bbox_geometry": False,
    }


def select_repeated_pose_group(transforms: Sequence[np.ndarray],
                               minimum_count: int = 4,
                               position_tolerance_m: float = 0.01,
                               rotation_tolerance_deg: float = 0.05) -> dict:
    """Find the largest frozen-pose group without consulting role labels."""
    matrices = [np.asarray(value, dtype=float) for value in transforms]
    best = []
    for reference_index, reference in enumerate(matrices):
        group = []
        for index, matrix in enumerate(matrices):
            position_error = float(np.linalg.norm(
                matrix[:3, 3] - reference[:3, 3]))
            relative = reference[:3, :3].T @ matrix[:3, :3]
            cosine = float(np.clip(
                (np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
            rotation_error = float(math.degrees(math.acos(cosine)))
            if (position_error <= float(position_tolerance_m) and
                    rotation_error <= float(rotation_tolerance_deg)):
                group.append(index)
        if len(group) > len(best):
            best = group
    return {
        "status": "PASS" if len(best) >= int(minimum_count) else "FAIL",
        "indices": best,
        "frame_count": len(best),
        "minimum_count": int(minimum_count),
        "position_tolerance_m": float(position_tolerance_m),
        "rotation_tolerance_deg": float(rotation_tolerance_deg),
        "uses_role_labels": False,
    }


def action_axis_from_transforms(transforms: Sequence[np.ndarray]) -> dict:
    """Derive the dominant translation axis from actual camera transforms."""
    matrices = [np.asarray(value, dtype=float) for value in transforms]
    positions = np.asarray([matrix[:3, 3] for matrix in matrices], dtype=float)
    if len(positions) < 2:
        return {"status": "FAIL", "reason": "at least two poses required"}
    centered = positions - positions.mean(axis=0)
    _u, singular, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    direction = positions[-1] - positions[0]
    if float(np.dot(axis, direction)) < 0:
        axis = -axis
    projections = centered @ axis
    residuals = np.linalg.norm(centered - np.outer(projections, axis), axis=1)
    motion_span = float(np.ptp(projections))
    return {
        "status": "PASS" if motion_span > 1e-6 else "FAIL",
        "axis": axis.tolist(),
        "motion_span_m": motion_span,
        "max_orthogonal_residual_m": float(np.max(residuals)),
        "singular_values": singular.tolist(),
        "uses_commanded_action": False,
        "uses_plane_or_bbox": False,
    }


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
