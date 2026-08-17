"""ACT-0 counterfactual facade pilot primitives.

Instance segmentation, semantic tags, depth and raycasts are privileged
capture/evaluation signals. They are never included in RGB probe features.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Mapping, Sequence

import numpy as np

from .geometry import camera_to_world, fit_plane, pixel_depth_to_camera_point
from .observability import local_contour_evidence, raw_ssim
from .segmentation import largest_connected_component_ratio, outer_transition_contour


def backproject_mask_points(mask: np.ndarray, depth_m: np.ndarray, K: np.ndarray,
                            camera_transform, pixel_step: int = 12) -> np.ndarray:
    """Backproject a bounded subsample of valid z-depth target pixels."""
    if pixel_step < 1:
        raise ValueError("pixel_step must be positive")
    target = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=float)
    ys, xs = np.where(target)
    keep = (ys % pixel_step == 0) & (xs % pixel_step == 0)
    points = []
    for y, x in zip(ys[keep].tolist(), xs[keep].tolist()):
        value = float(depth[y, x])
        if not np.isfinite(value) or value <= 0.05:
            continue
        camera = pixel_depth_to_camera_point((x, y), value, K, mode="z-depth")
        points.append(camera_to_world(camera, camera_transform))
    return np.asarray(points, dtype=float).reshape((-1, 3))


def fit_target_plane(points: Sequence[Sequence[float]], expected_normal,
                     seed_bin_width_m: float = 0.10,
                     inlier_tolerance_m: float = 0.15) -> dict:
    """Fit the dominant expected-normal plane before free SVD refinement."""
    arr = np.asarray(points, dtype=float)
    expected = np.asarray(expected_normal, dtype=float)
    expected /= max(float(np.linalg.norm(expected)), 1e-12)
    coordinate = arr @ expected
    lower, upper = float(np.min(coordinate)), float(np.max(coordinate))
    bins = max(1, int(math.ceil((upper - lower) / seed_bin_width_m)))
    histogram, edges = np.histogram(coordinate, bins=bins, range=(lower, upper + 1e-9))
    index = int(np.argmax(histogram))
    seed = float((edges[index] + edges[index + 1]) / 2.0)
    selected = np.abs(coordinate - seed) <= float(inlier_tolerance_m)
    if int(selected.sum()) < 3:
        selected = np.ones(len(arr), dtype=bool)
    origin, normal, _ = fit_plane(arr[selected])
    if float(np.dot(normal, expected)) < 0:
        normal = -normal
    residuals_all = np.abs((arr - origin) @ normal)
    selected = residuals_all <= float(inlier_tolerance_m)
    if int(selected.sum()) >= 3:
        origin, normal, _ = fit_plane(arr[selected])
        if float(np.dot(normal, expected)) < 0:
            normal = -normal
    residuals_all = np.abs((arr - origin) @ normal)
    residuals = residuals_all[selected]
    alignment = float(np.clip(np.dot(normal, expected), -1.0, 1.0))
    return {
        "origin": origin,
        "normal": normal,
        "median_residual_m": float(np.median(residuals)),
        "p95_residual_m": float(np.percentile(residuals, 95)),
        "max_residual_m": float(np.max(residuals)) if len(residuals) else None,
        "normal_error_deg": float(math.degrees(math.acos(alignment))),
        "support_count": int(len(arr)), "inlier_count": int(selected.sum()),
        "inlier_ratio": float(selected.mean()),
        "inlier_tolerance_m": float(inlier_tolerance_m),
    }


def directional_instance_contour(mask: np.ndarray, direction: str) -> dict:
    """Extract one image-side line from a closed exterior instance contour."""
    target = np.asarray(mask, dtype=bool)
    if direction not in {"LEFT", "RIGHT"}:
        raise ValueError("direction must be LEFT or RIGHT")
    exterior_contour = outer_transition_contour(target)
    contour = np.zeros_like(target)
    observed_rows = target_side = external_side = 0
    for y in range(1, target.shape[0] - 1):
        xs = np.where(exterior_contour[y])[0]
        if not len(xs):
            continue
        x = int(xs.min() if direction == "LEFT" else xs.max())
        if x <= 0 or x >= target.shape[1] - 1:
            continue
        contour[y, x] = True
        observed_rows += 1
        if direction == "LEFT":
            target_side += int(target[y, min(x + 2, target.shape[1] - 1)])
            external_side += int(not target[y, max(x - 2, 0)])
        else:
            target_side += int(target[y, max(x - 2, 0)])
            external_side += int(not target[y, min(x + 2, target.shape[1] - 1)])
    span = float(observed_rows / max(target.shape[0], 1))
    return {"contour": contour, "contour_present": observed_rows > 0,
            "contour_span_fraction": span,
            "target_side_fraction": float(target_side / max(observed_rows, 1)),
            "external_side_fraction": float(external_side / max(observed_rows, 1)),
            "local_straddle": bool(observed_rows and target_side and external_side),
            "reason": "directional row envelope of exterior instance contour" if observed_rows
                      else "no directional exterior instance contour"}


def orthogonal_surface_axes(candidate_h, plane_normal) -> tuple[np.ndarray, np.ndarray]:
    normal = np.asarray(plane_normal, dtype=float)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    horizontal = np.asarray(candidate_h, dtype=float)
    horizontal -= normal * float(np.dot(horizontal, normal))
    horizontal /= max(float(np.linalg.norm(horizontal)), 1e-12)
    vertical = np.cross(normal, horizontal)
    vertical /= max(float(np.linalg.norm(vertical)), 1e-12)
    if vertical[2] < 0:
        vertical = -vertical
    return horizontal, vertical


def fit_terminal_boundaries(boundary_points: Mapping[str, Sequence[np.ndarray]],
                            plane_origin, horizontal_axis, vertical_axis) -> dict:
    """Fit LEFT/RIGHT world lines from independent view contour points."""
    origin = np.asarray(plane_origin, dtype=float)
    h = np.asarray(horizontal_axis, dtype=float)
    v = np.asarray(vertical_axis, dtype=float)
    result = {}
    all_points = []
    for direction in ("LEFT", "RIGHT"):
        per_view = []
        merged = []
        for points in boundary_points.get(direction, []):
            arr = np.asarray(points, dtype=float).reshape((-1, 3))
            if not len(arr):
                continue
            coordinates = (arr - origin) @ np.column_stack([h, v])
            per_view.append(float(np.median(coordinates[:, 0])))
            merged.append(coordinates)
            all_points.append(coordinates)
        if not merged:
            result[direction] = {"view_count": 0, "world_line": None,
                                 "per_view_horizontal_m": [],
                                 "horizontal_std_m": None}
            continue
        coords = np.row_stack(merged)
        u = float(np.median(per_view))
        result[direction] = {
            "view_count": len(per_view),
            "per_view_horizontal_m": per_view,
            "horizontal_std_m": float(np.std(per_view)),
            "horizontal_coordinate_m": u,
            "vertical_range_m": [float(np.percentile(coords[:, 1], 5)),
                                 float(np.percentile(coords[:, 1], 95))],
        }
    if all_points:
        vertical = np.row_stack(all_points)[:, 1]
        v_min, v_max = (float(np.percentile(vertical, q)) for q in (5, 95))
    else:
        v_min, v_max = 0.0, 1.0
    for direction in ("LEFT", "RIGHT"):
        item = result[direction]
        if item.get("view_count", 0):
            u = item["horizontal_coordinate_m"]
            item["world_line"] = {
                "start": (origin + h * u + v * v_max).tolist(),
                "end": (origin + h * u + v * v_min).tolist(),
            }
    return result


def candidate_quality(masks: Sequence[np.ndarray], boundary_audits: Mapping[str, Sequence[dict]],
                      target_id_stable: bool, plane: Mapping, boundaries: Mapping,
                      raycast_agreement: float | None, thresholds: Mapping) -> tuple[bool, list[str]]:
    reasons = []
    center = np.asarray(masks[len(masks) // 2], dtype=bool)
    coverage = float(center.mean())
    if not target_id_stable:
        reasons.append("target_instance_id_not_stable")
    if coverage < float(thresholds["min_target_coverage"]) or coverage > float(thresholds["max_target_coverage"]):
        reasons.append("center_target_coverage_out_of_range")
    if largest_connected_component_ratio(center) < float(thresholds["min_component_ratio"]):
        reasons.append("target_mask_fragmented")
    for direction in ("LEFT", "RIGHT"):
        audits = boundary_audits.get(direction, [])
        valid = sum(bool(row.get("contour_present")) and
                    float(row.get("contour_span_fraction", 0.0)) >= float(thresholds["min_contour_span_fraction"])
                    for row in audits)
        if valid < int(thresholds["min_boundary_views"]):
            reasons.append(f"{direction.lower()}_external_contour_insufficient")
        fitted = boundaries.get(direction, {})
        if fitted.get("view_count", 0) < int(thresholds["min_boundary_views"]):
            reasons.append(f"{direction.lower()}_world_boundary_insufficient")
        std = fitted.get("horizontal_std_m")
        if std is None or float(std) > float(thresholds["max_boundary_std_m"]):
            reasons.append(f"{direction.lower()}_boundary_repeatability_failed")
    plane_p95 = plane.get("p95_residual_m")
    normal_error = plane.get("normal_error_deg")
    if plane_p95 is None or float(plane_p95) > float(thresholds["max_plane_p95_residual_m"]):
        reasons.append("depth_plane_residual_too_large")
    if normal_error is None or float(normal_error) > float(thresholds["max_normal_error_deg"]):
        reasons.append("depth_plane_normal_inconsistent")
    if float(plane.get("inlier_ratio", 0.0)) < float(thresholds["min_plane_inlier_ratio"]):
        reasons.append("depth_plane_inlier_ratio_too_low")
    if raycast_agreement is None:
        reasons.append("raycast_evidence_missing")
    elif raycast_agreement < float(thresholds["min_raycast_agreement"]):
        reasons.append("raycast_depth_disagreement")
    left = boundaries.get("LEFT", {}).get("horizontal_coordinate_m")
    right = boundaries.get("RIGHT", {}).get("horizontal_coordinate_m")
    if left is None or right is None or abs(float(right) - float(left)) < float(thresholds["min_width_m"]):
        reasons.append("terminal_width_too_small_or_degenerate")
    return not reasons, reasons


def scout_sensor_pairing(views: Sequence[Mapping]) -> dict:
    """Audit synchronized scout quartets without loading sensor payloads."""
    rows = []
    for view in views:
        frames = view.get("sensor_frames", {})
        timestamps = view.get("sensor_timestamps", {})
        frame_values = list(frames.values())
        timestamp_values = list(timestamps.values())
        paired = (set(frames) == {"rgb", "depth", "semantic", "instance"} and
                  len(set(frame_values)) == 1 and
                  set(timestamps) == {"rgb", "depth", "semantic", "instance"} and
                  max(timestamp_values, default=math.inf) -
                  min(timestamp_values, default=-math.inf) <= 1e-6)
        rows.append({"view_index": view.get("view_index"),
                     "frame_id": view.get("frame_id"), "paired": bool(paired)})
    return {"status": "PASS" if rows and all(row["paired"] for row in rows) else "FAIL",
            "view_count": len(rows), "paired_view_count": sum(row["paired"] for row in rows),
            "views": rows}


def _contour_side_audit(rows: Sequence[Mapping], thresholds: Mapping) -> dict:
    present = [row for row in rows if row.get("contour_present")]
    spans = [float(row.get("contour_span_fraction", 0.0)) for row in present]
    sided = [row for row in present
             if float(row.get("target_side_fraction", 0.0)) >=
             float(thresholds["min_target_side_fraction"]) and
             float(row.get("external_side_fraction", 0.0)) >=
             float(thresholds["min_external_side_fraction"])]
    span_range = max(spans) - min(spans) if spans else None
    if not present:
        status = "NOT_OBSERVED"
        reason = "no target/non-target transition in the three scout poses"
    elif len(sided) < int(thresholds["min_contour_views"]):
        status = "FAIL"
        reason = "insufficient bilateral target/non-target support"
    elif span_range is not None and span_range > float(thresholds["max_contour_span_range"]):
        status = "NOT_OBSERVED"
        reason = "scout offsets do not provide a stable view of the same exterior transition"
    else:
        status = "PASS"
        reason = "multi-view target/non-target exterior-contour reference"
    return {"status": status, "reason": reason, "present_views": len(present),
            "bilateral_views": len(sided), "span_fractions": spans,
            "span_range": span_range}


def tier_visual_event(candidate: Mapping, thresholds: Mapping,
                      evidence_interpretation: Mapping | None = None) -> dict:
    """Evaluate visual-event suitability independently of metric/plane gates."""
    interpretation = dict(evidence_interpretation or {})
    pairing = scout_sensor_pairing(candidate.get("views", []))
    target_id = candidate.get("target_instance_id_16bit")
    stable = bool(candidate.get("target_instance_id_stable"))
    reasons = set(candidate.get("rejection_reasons", []))
    semantic = "PASS" if target_id is not None and candidate.get("center_target_coverage", 0) > 0 else "NOT_OBSERVED"
    component = ("NOT_OBSERVED" if target_id is None else
                 "FAIL" if "target_mask_fragmented" in reasons else "PASS")
    sides = {name: _contour_side_audit(candidate.get("boundary_audits", {}).get(name, []),
                                       thresholds)
             for name in ("LEFT", "RIGHT")}
    override = interpretation.get("tier_v_status")
    if override in {"FAIL", "NOT_OBSERVED"}:
        status = override
        reason = interpretation.get("rationale", "recorded compact-evidence interpretation")
    elif pairing["status"] != "PASS" or semantic != "PASS" or not stable or component != "PASS":
        status = "FAIL" if component == "FAIL" else "NOT_OBSERVED"
        reason = "target instance evidence is fragmented, unstable, or absent"
    elif any(side["status"] == "PASS" for side in sides.values()):
        status = "PASS"
        reason = "at least one action side has a stable exterior transition"
    elif any(side["status"] == "NOT_OBSERVED" for side in sides.values()):
        status = "NOT_OBSERVED"
        reason = "current scout poses do not establish an exterior transition"
    else:
        status = "FAIL"
        reason = "observed contours fail visual-event semantics"
    return {
        "status": status, "reason": reason, "sensor_pairing": pairing,
        "building_semantic": semantic,
        "instance_id_stability": "PASS" if stable else "NOT_OBSERVED",
        "instance_grouping": interpretation.get("instance_grouping", "RESOLVED_FOR_SELECTED_ID"),
        "largest_component": component,
        "largest_component_ratio": None,
        "largest_component_note": "exact ratio was not persisted; legacy threshold result only",
        "sides": sides,
        "evidence_scope": "geometry reference pending operator visual review",
        "uses_plane_or_raycast": False,
    }


def tier_metric_repeatability(candidate: Mapping, thresholds: Sequence[float],
                              official_evidence: Mapping | None = None) -> dict:
    """Report official contour repeatability only from plane/bbox-free evidence.

    Historical ``boundaries`` fields are exposed as a descriptive proxy but
    can never satisfy this gate because their coordinates used the old fitted
    plane basis.
    """
    required_sources = {"target_side_contour", "z_depth", "K", "T_world_camera",
                        "camera_motion_action_axis"}
    evidence = dict(official_evidence or {})
    sources = set(evidence.get("sources", []))
    official_values = evidence.get("per_direction_action_axis_coordinates_m", {})
    official_available = required_sources.issubset(sources) and bool(official_values)
    official = {}
    if official_available:
        for direction in ("LEFT", "RIGHT"):
            values = [float(value) for value in official_values.get(direction, [])]
            official[direction] = {"sample_count": len(values),
                                   "spread_m": float(np.std(values)) if len(values) >= 2 else None}
    legacy = {}
    for direction in ("LEFT", "RIGHT"):
        item = candidate.get("boundaries", {}).get(direction, {})
        legacy[direction] = {"view_count": int(item.get("view_count", 0)),
                             "spread_m": item.get("horizontal_std_m"),
                             "source": "legacy_plane_basis_proxy_not_an_official_tier_m_input"}
    sensitivity = {}
    for threshold in thresholds:
        key = f"{float(threshold):.2f}m"
        sensitivity[key] = any(item["view_count"] >= 2 and item["spread_m"] is not None and
                               float(item["spread_m"]) <= float(threshold)
                               for item in legacy.values())
    if not official_available:
        return {"status": "NOT_EVALUATED",
                "reason": "raw contour/depth/K records were not persisted; legacy plane-basis summaries are excluded",
                "required_sources": sorted(required_sources), "official": {},
                "legacy_proxy": legacy, "legacy_proxy_sensitivity": sensitivity,
                "uses_legacy_plane_or_bbox": False}
    passed = any(item["sample_count"] >= 2 and item["spread_m"] is not None and
                 item["spread_m"] <= 0.10 for item in official.values())
    return {"status": "PASS" if passed else "FAIL", "reason": "official action-axis spread",
            "required_sources": sorted(required_sources), "official": official,
            "legacy_proxy": legacy, "legacy_proxy_sensitivity": sensitivity,
            "uses_legacy_plane_or_bbox": False}


def tier_physical_plane(candidate: Mapping, thresholds: Mapping) -> dict:
    """Retain strict absolute-plane checks without changing Tier V."""
    plane = candidate.get("plane", {})
    raycast = candidate.get("raycast_audit", {})
    left = candidate.get("boundaries", {}).get("LEFT", {}).get("horizontal_coordinate_m")
    right = candidate.get("boundaries", {}).get("RIGHT", {}).get("horizontal_coordinate_m")
    width = abs(float(right) - float(left)) if left is not None and right is not None else None
    values = {
        "plane_p95_residual": plane.get("p95_residual_m"),
        "normal_error": plane.get("normal_error_deg"),
        "plane_inlier_ratio": plane.get("inlier_ratio"),
        "raycast_agreement": raycast.get("agreement"),
        "physical_width": width,
    }
    if all(value is None for value in values.values()):
        return {"status": "NOT_OBSERVED", "metrics": values,
                "reason": "no target plane or raycast evidence"}
    checks = {
        "plane_p95_residual": values["plane_p95_residual"] is not None and values["plane_p95_residual"] <= float(thresholds["max_plane_p95_residual_m"]),
        "normal_error": values["normal_error"] is not None and values["normal_error"] <= float(thresholds["max_normal_error_deg"]),
        "plane_inlier_ratio": values["plane_inlier_ratio"] is not None and values["plane_inlier_ratio"] >= float(thresholds["min_plane_inlier_ratio"]),
        "raycast_agreement": values["raycast_agreement"] is not None and values["raycast_agreement"] >= float(thresholds["min_raycast_agreement"]),
        "physical_width": values["physical_width"] is not None and values["physical_width"] >= float(thresholds["min_width_m"]),
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "metrics": values,
            "checks": checks, "reason": "strict single-plane absolute-3D gate"}


def start_match_metrics(left_rgb: np.ndarray, right_rgb: np.ndarray,
                        left_pose: np.ndarray, right_pose: np.ndarray) -> dict:
    a, b = np.asarray(left_pose, dtype=float), np.asarray(right_pose, dtype=float)
    position_error = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    rotation = a[:3, :3].T @ b[:3, :3]
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    rotation_error = float(math.degrees(math.acos(cosine)))
    x, y = np.asarray(left_rgb, dtype=np.int16), np.asarray(right_rgb, dtype=np.int16)
    return {"position_error_m": position_error, "rotation_error_deg": rotation_error,
            "initial_rgb_ssim": raw_ssim(left_rgb, right_rgb),
            "initial_rgb_mean_absolute_pixel_error": float(np.abs(x - y).mean())}


def directional_external_coverage(mask: np.ndarray, direction: str) -> float:
    """Measure non-target pixels on the commanded side of the target envelope."""
    target = np.asarray(mask, dtype=bool)
    if direction not in {"LEFT", "RIGHT"}:
        raise ValueError("direction must be LEFT or RIGHT")
    rows = np.where(target.any(axis=1))[0]
    if not len(rows):
        return 1.0
    external = 0
    total = target.shape[1] * len(rows)
    for y in rows.tolist():
        xs = np.where(target[y])[0]
        if direction == "LEFT":
            external += int(xs.min())
        else:
            external += int(target.shape[1] - 1 - xs.max())
    return float(external / max(total, 1))


def trajectory_outcome(frames: Sequence[Mapping], horizon_m: float) -> dict:
    event = next((row for row in frames if row.get("state") == "STRADDLE" and
                  row.get("capture_role") == "moving"), None)
    if event is None:
        distance = None
        censored = True
        steps = None
    else:
        distance = float(event["actual_displacement_m"])
        censored = False
        steps = int(event["step_index"])
    return {"event_within_horizon": event is not None,
            "distance_to_event_m": distance, "time_to_event_steps": steps,
            "censored": censored, "horizon_m": float(horizon_m)}


def preferred_action(left: Mapping, right: Mapping, tie_tolerance_m: float = 0.05) -> str:
    left_event, right_event = bool(left["event_within_horizon"]), bool(right["event_within_horizon"])
    if left_event and not right_event:
        return "LEFT"
    if right_event and not left_event:
        return "RIGHT"
    if not left_event and not right_event:
        return "NO_DECISION"
    delta = float(left["distance_to_event_m"]) - float(right["distance_to_event_m"])
    if abs(delta) <= float(tie_tolerance_m):
        return "TIE"
    return "LEFT" if delta < 0 else "RIGHT"


def fixed_schedule_audit(trajectories: Sequence[Mapping]) -> dict:
    events = [row for row in trajectories if row.get("event_within_horizon")]
    steps = {int(row["time_to_event_steps"]) for row in events}
    distances = {round(float(row["distance_to_event_m"]), 3) for row in events}
    step_sizes = {round(float(row["step_m"]), 3) for row in trajectories}
    passed = len(step_sizes) >= 3 and len(steps) >= 2 and len(distances) >= 3
    return {"status": "PASS" if passed else "FAIL", "event_steps": sorted(steps),
            "event_distances_m": sorted(distances), "step_sizes_m": sorted(step_sizes)}


def paired_surface_folds(rows: Sequence[Mapping]) -> list[dict]:
    surfaces = sorted({str(row["surface_id"]) for row in rows})
    folds = []
    for held_out in surfaces:
        train = [index for index, row in enumerate(rows) if str(row["surface_id"]) != held_out]
        test = [index for index, row in enumerate(rows) if str(row["surface_id"]) == held_out]
        folds.append({"held_out_surface": held_out, "train_indices": train, "test_indices": test})
    return folds


def label_balance(pairs: Iterable[Mapping]) -> dict:
    counts = Counter(str(row.get("preferred_action", "NO_DECISION")) for row in pairs)
    return {name: int(counts[name]) for name in ("LEFT", "RIGHT", "TIE", "NO_DECISION")}
