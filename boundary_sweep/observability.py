"""Offline MASK-1R1 event and OBS-0 observability primitives.

This module deliberately keeps simulator privileged data on the target/GT
side of the analysis.  Probe features are RGB, action and pose only.
"""

from __future__ import annotations

import math
import warnings
from typing import Iterable, Sequence

import cv2
import numpy as np
from scipy.stats import spearmanr

from .geometry import camera_to_world, pixel_depth_to_camera_point
from .segmentation import outer_transition_contour


def first_threshold_step(values: Sequence[float], threshold: float) -> int | None:
    """Return the first index whose value reaches a threshold."""
    for index, value in enumerate(values):
        if np.isfinite(value) and float(value) >= float(threshold):
            return int(index)
    return None


def local_contour_evidence(mask: np.ndarray, direction: str,
                           min_side_fraction: float = 0.05,
                           min_span_fraction: float = 0.70) -> dict:
    """Recompute a directional exterior contour event from one instance mask."""
    if direction not in {"LEFT", "RIGHT"}:
        raise ValueError("direction must be LEFT or RIGHT")
    target = np.asarray(mask, dtype=bool)
    contour = outer_transition_contour(target)
    height, width = target.shape
    ys, xs = np.where(contour)
    if len(xs) == 0:
        return {
            "local_straddle": False, "contour_present": False,
            "contour_span_fraction": 0.0, "contour_centroid_px": None,
            "target_side_fraction": 1.0, "external_side_fraction": 0.0,
            "contour": contour, "reason": "no exterior transition contour",
        }
    # Select the longest directional connected component.  This avoids
    # unrelated building fragments while retaining the actual target edge.
    count, labels = cv2.connectedComponents(contour.astype(np.uint8), connectivity=8)
    components = []
    for label in range(1, count):
        cy, cx = np.where(labels == label)
        if len(cx) < 10:
            continue
        centroid = float(cx.mean())
        expected = centroid < width * 0.5 if direction == "LEFT" else centroid > width * 0.5
        span = float((cy.max() - cy.min() + 1) / max(height, 1))
        components.append((expected, span, len(cx), centroid, label))
    candidates = [row for row in components if row[0]]
    if candidates:
        selected = max(candidates, key=lambda row: (row[1], row[2]))[-1]
        contour = labels == selected
        ys, xs = np.where(contour)
    span = float((ys.max() - ys.min() + 1) / max(height, 1))
    centroid = float(xs.mean())
    split = max(min(int(round(centroid)), width - 1), 1)
    if direction == "LEFT":
        external = target[:, :split]
        target_side = target[:, min(split + 1, width - 1):]
    else:
        external = target[:, min(split + 1, width - 1):]
        target_side = target[:, :split]
    target_fraction = float(target_side.mean()) if target_side.size else 0.0
    external_fraction = float((~external).mean()) if external.size else 0.0
    directional = centroid < width * 0.5 if direction == "LEFT" else centroid > width * 0.5
    local = bool(directional and span >= min_span_fraction and
                target_fraction >= min_side_fraction and
                external_fraction >= min_side_fraction)
    return {
        "local_straddle": local, "contour_present": True,
        "contour_span_fraction": span, "contour_centroid_px": centroid,
        "target_side_fraction": target_fraction,
        "external_side_fraction": external_fraction, "contour": contour,
        "reason": "directional exterior contour with both sides observed" if local
                   else "contour does not meet directional side/span criteria",
    }


def action_axis_from_poses(positions: Sequence[Sequence[float]]) -> tuple[np.ndarray, dict]:
    """Derive a unit action axis from observed camera translations."""
    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("positions must have shape (N,3)")
    deltas = np.diff(points, axis=0)
    lengths = np.linalg.norm(deltas, axis=1) if len(deltas) else np.empty(0)
    valid = deltas[lengths > 1e-8]
    if len(valid) == 0:
        return np.array([0.0, 1.0, 0.0]), {"motion_samples": 0, "fallback": True}
    axis = np.median(valid, axis=0)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    return axis, {"motion_samples": int(len(valid)), "fallback": False,
                  "median_delta_m": np.median(valid, axis=0).tolist()}


def robust_median_mad(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    median = np.median(arr, axis=0)
    mad = np.median(np.abs(arr - median), axis=0)
    return median, mad


def backproject_contour(contour: np.ndarray, depth_m: np.ndarray, K: np.ndarray,
                        transform, action_axis: Sequence[float],
                        depth_reference: np.ndarray | None = None,
                        depth_tolerance_m: float = 0.25) -> dict:
    """Backproject target-side contour pixels with z-depth, without a plane filter.

    A robust target-mask depth reference rejects pixels whose sensor depth is
    inconsistent with the observed target surface.  This is an occlusion/
    sensor-consistency check, not a fitted-plane or bbox acceptance filter.
    """
    ys, xs = np.where(np.asarray(contour, dtype=bool))
    axis = np.asarray(action_axis, dtype=float)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    points = []
    rejected = 0
    height, width = depth_m.shape[:2]
    reference = None
    if depth_reference is not None:
        values = np.asarray(depth_reference, dtype=float)
        values = values[np.isfinite(values) & (values > 0.05)]
        if len(values):
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            reference = (median, max(float(depth_tolerance_m), 6.0 * mad))
    for y, x in zip(ys.tolist(), xs.tolist()):
        value = float(depth_m[y, x]) if 0 <= y < height and 0 <= x < width else float("nan")
        if not np.isfinite(value) or value <= 0.05 or (reference is not None and abs(value - reference[0]) > reference[1]):
            rejected += 1
            continue
        camera = pixel_depth_to_camera_point((x, y), value, K, mode="z-depth")
        points.append(camera_to_world(camera, transform))
    if not points:
        return {
            "valid_boundary_points": 0, "rejected_depth_points": int(rejected),
            "median_world_xyz": None, "MAD_world_xyz": None,
            "median_action_axis_coordinate": None, "action_axis_MAD": None,
            "vertical_span_m": None, "world_points_sample": [],
        }
    world = np.asarray(points, dtype=float)
    median, mad = robust_median_mad(world)
    coordinates = world @ axis
    return {
        "valid_boundary_points": int(len(world)),
        "rejected_depth_points": int(rejected),
        "median_world_xyz": median.tolist(), "MAD_world_xyz": mad.tolist(),
        "median_action_axis_coordinate": float(np.median(coordinates)),
        "action_axis_MAD": float(np.median(np.abs(coordinates - np.median(coordinates)))),
        "vertical_span_m": float(np.max(world[:, 2]) - np.min(world[:, 2])),
        "world_points_sample": world[::max(1, len(world) // 200)].tolist(),
    }


def grayscale_descriptor(image: np.ndarray, size: tuple[int, int] = (32, 24)) -> np.ndarray:
    """Deterministic normalized grayscale descriptor."""
    arr = np.asarray(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
    small = cv2.resize(gray, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    small -= float(small.mean())
    scale = float(small.std())
    return (small / max(scale, 1e-6)).reshape(-1)


def color_histogram(image: np.ndarray, bins: int = 16) -> np.ndarray:
    arr = np.asarray(image, dtype=np.uint8)
    parts = []
    for channel in range(3):
        hist, _ = np.histogram(arr[..., channel], bins=bins, range=(0, 256), density=True)
        parts.append(hist.astype(np.float32))
    return np.concatenate(parts)


def hog_descriptor(image: np.ndarray, cell: int = 16, bins: int = 9) -> np.ndarray:
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=False)
    height, width = gray.shape
    values = []
    for y in range(0, height - cell + 1, cell):
        for x in range(0, width - cell + 1, cell):
            hist, _ = np.histogram(angle[y:y + cell, x:x + cell], bins=bins,
                                   range=(0, 2 * np.pi), weights=magnitude[y:y + cell, x:x + cell])
            values.extend(hist / max(float(np.linalg.norm(hist)), 1e-6))
    return np.asarray(values, dtype=np.float32)


def rgb_descriptor(image: np.ndarray) -> np.ndarray:
    compact = cv2.resize(np.asarray(image), (160, 120), interpolation=cv2.INTER_AREA)
    return np.concatenate([grayscale_descriptor(image), color_histogram(image), hog_descriptor(compact)])


def fixed_length_descriptor(descriptor: np.ndarray, length: int = 128) -> np.ndarray:
    """Deterministically subsample a descriptor to a fixed memory budget."""
    values = np.asarray(descriptor, dtype=np.float32).reshape(-1)
    if length <= 0:
        raise ValueError("length must be positive")
    if len(values) < length:
        raise ValueError("descriptor is shorter than requested fixed length")
    indices = np.linspace(0, len(values) - 1, num=length, dtype=np.int64)
    return values[indices]


def aligned_ssim(left: np.ndarray, right: np.ndarray) -> tuple[float, float]:
    """Return phase-aligned global SSIM and the estimated x/y shift."""
    a = cv2.cvtColor(np.asarray(left), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    b = cv2.cvtColor(np.asarray(right), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    a = cv2.resize(a, (320, 240), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (320, 240), interpolation=cv2.INTER_AREA)
    try:
        shift, _response = cv2.phaseCorrelate(a, b)
        # phaseCorrelate(a, b) reports the translation from ``a`` to ``b``;
        # warping ``b`` back onto ``a`` therefore requires the inverse shift.
        matrix = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])
        b = cv2.warpAffine(b, matrix, (b.shape[1], b.shape[0]), borderMode=cv2.BORDER_REFLECT)
    except cv2.error:
        shift = (0.0, 0.0)
    mean_a, mean_b = float(a.mean()), float(b.mean())
    var_a, var_b = float(a.var()), float(b.var())
    cov = float(((a - mean_a) * (b - mean_b)).mean())
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mean_a * mean_b + c1) * (2 * cov + c2) /
             max((mean_a * mean_a + mean_b * mean_b + c1) *
                 (var_a + var_b + c2), 1e-12))
    return float(np.clip(score, -1.0, 1.0)), float(math.hypot(*shift))


def ridge_fit_predict(train_x: np.ndarray, train_y: np.ndarray,
                      test_x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Fit ridge with the smaller of the sample- and feature-space systems."""
    x = np.asarray(train_x, dtype=float)
    z = np.asarray(test_x, dtype=float)
    y = np.asarray(train_y, dtype=float)
    if x.ndim != 2 or z.ndim != 2 or x.shape[1] != z.shape[1]:
        raise ValueError("train_x and test_x must be 2-D with equal feature counts")
    if len(x) != len(y) or len(x) == 0:
        raise ValueError("train_y must contain one target per non-empty train row")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x = (x - mean) / scale
    z = (z - mean) / scale
    target_mean = y.mean(axis=0)
    centered_y = y - target_mean
    if x.shape[1] <= x.shape[0]:
        system = x.T @ x + np.eye(x.shape[1], dtype=float) * float(alpha)
        weights = np.linalg.solve(system, x.T @ centered_y)
    else:
        # The OBS probes are deliberately wide and tiny.  Solving in sample
        # space prevents feature_dim^2 allocation and feature_dim^3 compute.
        system = x @ x.T + np.eye(x.shape[0], dtype=float) * float(alpha)
        dual = np.linalg.solve(system, centered_y)
        weights = x.T @ dual
    return z @ weights + target_mean


def regression_metrics(target: Sequence[float], prediction: Sequence[float],
                        baseline_mae: float | None = None) -> dict:
    actual = np.asarray(target, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    errors = np.abs(pred - actual)
    if len(actual) > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            correlation = spearmanr(actual, pred)
        rho = getattr(correlation, "statistic", getattr(correlation, "correlation", float("nan")))
    else:
        rho = float("nan")
    result = {
        "MAE_m": float(errors.mean()),
        "median_absolute_error_m": float(np.median(errors)),
        "spearman_rho": None if not np.isfinite(rho) else float(rho),
        "prediction_range_m": [float(pred.min()), float(pred.max())] if len(pred) else [],
    }
    if baseline_mae is not None:
        result["constant_baseline_improvement_m"] = float(baseline_mae - result["MAE_m"])
    return result


def grouped_history_descriptors(records: Sequence[dict], descriptor_key: str = "descriptor",
                                group_key: str = "trajectory_id",
                                order_key: str = "step_index") -> tuple[np.ndarray, np.ndarray]:
    """Build previous-frame descriptors independently inside each trajectory.

    The returned ``history_valid_mask`` is false for every trajectory's first
    frame.  No ordering, normalization or feature value crosses a group.
    """
    current = [np.asarray(row[descriptor_key], dtype=float) for row in records]
    previous = [value.copy() for value in current]
    valid = np.zeros(len(records), dtype=bool)
    groups: dict[object, list[int]] = {}
    for index, row in enumerate(records):
        groups.setdefault(row[group_key], []).append(index)
    for indices in groups.values():
        ordered = sorted(indices, key=lambda index: records[index][order_key])
        for position, index in enumerate(ordered):
            if position == 0:
                previous[index] = current[index].copy()
                valid[index] = False
            else:
                previous[index] = current[ordered[position - 1]].copy()
                valid[index] = True
    return np.asarray(previous), valid


def _ssim_gray(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    mean_a, mean_b = float(a.mean()), float(b.mean())
    var_a, var_b = float(a.var()), float(b.var())
    cov = float(((a - mean_a) * (b - mean_b)).mean())
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mean_a * mean_b + c1) * (2 * cov + c2) /
             max((mean_a * mean_a + mean_b * mean_b + c1) *
                 (var_a + var_b + c2), 1e-12))
    return float(np.clip(score, -1.0, 1.0))


def raw_ssim(left: np.ndarray, right: np.ndarray) -> float:
    """Global SSIM before geometric alignment."""
    a = cv2.cvtColor(np.asarray(left), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    b = cv2.cvtColor(np.asarray(right), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    a = cv2.resize(a, (320, 240), interpolation=cv2.INTER_AREA)
    b = cv2.resize(b, (320, 240), interpolation=cv2.INTER_AREA)
    return _ssim_gray(a, b)


def similarity_metrics(left: np.ndarray, right: np.ndarray) -> dict:
    """Return raw/aligned SSIM, HOG cosine and colour-histogram distance."""
    phase_score, shift = aligned_ssim(left, right)
    hog_left = hog_descriptor(cv2.resize(np.asarray(left), (160, 120), interpolation=cv2.INTER_AREA))
    hog_right = hog_descriptor(cv2.resize(np.asarray(right), (160, 120), interpolation=cv2.INTER_AREA))
    cosine = float(np.dot(hog_left, hog_right) /
                   max(float(np.linalg.norm(hog_left) * np.linalg.norm(hog_right)), 1e-12))
    hist_left = color_histogram(left)
    hist_right = color_histogram(right)
    hist_distance = float(0.5 * np.abs(hist_left - hist_right).sum())
    return {"raw_ssim": raw_ssim(left, right), "phase_aligned_ssim": phase_score,
            "phase_shift_px": shift, "hog_cosine": cosine,
            "color_histogram_distance": hist_distance}
