"""CAP-0 sensor-health metrics and fail-closed gate helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SEARCH_PLAN_SHA256 = "a56310883bb15513ea25c97c919d7faf14edb217b1a05fb0c4e12b060c664f73"
HEALTH_GATES = (
    "TICK_FAIL_FAST",
    "QUEUE_DEADLINE",
    "RAW_BUFFER_OWNERSHIP",
    "RAW_LENGTH_AND_HASH",
    "GPU_WARMUP_COMPLETE",
    "KNOWN_GOOD_POSE_RGB_INTEGRITY",
    "QUARTET_PAIRING_HEALTH",
    "POST_TELEPORT_HEALTH",
    "RENDER_INTEGRITY",
    "ROOT_CAUSE_CLASSIFIED",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_search_plan(path: str | Path, expected_sha256: str = SEARCH_PLAN_SHA256) -> dict:
    actual = sha256_file(path)
    return {"status": "PASS" if actual == expected_sha256 else "FAIL",
            "expected_sha256": expected_sha256, "actual_sha256": actual}


def enforce_saved_frame_limit(saved_count: int, maximum: int) -> None:
    if int(saved_count) > int(maximum):
        raise RuntimeError(f"saved diagnostic frame limit exceeded: {saved_count} > {maximum}")


def should_run_act0r1(gates: Mapping[str, Mapping]) -> bool:
    for name in HEALTH_GATES:
        status = gates.get(name, {}).get("status")
        if name == "ROOT_CAUSE_CLASSIFIED":
            if status not in {"PASS", "CONDITIONAL_PASS"}:
                return False
        elif status != "PASS":
            return False
    return True


def rgb_entropy(rgb: np.ndarray) -> float:
    image = np.asarray(rgb, dtype=np.uint8)
    gray = np.rint(image[..., :3].mean(axis=2)).astype(np.uint8)
    counts = np.bincount(gray.ravel(), minlength=256).astype(float)
    probabilities = counts[counts > 0] / max(float(counts.sum()), 1.0)
    return float(-(probabilities * np.log2(probabilities)).sum())


def unique_rgb_colors(rgb: np.ndarray) -> int:
    image = np.asarray(rgb, dtype=np.uint8)[..., :3]
    packed = (image[..., 0].astype(np.uint32) << 16) | (image[..., 1].astype(np.uint32) << 8) | image[..., 2]
    return int(np.unique(packed).size)


def ssim_rgb(left: np.ndarray, right: np.ndarray) -> float:
    """Compute standard global SSIM on luminance without optional dependencies."""
    a = np.asarray(left, dtype=np.float64)[..., :3].mean(axis=2)
    b = np.asarray(right, dtype=np.float64)[..., :3].mean(axis=2)
    if a.shape != b.shape:
        raise ValueError("SSIM inputs must have the same shape")
    mu_a, mu_b = float(a.mean()), float(b.mean())
    var_a, var_b = float(a.var()), float(b.var())
    covariance = float(((a - mu_a) * (b - mu_b)).mean())
    c1, c2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2
    denominator = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    return float(((2.0 * mu_a * mu_b + c1) * (2.0 * covariance + c2)) /
                 max(denominator, 1e-12))


def repeated_tile_similarity(rgb: np.ndarray, rows: int = 4, columns: int = 6) -> dict:
    """Report suspicious non-neighbor tile repetition as a diagnostic, not a label."""
    image = np.asarray(rgb, dtype=np.float64)[..., :3].mean(axis=2)
    height, width = image.shape
    tiles = []
    for row in range(rows):
        for column in range(columns):
            tile = image[row * height // rows:(row + 1) * height // rows,
                         column * width // columns:(column + 1) * width // columns]
            tile = tile - tile.mean()
            norm = float(np.linalg.norm(tile))
            tiles.append((row, column, tile / max(norm, 1e-12)))
    values = []
    for index, (row_a, col_a, tile_a) in enumerate(tiles):
        for row_b, col_b, tile_b in tiles[index + 1:]:
            if abs(row_a - row_b) <= 1 and abs(col_a - col_b) <= 1:
                continue
            if tile_a.shape == tile_b.shape:
                values.append(float(np.sum(tile_a * tile_b)))
    return {"pair_count": len(values),
            "max_similarity": max(values, default=0.0),
            "fraction_above_0_98": float(sum(value >= 0.98 for value in values) /
                                         max(len(values), 1))}


def evaluate_rgb_sequence(images: Sequence[np.ndarray], thresholds: Mapping,
                          historical: np.ndarray | None = None) -> dict:
    if not images:
        return {"status": "FAIL", "reason": "no RGB images"}
    entropies = [rgb_entropy(image) for image in images]
    colors = [unique_rgb_colors(image) for image in images]
    consecutive = [ssim_rgb(left, right) for left, right in zip(images, images[1:])]
    historical_ssim = ssim_rgb(images[0], historical) if historical is not None else None
    repetition = [repeated_tile_similarity(image) for image in images]
    passed = (min(entropies) >= float(thresholds["min_entropy_bits"]) and
              min(colors) >= int(thresholds["min_unique_colors"]) and
              (not consecutive or min(consecutive) >= float(thresholds["min_consecutive_ssim"])) and
              (historical_ssim is None or
               historical_ssim >= float(thresholds["min_historical_ssim"])))
    return {"status": "PASS" if passed else "FAIL", "entropy_bits": entropies,
            "unique_colors": colors, "consecutive_ssim": consecutive,
            "historical_ssim": historical_ssim,
            "tile_repetition_diagnostic": repetition,
            "thresholds": dict(thresholds)}


def evaluate_motion_sequence_rgb(images: Sequence[np.ndarray],
                                 same_pose_groups: Sequence[Sequence[int]],
                                 thresholds: Mapping) -> dict:
    """Gate image integrity without comparing deliberately different poses.

    Entropy, color diversity and tile-repetition diagnostics apply to every
    frame. The consecutive-SSIM threshold applies only within explicitly
    identified frozen-pose groups; cross-pose SSIM remains diagnostic.
    """
    diagnostic_thresholds = dict(thresholds)
    diagnostic_thresholds["min_consecutive_ssim"] = -1.0
    result = evaluate_rgb_sequence(images, diagnostic_thresholds)
    movement_ssim = result.pop("consecutive_ssim")
    same_pose_ssim = []
    checked_groups = []
    for group in same_pose_groups:
        indices = [int(index) for index in group]
        if len(indices) < 2:
            continue
        values = [ssim_rgb(images[left], images[right])
                  for left, right in zip(indices, indices[1:])]
        same_pose_ssim.extend(values)
        checked_groups.append({"indices": indices, "consecutive_ssim": values})
    threshold = float(thresholds["min_consecutive_ssim"])
    passed = result["status"] == "PASS" and bool(same_pose_ssim) and \
        min(same_pose_ssim) >= threshold
    result.update({
        "status": "PASS" if passed else "FAIL",
        "all_frame_consecutive_ssim_diagnostic": movement_ssim,
        "same_pose_groups": checked_groups,
        "same_pose_consecutive_ssim": same_pose_ssim,
        "threshold_application": "consecutive SSIM is gated only within frozen-pose groups",
        "thresholds": dict(thresholds),
    })
    return result


def classify_root_cause(tests: Mapping[str, Mapping], as_probe: Mapping | None = None) -> dict:
    def healthy(name):
        row = tests.get(name, {})
        return row.get("capture_status") == "PASS" and row.get("rgb_integrity", {}).get("status") == "PASS"

    h1, h2, h3, h4, h5 = (healthy(name) for name in ("H1", "H2", "H3", "H4", "H5"))
    if as_probe and h1 and as_probe.get("status") == "FAIL":
        return {"status": "PASS", "classification": "PYTHON_ADDRESS_SPACE_LIMIT_FAILURE",
                "confidence": "CONFIRMED", "reason": "4 GiB OLD RGB passed and 2 GiB OLD RGB failed"}
    if h1 and not h2:
        return {"status": "PASS", "classification": "POSE_OR_GEOMETRY_FAILURE",
                "confidence": "HIGH", "reason": "OLD RGB passed while NEW RGB failed"}
    if h1 and h2 and not h3 and not h4:
        return {"status": "PASS", "classification": "QUARTET_SENSOR_LOAD_FAILURE",
                "confidence": "HIGH", "reason": "RGB-only passed while both quartet tests failed"}
    if h3 and h4 and not h5:
        return {"status": "PASS", "classification": "POST_TELEPORT_FAILURE",
                "confidence": "HIGH", "reason": "fresh quartets passed and post-teleport failed"}
    if all((h1, h2, h3, h4, h5)):
        return {"status": "CONDITIONAL_PASS",
                "classification": "LOCATOR_TO_QUARTET_LIFECYCLE_FAILURE",
                "confidence": "CONDITIONAL",
                "reason": "fresh diagnostic stack passed; historical locator-to-quartet run failed"}
    return {"status": "FAIL", "classification": "UNRESOLVED", "confidence": "LOW",
            "reason": "diagnostic matrix does not isolate one cause"}
