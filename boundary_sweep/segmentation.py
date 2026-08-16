"""CARLA semantic/instance decoding and mask geometry utilities.

CARLA stores semantic tag in R and the 16-bit instance id in G/B. The packed
semantic-instance key is kept as a separate API.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping

import numpy as np


BUILDING_TAG = 3


def bgra_array(image) -> np.ndarray:
    """Return raw CARLA image bytes as an HxWx4 BGRA uint8 array."""
    return np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))


def decode_semantic_tag(image_or_bgra) -> np.ndarray:
    """Decode semantic tags from CARLA raw BGRA; the RGB red byte is index 2."""
    arr = image_or_bgra if isinstance(image_or_bgra, np.ndarray) else bgra_array(image_or_bgra)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError("expected HxWx4 BGRA data")
    return arr[..., 2].astype(np.uint8, copy=False)


def decode_instance_id(image_or_bgra) -> np.ndarray:
    """Decode CARLA's 16-bit instance id: G | (B << 8)."""
    arr = image_or_bgra if isinstance(image_or_bgra, np.ndarray) else bgra_array(image_or_bgra)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError("expected HxWx4 BGRA data")
    g = arr[..., 1].astype(np.uint32)
    b = arr[..., 0].astype(np.uint32)
    return g | (b << 8)


def decode_packed_semantic_instance_key(image_or_bgra) -> np.ndarray:
    """Decode the index key R | (G << 8) | (B << 16)."""
    arr = image_or_bgra if isinstance(image_or_bgra, np.ndarray) else bgra_array(image_or_bgra)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError("expected HxWx4 BGRA data")
    r = arr[..., 2].astype(np.uint32)
    g = arr[..., 1].astype(np.uint32)
    b = arr[..., 0].astype(np.uint32)
    return r | (g << 8) | (b << 16)


def decode_instance_channels(image_or_rgb) -> dict[str, np.ndarray]:
    """Return semantic tag, 16-bit id and packed key from BGRA or RGB data."""
    arr = image_or_rgb if isinstance(image_or_rgb, np.ndarray) else bgra_array(image_or_rgb)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError("expected HxWx3/4 image data")
    if arr.shape[-1] == 4:
        r, g, b = arr[..., 2], arr[..., 1], arr[..., 0]
    else:
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    r = r.astype(np.uint32, copy=False)
    g = g.astype(np.uint32, copy=False)
    b = b.astype(np.uint32, copy=False)
    return {
        "semantic_tag": r.astype(np.uint8, copy=False),
        "instance_id_16bit": g | (b << 8),
        "packed_semantic_instance_key": r | (g << 8) | (b << 16),
    }


def semantic_instance_consistency(semantic_rgb: np.ndarray, instance_rgb: np.ndarray) -> dict:
    """Compare semantic-camera R with instance-camera R pixel by pixel."""
    semantic = decode_instance_channels(semantic_rgb)["semantic_tag"]
    instance = decode_instance_channels(instance_rgb)["semantic_tag"]
    if semantic.shape != instance.shape:
        raise ValueError("semantic and instance image shapes differ")
    mismatch = semantic != instance
    ys, xs = np.where(mismatch)
    examples = [{"x": int(x), "y": int(y), "semantic": int(semantic[y, x]),
                 "instance": int(instance[y, x])}
                for y, x in zip(ys[:10], xs[:10])]
    total = int(mismatch.size)
    errors = int(mismatch.sum())
    return {"pixel_count": total, "matching_pixels": total - errors,
            "error_pixels": errors, "agreement": float((total - errors) / max(total, 1)),
            "examples": examples}


def rgb_from_bgra(bgra: np.ndarray) -> np.ndarray:
    """Convert encoded BGRA bytes to RGB without changing channel values."""
    return np.asarray(bgra)[..., [2, 1, 0]].copy()


def inventory(instance_ids: np.ndarray, semantic_tags: np.ndarray | None = None,
              building_tag: int = BUILDING_TAG, limit: int | None = None) -> list[dict]:
    """Return instance counts, optionally restricted to Building pixels."""
    ids = np.asarray(instance_ids)
    if semantic_tags is not None:
        ids = ids[np.asarray(semantic_tags) == int(building_tag)]
    counts = Counter(int(value) for value in ids.ravel() if int(value) != 0)
    return [{"instance_id": key, "pixel_count": value} for key, value in counts.most_common(limit)]


def mask_for_ids(instance_ids: np.ndarray, ids: Iterable[int]) -> np.ndarray:
    values = np.asarray(list(ids), dtype=np.uint32)
    if values.size == 0:
        return np.zeros(np.asarray(instance_ids).shape, dtype=bool)
    return np.isin(np.asarray(instance_ids), values)


def binary_close_holes(mask: np.ndarray, max_kernel_px: int = 3) -> np.ndarray:
    """Fill enclosed holes with a configurable closing operation of at most 3px."""
    if max_kernel_px < 0 or max_kernel_px > 3:
        raise ValueError("max_kernel_px must be between 0 and 3")
    source = np.asarray(mask, dtype=bool)
    if max_kernel_px == 0:
        return source.copy()
    try:
        from scipy import ndimage
    except ImportError:
        return source.copy()
    structure = np.ones((2 * max_kernel_px + 1, 2 * max_kernel_px + 1), dtype=bool)
    closed = ndimage.binary_closing(source, structure=structure)
    filled = ndimage.binary_fill_holes(closed)
    observed = ndimage.binary_dilation(source, structure=np.ones((3, 3), dtype=bool))
    return np.asarray((filled & observed) | source, dtype=bool)


def mask_metrics(target_mask: np.ndarray, envelope_mask: np.ndarray, image_shape: tuple[int, int]) -> dict:
    total = max(int(image_shape[0] * image_shape[1]), 1)
    target_count = int(np.count_nonzero(target_mask))
    envelope_count = int(np.count_nonzero(envelope_mask))
    return {
        "target_mask_pixels": target_count,
        "target_mask_coverage": float(target_count / total),
        "envelope_pixels": envelope_count,
        "envelope_coverage": float(envelope_count / total),
        "hole_filled_pixels": int(max(envelope_count - target_count, 0)),
    }


def choose_center_ids(instance_ids: np.ndarray, semantic_tags: np.ndarray,
                      building_tag: int = BUILDING_TAG, fraction: float = 0.30) -> list[int]:
    """Choose candidate IDs from the geometry-seeded central region."""
    h, w = instance_ids.shape[:2]
    y0, y1 = int(h * (0.5 - fraction / 2)), int(h * (0.5 + fraction / 2))
    x0, x1 = int(w * (0.5 - fraction / 2)), int(w * (0.5 + fraction / 2))
    return [row["instance_id"] for row in inventory(instance_ids[y0:y1, x0:x1],
                                                     semantic_tags[y0:y1, x0:x1], building_tag)]


def stable_id_intersection(rows: Iterable[Mapping], min_views: int) -> list[int]:
    counts = Counter()
    for row in rows:
        counts.update(int(value) for value in row.get("candidate_ids", []))
    return sorted(key for key, value in counts.items() if value >= int(min_views))


def largest_connected_component_ratio(mask: np.ndarray) -> float:
    """Return largest 8-connected component divided by all foreground pixels."""
    source = np.asarray(mask, dtype=bool)
    total = int(source.sum())
    if total == 0:
        return 0.0
    try:
        from scipy import ndimage
        labels, count = ndimage.label(source, structure=np.ones((3, 3), dtype=bool))
        sizes = np.bincount(labels.ravel())[1:]
        return float(sizes.max() / total) if count and sizes.size else 0.0
    except ImportError:
        return 1.0


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left, dtype=bool), np.asarray(right, dtype=bool)
    union = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b) / max(union, 1))


def outer_transition_contour(target_mask: np.ndarray) -> np.ndarray:
    """Extract target/non-target transitions whose non-target side reaches frame edge.

    Enclosed holes and image-border truncation are excluded.
    """
    target = np.asarray(target_mask, dtype=bool)
    if target.ndim != 2:
        raise ValueError("target_mask must be two-dimensional")
    try:
        from scipy import ndimage
    except ImportError:
        return np.zeros_like(target)
    labels, count = ndimage.label(~target, structure=np.ones((3, 3), dtype=bool))
    if count == 0:
        return np.zeros_like(target)
    edge_labels = np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    edge_labels = edge_labels[edge_labels > 0]
    exterior = np.isin(labels, edge_labels)
    contour = target & ndimage.binary_dilation(exterior, structure=np.ones((3, 3), dtype=bool))
    contour[[0, -1], :] = False
    contour[:, [0, -1]] = False
    return contour
