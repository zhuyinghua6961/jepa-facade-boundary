"""CARLA semantic and instance camera decoding plus mask utilities."""

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
    """Decode CARLA's 24-bit instance ID from RGB little-endian bytes."""
    arr = image_or_bgra if isinstance(image_or_bgra, np.ndarray) else bgra_array(image_or_bgra)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError("expected HxWx4 BGRA data")
    r = arr[..., 2].astype(np.uint32)
    g = arr[..., 1].astype(np.uint32)
    b = arr[..., 0].astype(np.uint32)
    return r | (g << 8) | (b << 16)


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
