"""Reusable GEO-0.6 sweep planning and independent trajectory metrics."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from .surfaces import physical_corners, surface_axes


DIRECTIONS = {
    "LEFT": ("LEFT", -1.0, "horizontal"),
    "RIGHT": ("RIGHT", 1.0, "horizontal"),
    "UP": ("TOP", 1.0, "vertical"),
    "DOWN": ("BOTTOM", -1.0, "vertical"),
}
STATE_RANK = {"IN": 0, "STRADDLE": 1, "OUT": 2}


def surface_dimensions(surface: Mapping) -> tuple[float, float]:
    corners = physical_corners(surface)
    return float(np.linalg.norm(corners[1] - corners[0])), float(np.linalg.norm(corners[0] - corners[2]))


def required_distance(surface: Mapping, width: int = 640, height: int = 480,
                      fov_deg: float = 90.0, step_m: float = 1.0,
                      in_frames: int = 5) -> float:
    """Choose a close distance so the centered facade begins in ``IN``."""
    facade_w, facade_h = surface_dimensions(surface)
    half_fov = math.tan(math.radians(fov_deg) / 2.0)
    horizontal = (facade_w - 2.0 * in_frames * step_m) / (2.0 * half_fov)
    vertical = (facade_h - 2.0 * in_frames * step_m) / (2.0 * half_fov * height / width)
    return float(0.80 * max(min(horizontal, vertical), 2.5))


def plan_sweep(surface: Mapping, direction: str, distance_m: float,
               width: int = 640, height: int = 480, fov_deg: float = 90.0,
               step_m: float = 1.0, max_steps: int = 80) -> dict:
    if direction not in DIRECTIONS:
        raise ValueError(f"unsupported GEO-0.6 direction: {direction}")
    facade_w, facade_h = surface_dimensions(surface)
    footprint_w = 2.0 * distance_m * math.tan(math.radians(fov_deg) / 2.0)
    footprint_h = footprint_w * height / width
    axis_size = facade_w if DIRECTIONS[direction][2] == "horizontal" else facade_h
    travel = 0.5 * (axis_size + (footprint_w if DIRECTIONS[direction][2] == "horizontal" else footprint_h)) + 2.0 * step_m
    steps = int(math.ceil(travel / step_m)) + 1
    actual_step = float(step_m)
    if steps > max_steps:
        actual_step = float(travel / (max_steps - 1))
        steps = max_steps
    return {
        "direction": direction,
        "active_boundary": DIRECTIONS[direction][0],
        "axis": DIRECTIONS[direction][2],
        "distance_m": float(distance_m),
        "facade_width_m": facade_w,
        "facade_height_m": facade_h,
        "image_footprint_width_m": float(footprint_w),
        "image_footprint_height_m": float(footprint_h),
        "step_m": actual_step,
        "steps": steps,
        "travel_m": float(travel),
        "thresholds": {"min_in_frames": 5, "min_straddle_frames": 5, "min_out_frames": 5},
    }


def _rank_correlation(values: Sequence[float]) -> float:
    if len(values) < 2 or len(set(values)) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.asarray(values, dtype=float)
    xr = np.argsort(np.argsort(x)).astype(float)
    yr = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(xr, yr)[0, 1])


def trajectory_metrics(frames: Sequence[Mapping]) -> dict:
    states = [str(frame.get("labels", {}).get("label", "UNKNOWN")) for frame in frames]
    counts = {state: states.count(state) for state in ("IN", "STRADDLE", "OUT", "UNKNOWN")}
    filtered = [state for state in states if state != "UNKNOWN"]
    bad = [(a, b) for a, b in zip(filtered, filtered[1:]) if STATE_RANK.get(a, 99) > STATE_RANK.get(b, 99)]
    overlap = [float(frame.get("labels", {}).get("target_pixel_coverage", 0.0)) for frame in frames]
    outward_overlap = [1.0 - value for value in overlap]
    reverse_jumps = [max(0.0, a - b) for a, b in zip(outward_overlap, outward_overlap[1:])]
    straddle_inside = all(bool(frame.get("labels", {}).get("boundary", {}).get("boundary_in_image"))
                          for frame in frames if frame.get("labels", {}).get("label") == "STRADDLE")
    return {
        "frames": len(frames),
        "state_counts": counts,
        "compressed_state_sequence": [state for i, state in enumerate(states) if i == 0 or state != states[i - 1]],
        "unknown_ratio": counts["UNKNOWN"] / max(len(frames), 1),
        "monotonic_ignoring_unknown": not bad,
        "bad_transitions": bad,
        "outward_overlap_spearman": _rank_correlation(outward_overlap),
        "max_reverse_overlap_jump": max(reverse_jumps, default=0.0),
        "straddle_boundary_inside": straddle_inside,
        "event_coverage": all(counts[state] >= 5 for state in ("IN", "STRADDLE", "OUT")),
    }
