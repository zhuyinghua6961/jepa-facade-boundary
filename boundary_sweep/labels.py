"""GEO-0.5R2 boundary labels.

The active horizontal boundary is the only state label for LEFT/RIGHT tracks.
Visibility is measured on a dense projected facade mask, while target coverage
is measured against the full image area.  These metrics are deliberately
separate so an occluded facade cannot be mislabeled OUT.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image, ImageDraw

from .geometry import camera_to_world, pixel_to_camera_ray, ray_plane_intersection, world_to_camera, world_to_pixel
from .surfaces import boundary_line, physical_corners, surface_axes


def _inside(uv, width, height):
    return bool(0 <= float(uv[0]) < width and 0 <= float(uv[1]) < height)


def _segment_intersects_view(a, b, width, height):
    dx, dy = float(b[0] - a[0]), float(b[1] - a[1])
    p = (-dx, dx, -dy, dy)
    q = (float(a[0]), float(width - 1 - a[0]), float(a[1]), float(height - 1 - a[1]))
    lo, hi = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return False
            continue
        r = qi / pi
        if pi < 0:
            lo = max(lo, r)
        else:
            hi = min(hi, r)
        if lo > hi:
            return False
    return True


def _depth_patch(depth_m, uv, radius=1):
    x, y = np.rint(uv).astype(int)
    h, w = depth_m.shape
    if x < 0 or y < 0 or x >= w or y >= h:
        return None
    patch = depth_m[max(0, y - radius):min(h, y + radius + 1), max(0, x - radius):min(w, x + radius + 1)]
    valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 1000.0)]
    return float(np.median(valid)) if valid.size else None


def depth_tolerance(theoretical_z: float) -> float:
    return max(0.15, 0.02 * float(theoretical_z))


def _projected_polygon(surface, camera_transform, K, width, height):
    corners = physical_corners(surface)
    projected = world_to_pixel(corners, camera_transform, K)
    if not np.isfinite(projected).all():
        return projected[:, :2], None
    # physical_corners is TL,TR,BL,BR; polygon winding is TL,TR,BR,BL.
    return projected[[0, 1, 3, 2], :2], projected[[0, 1, 3, 2], 2]


def _pixel_plane_point(pixel, camera_transform, K, plane_origin, plane_normal):
    ray_camera = pixel_to_camera_ray(pixel, K)
    camera_origin = camera_to_world([0.0, 0.0, 0.0], camera_transform)
    ray_world = camera_to_world(ray_camera, camera_transform) - camera_origin
    return ray_plane_intersection(camera_origin, ray_world, plane_origin, plane_normal)


def dense_target_samples(surface: Mapping, camera_transform, K: np.ndarray, depth_m: np.ndarray,
                         width: int, height: int, pixel_step: int = 4) -> dict:
    """Sample every <=4 pixels inside the projected physical facade polygon."""
    if pixel_step < 1 or pixel_step > 4:
        raise ValueError("pixel_step must be in [1,4]")
    polygon, _ = _projected_polygon(surface, camera_transform, K, width, height)
    finite = np.isfinite(polygon).all()
    samples = []
    visible = 0
    projected_count = 0
    if not finite:
        return {"samples": samples, "visible_projected_target_pixels": 0,
                "projected_target_pixels": 0, "occlusion_visibility_ratio": 0.0,
                "target_pixel_coverage": 0.0}
    x0 = max(0, int(np.floor(np.min(polygon[:, 0]))))
    x1 = min(width - 1, int(np.ceil(np.max(polygon[:, 0]))))
    y0 = max(0, int(np.floor(np.min(polygon[:, 1]))))
    y1 = min(height - 1, int(np.ceil(np.max(polygon[:, 1]))))
    # Convex quadrilateral test in pixel coordinates.
    for y in range(y0, y1 + 1, pixel_step):
        for x in range(x0, x1 + 1, pixel_step):
            p = np.array([x + 0.5, y + 0.5], dtype=float)
            cross = []
            for i in range(4):
                a, b = polygon[i], polygon[(i + 1) % 4]
                cross.append(float(np.cross(b - a, p - a)))
            if not (all(c >= -1e-6 for c in cross) or all(c <= 1e-6 for c in cross)):
                continue
            world = _pixel_plane_point(p, camera_transform, K,
                                       np.asarray(surface["plane_origin"], dtype=float),
                                       np.asarray(surface["plane_normal"], dtype=float))
            if world is None:
                continue
            projected_count += 1
            cam = world_to_camera(world, camera_transform)
            measured = _depth_patch(depth_m, p)
            valid = measured is not None
            match = bool(valid and abs(float(measured) - float(cam[2])) <= depth_tolerance(cam[2]))
            visible += int(match)
            samples.append({"pixel": [float(p[0]), float(p[1])], "world": np.asarray(world).tolist(),
                            "theoretical_z": float(cam[2]), "sensor_depth": measured,
                            "depth_match": match, "valid_depth": valid})
    return {"samples": samples, "visible_projected_target_pixels": int(visible),
            "projected_target_pixels": int(projected_count),
            "occlusion_visibility_ratio": float(visible / max(projected_count, 1)),
            "target_pixel_coverage": float(visible * pixel_step * pixel_step / max(width * height, 1))}


def _probe(surface, boundary_name, camera_transform, K, depth_m, width, height, amount=0.5):
    corners = physical_corners(surface)
    center = corners.mean(axis=0)
    h, v, _ = surface_axes(surface)
    width_m = float(np.linalg.norm(corners[1] - corners[0]))
    height_m = float(np.linalg.norm(corners[0] - corners[2]))
    offsets = {"LEFT": (amount, height_m / 2, -amount, height_m / 2),
               "RIGHT": (width_m - amount, height_m / 2, width_m + amount, height_m / 2),
               "TOP": (width_m / 2, height_m - amount, width_m / 2, height_m + amount),
               "BOTTOM": (width_m / 2, amount, width_m / 2, -amount)}
    iu, iv, ou, ov = offsets[boundary_name]
    # plane_origin is a fit centroid, while probe coordinates are measured
    # from the physical bottom-left corner of the terminal surface.
    origin = corners[2]
    target = origin + h * iu + v * iv
    external = origin + h * ou + v * ov
    target_uvz = world_to_pixel(target, camera_transform, K)
    external_uvz = world_to_pixel(external, camera_transform, K)

    def observe(point, uvz, require_target):
        in_image = bool(np.isfinite(uvz).all() and uvz[2] > 0 and _inside(uvz[:2], width, height))
        depth = _depth_patch(depth_m, uvz[:2]) if in_image else None
        valid = depth is not None
        match = bool(valid and abs(float(depth) - float(uvz[2])) <= depth_tolerance(uvz[2]))
        return {"pixel": uvz[:2].tolist() if np.isfinite(uvz).all() else [None, None],
                "in_image": in_image, "depth_valid": valid, "target_plane_match": match,
                "target_side_observed": bool(in_image and valid and (match if require_target else True)),
                "external_side_observed": bool(in_image and valid),
                "external_side_is_not_target": bool(in_image and valid and not match)}

    return observe(target, target_uvz, True), observe(external, external_uvz, False)


def classify_boundary(surface, active_boundary: str, dense: Mapping, boundary_pixel, probes: Mapping,
                     width: int, height: int, central_target: bool) -> dict:
    if active_boundary not in ("LEFT", "RIGHT", "TOP", "BOTTOM"):
        raise ValueError("invalid active boundary")
    interior = bool(np.isfinite(boundary_pixel).all() and
                    _segment_intersects_view(boundary_pixel[0], boundary_pixel[1], width, height))
    coverage = float(dense["target_pixel_coverage"])
    projected = int(dense["projected_target_pixels"])
    target = probes["target"]
    external = probes["external"]
    exit_confirmed = bool(projected == 0 and not interior and not central_target)
    if not interior and float(dense["occlusion_visibility_ratio"]) >= 0.95 and central_target:
        label = "IN"
    elif interior and target["target_side_observed"] and external["external_side_observed"] and external["external_side_is_not_target"]:
        label = "STRADDLE"
    elif coverage <= 0.05 and exit_confirmed:
        label = "OUT"
    else:
        label = "UNKNOWN"
    return {"label": label, "boundary_in_image": interior, "target_pixel_coverage": coverage,
            "occlusion_visibility_ratio": float(dense["occlusion_visibility_ratio"]),
            "projected_target_pixels": projected, "target_exit_confirmed": exit_confirmed,
            "target_side_observed": target["target_side_observed"],
            "external_side_observed": external["external_side_observed"],
            "external_side_is_not_target": external["external_side_is_not_target"]}


def generate_frame_label(surface: Mapping, active_boundary: str, camera_transform, K: np.ndarray,
                         depth_m: np.ndarray, width: int, height: int, pixel_step: int = 4,
                         overlay_path: str | Path | None = None, rgb_path: str | Path | None = None) -> dict:
    dense = dense_target_samples(surface, camera_transform, K, depth_m, width, height, pixel_step)
    center_world = np.mean(physical_corners(surface), axis=0)
    center_uvz = world_to_pixel(center_world, camera_transform, K)
    center_depth = _depth_patch(depth_m, center_uvz[:2]) if np.isfinite(center_uvz).all() else None
    central_target = bool(np.isfinite(center_uvz).all() and _inside(center_uvz[:2], width, height) and
                          center_depth is not None and abs(center_depth - center_uvz[2]) <= depth_tolerance(center_uvz[2]))
    boundaries = {}
    for name in ("LEFT", "RIGHT", "TOP", "BOTTOM"):
        world_line = boundary_line(surface, name)
        projected = world_to_pixel(world_line, camera_transform, K)
        target_probe, external_probe = _probe(surface, name, camera_transform, K, depth_m, width, height)
        probes = {"target": target_probe, "external": external_probe}
        boundaries[name] = {"boundary_type": name, "boundary_world_line": world_line.tolist(),
                            "boundary_pixel_line": projected[:, :2].tolist(),
                            "probes": probes}
        if name == active_boundary:
            boundaries[name].update(classify_boundary(surface, name, dense, projected[:, :2], probes,
                                                      width, height, central_target))
    result = {"surface_id": surface.get("surface_id"), "active_boundary": active_boundary,
              "label": boundaries[active_boundary]["label"], "occlusion_visibility_ratio": dense["occlusion_visibility_ratio"],
              "target_pixel_coverage": dense["target_pixel_coverage"],
              "visible_projected_target_pixels": dense["visible_projected_target_pixels"],
              "projected_target_pixels": dense["projected_target_pixels"],
              "central_target": central_target, "samples": dense["samples"], "boundary": boundaries[active_boundary],
              "auxiliary_boundaries": {k: v for k, v in boundaries.items() if k != active_boundary}}
    if overlay_path and rgb_path:
        render_overlay(rgb_path, result, overlay_path)
    return result


def render_overlay(rgb_path: str | Path, labels: Mapping, output_path: str | Path):
    image = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for sample in labels.get("samples", []):
        p = sample.get("pixel")
        if p and sample.get("depth_match"):
            draw.ellipse((p[0] - 1, p[1] - 1, p[0] + 1, p[1] + 1), fill=(30, 220, 70))
        elif p:
            draw.ellipse((p[0] - 1, p[1] - 1, p[0] + 1, p[1] + 1), fill=(220, 50, 50))
    colors = {"IN": (40, 210, 80), "STRADDLE": (255, 185, 0), "OUT": (230, 50, 50), "UNKNOWN": (180, 180, 180)}
    for name, item in [(labels["active_boundary"], labels["boundary"]), *labels.get("auxiliary_boundaries", {}).items()]:
        points = np.asarray(item["boundary_pixel_line"], dtype=float)
        if np.isfinite(points).all():
            draw.line([tuple(points[0]), tuple(points[1])], fill=colors.get(item.get("label", "UNKNOWN"), (180, 180, 180)), width=4 if name == labels["active_boundary"] else 2)
            draw.text(tuple(points[0]), f"{name}:{item.get('label', 'AUX')}", fill=colors.get(item.get("label", "UNKNOWN"), (180, 180, 180)))
    draw.text((8, 8), f"active={labels['active_boundary']} label={labels['label']} coverage={labels['target_pixel_coverage']:.3f} occ={labels['occlusion_visibility_ratio']:.3f}", fill=(255, 255, 0))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def facade_outer_envelope(target_mask, closing_kernel_px: int = 3):
    """Build an outer envelope by filling enclosed internal holes only."""
    from .segmentation import binary_close_holes
    return binary_close_holes(target_mask, closing_kernel_px)


def classify_mask_state(target_mask, envelope_mask, boundary_pixel_line, target_center_pixel,
                        stable_target_ids: bool, thresholds=None):
    """Classify a key pose from simulator masks and a projected physical edge."""
    thresholds = thresholds or {}
    in_min = float(thresholds.get("in_envelope_coverage", 0.50))
    out_max = float(thresholds.get("out_envelope_coverage", 0.05))
    side_min = float(thresholds.get("straddle_side_fraction", 0.05))
    target_mask = np.asarray(target_mask, dtype=bool)
    envelope_mask = np.asarray(envelope_mask, dtype=bool)
    h, w = target_mask.shape
    total = float(max(h * w, 1))
    target_cov = float(np.count_nonzero(target_mask) / total)
    envelope_cov = float(np.count_nonzero(envelope_mask) / total)
    line = np.asarray(boundary_pixel_line, dtype=float)
    center = np.asarray(target_center_pixel, dtype=float)
    valid_line = line.shape == (2, 2) and np.isfinite(line).all() and np.isfinite(center).all()
    d = line[1] - line[0] if valid_line else np.zeros(2)
    cross_center = float(d[0] * (center[1] - line[0, 1]) - d[1] * (center[0] - line[0, 0])) if valid_line else 0.0
    yy, xx = np.indices((h, w), dtype=float)
    signed = d[0] * (yy - line[0, 1]) - d[1] * (xx - line[0, 0]) if valid_line else np.zeros((h, w))
    target_side = signed * (1.0 if cross_center >= 0 else -1.0) >= 0
    target_side_pixels = int(np.count_nonzero(target_mask & target_side))
    external_side_pixels = int(np.count_nonzero((~envelope_mask) & (~target_side)))
    boundary_inside = bool(valid_line and _segment_intersects_view(line[0], line[1], w, h))
    central = envelope_mask[int(h * 0.30):int(h * 0.70), int(w * 0.30):int(w * 0.70)]
    central_target = bool(central.size and np.mean(central) >= 0.25)
    if not stable_target_ids or not valid_line:
        label = "UNKNOWN"
    elif envelope_cov <= out_max:
        label = "OUT"
    elif boundary_inside and target_side_pixels / total >= side_min and external_side_pixels / total >= side_min:
        label = "STRADDLE"
    elif not boundary_inside and envelope_cov >= in_min and central_target:
        label = "IN"
    else:
        label = "UNKNOWN"
    return {"label": label, "target_pixel_coverage": target_cov, "envelope_coverage": envelope_cov,
            "boundary_in_image": boundary_inside, "central_target": central_target,
            "target_side_pixels": target_side_pixels, "external_side_pixels": external_side_pixels,
            "stable_target_ids": bool(stable_target_ids), "out_threshold": out_max,
            "in_threshold": in_min, "straddle_side_fraction": side_min}
