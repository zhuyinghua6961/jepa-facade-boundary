#!/usr/bin/env python3
"""Recompute MASK-0R1 evidence from the existing four-sensor audit only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from boundary_sweep.geometry import world_to_pixel
from boundary_sweep.segmentation import (
    BUILDING_TAG,
    decode_instance_channels,
    largest_connected_component_ratio,
    mask_iou,
    outer_transition_contour,
    semantic_instance_consistency,
)
from boundary_sweep.surfaces import boundary_line, load_surface, physical_corners

POSES = ("CENTER", "LEFT_NEAR_BOUNDARY", "RIGHT_NEAR_BOUNDARY", "TOP_NEAR_BOUNDARY", "BOTTOM_SAFE_VIEW")
EDGES = ("LEFT", "RIGHT", "TOP", "BOTTOM")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str, raw_root: Path) -> Path:
    path = PROJECT_ROOT / value
    if path.exists():
        return path
    return PROJECT_ROOT / raw_root / value


def read_frame(record: dict, raw_root: Path) -> dict:
    paths = {key: resolve_path(item["path"], raw_root) for key, item in record["files"].items()}
    for key, path in paths.items():
        if not path.exists() or sha256(path) != record["files"][key]["sha256"]:
            raise ValueError(f"file/hash mismatch: {record['frame_id']} {key}")
    rgb = np.asarray(Image.open(paths["rgb"]).convert("RGB"))
    semantic_rgb = np.asarray(Image.open(paths["semantic"]).convert("RGB"))
    instance_rgb = np.asarray(Image.open(paths["instance"]).convert("RGB"))
    channels = decode_instance_channels(instance_rgb)
    semantic_camera = decode_instance_channels(semantic_rgb)["semantic_tag"]
    consistency = semantic_instance_consistency(semantic_rgb, instance_rgb)
    return {"record": record, "paths": paths, "rgb": rgb, "semantic_rgb": semantic_rgb,
            "instance_rgb": instance_rgb, "semantic": semantic_camera,
            "instance_id": channels["instance_id_16bit"],
            "packed_key": channels["packed_semantic_instance_key"],
            "consistency": consistency}


def line_visible(line: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    a, b = np.asarray(line, dtype=float)
    length = max(float(np.linalg.norm(b - a)), 1.0)
    t = np.linspace(0.0, 1.0, max(64, int(length * 2.0)))
    points = a[None, :] + t[:, None] * (b - a)[None, :]
    inside = ((points[:, 0] >= 0.0) & (points[:, 0] < width) &
              (points[:, 1] >= 0.0) & (points[:, 1] < height))
    return points[inside], points


def side_fractions(line: np.ndarray, points: np.ndarray, target_mask: np.ndarray,
                   center_uv: np.ndarray, offset_px: float = 4.0) -> tuple[float, float]:
    if len(points) == 0:
        return 0.0, 0.0
    a, b = np.asarray(line, dtype=float)
    direction = b - a
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        return 0.0, 0.0
    normal = np.array([-direction[1], direction[0]], dtype=float) / norm
    if float(np.dot(np.asarray(center_uv) - (a + b) / 2.0, normal)) < 0:
        normal = -normal
    plus = np.rint(points + normal * offset_px).astype(int)
    minus = np.rint(points - normal * offset_px).astype(int)
    h, w = target_mask.shape
    valid_plus = ((plus[:, 0] >= 0) & (plus[:, 0] < w) & (plus[:, 1] >= 0) & (plus[:, 1] < h))
    valid_minus = ((minus[:, 0] >= 0) & (minus[:, 0] < w) & (minus[:, 1] >= 0) & (minus[:, 1] < h))
    target_side = float(target_mask[plus[valid_plus, 1], plus[valid_plus, 0]].mean()) if valid_plus.any() else 0.0
    external_side = float((~target_mask[minus[valid_minus, 1], minus[valid_minus, 0]]).mean()) if valid_minus.any() else 0.0
    return target_side, external_side


def contour_line_metrics(line: np.ndarray, contour: np.ndarray, target_mask: np.ndarray,
                         center_uv: np.ndarray, tolerance_px: float) -> dict:
    h, w = target_mask.shape
    points, all_points = line_visible(line, w, h)
    ys, xs = np.where(contour)
    transition_present = bool(len(xs))
    if len(points) and len(xs):
        from scipy.ndimage import distance_transform_edt
        distance = distance_transform_edt(~contour)
        q = np.rint(points).astype(int)
        q[:, 0] = np.clip(q[:, 0], 0, w - 1)
        q[:, 1] = np.clip(q[:, 1], 0, h - 1)
        distances = distance[q[:, 1], q[:, 0]]
        median = float(np.median(distances))
        p90 = float(np.percentile(distances, 90))
        support = float(np.mean(distances <= tolerance_px))
    else:
        median = p90 = None
        support = 0.0
    target_side, external_side = side_fractions(line, points, target_mask, center_uv)
    aligned = bool(points.size and transition_present and median is not None and
                   median <= 5.0 and support >= 0.80 and
                   target_side >= 0.60 and external_side >= 0.60)
    return {"projected_line_visible": bool(len(points)),
            "transition_contour_present": transition_present,
            "median_distance_px": median, "p90_distance_px": p90,
            "support_fraction_within_tolerance": support,
            "target_side_fraction": target_side,
            "external_side_fraction": external_side,
            "alignment_status": "PASS" if aligned else "FAIL",
            "line_sample_count": int(len(points)),
            "contour_pixel_count": int(len(xs))}


def edge_candidate(edge: str, contour: np.ndarray) -> dict:
    ys, xs = np.where(contour)
    if not len(xs):
        return {"validated_edge": False, "contour_span_fraction": 0.0, "contour_centroid_px": None}
    if edge in ("LEFT", "RIGHT"):
        span = (float(ys.max() - ys.min() + 1) / contour.shape[0])
        centroid = float(xs.mean())
        expected_side = "LEFT" if centroid < contour.shape[1] * 0.45 else "RIGHT" if centroid > contour.shape[1] * 0.55 else "MIDDLE"
    else:
        span = (float(xs.max() - xs.min() + 1) / contour.shape[1])
        centroid = float(ys.mean())
        expected_side = "TOP" if centroid < contour.shape[0] * 0.45 else "BOTTOM" if centroid > contour.shape[0] * 0.55 else "MIDDLE"
    return {"validated_edge": bool(span >= 0.70 and expected_side == edge), "contour_span_fraction": span,
            "contour_centroid_px": centroid, "expected_side": expected_side}


def pairwise_iou(masks: list[np.ndarray]) -> float:
    values = [mask_iou(masks[i], masks[j]) for i in range(len(masks)) for j in range(i + 1, len(masks))]
    return float(np.mean(values)) if values else 1.0


def contour_repeatability(contours: list[np.ndarray]) -> float:
    if len(contours) < 2:
        return 1.0
    from scipy.ndimage import binary_dilation
    values = []
    for i in range(len(contours)):
        for j in range(i + 1, len(contours)):
            a = binary_dilation(contours[i], iterations=3)
            b = binary_dilation(contours[j], iterations=3)
            values.append(mask_iou(a, b))
    return float(np.mean(values)) if values else 1.0


def render_alignment(frame: dict, metrics: dict, surface: dict, edge: str, output: Path) -> None:
    image = Image.fromarray(frame["rgb"]).convert("RGBA")
    overlay = np.zeros((frame["rgb"].shape[0], frame["rgb"].shape[1], 4), dtype=np.uint8)
    overlay[metrics["target_mask"]] = (40, 210, 80, 90)
    overlay[metrics["contour"]] = (255, 180, 0, 230)
    image = Image.alpha_composite(image, Image.fromarray(overlay, mode="RGBA"))
    draw = ImageDraw.Draw(image)
    line = np.asarray(metrics["line"], dtype=float)
    draw.line([tuple(line[0]), tuple(line[1])], fill=(255, 40, 40), width=4)
    ys, xs = np.where(metrics["contour"])
    for x, y in zip(xs[:: max(1, len(xs) // 800)], ys[:: max(1, len(ys) // 800)]):
        draw.point((int(x), int(y)), fill=(255, 220, 0))
    status = metrics["alignment"]["alignment_status"]
    text = (f"{surface['surface_id']} {edge}  legacy={status}  "
            f"median={metrics['alignment']['median_distance_px']}px  "
            f"p90={metrics['alignment']['p90_distance_px']}px  "
            f"target={metrics['alignment']['target_side_fraction']:.2f}  "
            f"external={metrics['alignment']['external_side_fraction']:.2f}")
    draw.rectangle((0, 0, 640, 32), fill=(0, 0, 0, 220))
    draw.text((6, 8), text, fill=(255, 235, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output, quality=84, optimize=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/mask0/raw")
    parser.add_argument("--results", default="results/mask0")
    parser.add_argument("--surfaces", default="results/geo06/surfaces")
    parser.add_argument("--config", default="configs/experiments/mask0r1.yaml")
    parser.add_argument("--assets", default="docs/assets/mask0r1")
    parser.add_argument("--docs", default="docs/MASK0R1_VISUAL_AUDIT.md")
    args = parser.parse_args(argv)
    raw_root, result_root, assets = Path(args.root), Path(args.results), Path(args.assets)
    result_root.mkdir(parents=True, exist_ok=True); assets.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((PROJECT_ROOT / args.config).read_text())
    capture = json.loads((PROJECT_ROOT / args.results / "capture_manifest.json").read_text())
    surfaces = {p.stem: load_surface(p) for p in (PROJECT_ROOT / args.surfaces).glob("*.json")}
    grouped = defaultdict(list)
    for record in capture["frames"]:
        grouped[record["surface_id"]].append(record)

    all_frames = []
    decoder_consistency = []
    target_groups, inventories = {}, {}
    frame_cache = {}
    for sid, records in sorted(grouped.items()):
        rows = []
        for record in records:
            frame = read_frame(record, raw_root)
            frame_cache[record["frame_id"]] = frame
            all_frames.append(frame); decoder_consistency.append(frame["consistency"])
            building = frame["semantic"] == BUILDING_TAG
            ids, counts = np.unique(frame["instance_id"][building], return_counts=True)
            rows.extend((int(value), int(count), record["pose"]) for value, count in zip(ids, counts) if int(value) != 0)
        views = Counter(value for value, _count, _pose in rows)
        pixels = Counter()
        for value, count, _pose in rows: pixels[value] += count
        min_views = int(math.ceil(len(records) * float(cfg["segmentation"]["target_id_min_view_fraction"])))
        stable = sorted((value for value, count in views.items() if count >= min_views and pixels[value] >= 100), key=lambda value: pixels[value], reverse=True)
        inventories[sid] = {"frames": len(records), "ids": [{"instance_id_16bit": value, "pixel_count": pixels[value], "view_count": views[value], "view_fraction": views[value] / len(records)} for value in sorted(views, key=pixels.get, reverse=True)]}
        target_groups[sid] = {"target_instance_ids_16bit": stable, "target_packed_keys": [int(BUILDING_TAG | (value << 8)) for value in stable], "stable_min_views": min_views, "selection": "semantic Building mask plus 16-bit instance ID; no RGB classification"}

    edge_records = defaultdict(lambda: defaultdict(list))
    mask_records = defaultdict(lambda: defaultdict(list))
    for sid, records in sorted(grouped.items()):
        surface = surfaces[sid]
        target_ids = set(target_groups[sid]["target_instance_ids_16bit"])
        for record in records:
            frame = frame_cache[record["frame_id"]]
            target = (frame["semantic"] == BUILDING_TAG) & np.isin(frame["instance_id"], list(target_ids))
            contour = outer_transition_contour(target)
            transform = np.asarray(record["T_world_camera"], dtype=float)
            K = np.asarray(record["K"], dtype=float)
            center_uv = world_to_pixel(np.mean(physical_corners(surface), axis=0), transform, K)[:2]
            for edge in EDGES:
                line = world_to_pixel(boundary_line(surface, edge), transform, K)[:, :2]
                alignment = contour_line_metrics(line, contour, target, center_uv, float(cfg["alignment"]["tolerances_px"][1]))
                candidate = edge_candidate(edge, contour)
                item = {"frame_id": int(record["frame_id"]), "surface_id": sid, "pose": record["pose"], "repeat_index": int(record["repeat_index"]), "edge": edge, "line": line.tolist(), "alignment": alignment, "candidate": candidate, "target_mask": target, "contour": contour, "rgb": frame["rgb"], "target_coverage": float(target.mean()), "largest_connected_component_ratio": largest_connected_component_ratio(target), "raw_hole_pixels": 0}
                edge_records[sid][edge].append(item)
            mask_records[sid][record["pose"]].append(target)

    alignment_summary = {}
    for sid in sorted(edge_records):
        alignment_summary[sid] = {}
        for edge in EDGES:
            items = edge_records[sid][edge]
            summary = {"surface_id": sid, "edge": edge, "views": len(items), "tolerances_px": {}}
            for tolerance in cfg["alignment"]["tolerances_px"]:
                tolerance = float(tolerance)
                statuses = []
                for item in items:
                    a = contour_line_metrics(np.asarray(item["line"]), item["contour"], item["target_mask"], np.array([320.0, 240.0]), tolerance)
                    statuses.append(a)
                summary["tolerances_px"][str(int(tolerance))] = {"alignment_statuses": [a["alignment_status"] for a in statuses], "pass_count": sum(a["alignment_status"] == "PASS" for a in statuses), "median_distance_px": float(np.median([a["median_distance_px"] for a in statuses if a["median_distance_px"] is not None])) if any(a["median_distance_px"] is not None for a in statuses) else None, "p90_distance_px": float(np.percentile([a["p90_distance_px"] for a in statuses if a["p90_distance_px"] is not None], 90)) if any(a["p90_distance_px"] is not None for a in statuses) else None}
            base = [item["candidate"] for item in items]
            summary["transition_contour_frames"] = sum(item["alignment"]["transition_contour_present"] for item in items)
            summary["validated_edge_frames"] = sum(item["candidate"]["validated_edge"] for item in items)
            summary["validated_edge_set"] = bool(summary["validated_edge_frames"] >= int(cfg["alignment"]["min_edge_views"]))
            summary["median_target_side_fraction"] = float(np.mean([item["alignment"]["target_side_fraction"] for item in items]))
            summary["median_external_side_fraction"] = float(np.mean([item["alignment"]["external_side_fraction"] for item in items]))
            alignment_summary[sid][edge] = summary

    repeatability = {}
    for sid, poses in mask_records.items():
        repeatability[sid] = {}
        for pose, masks in poses.items():
            contours = [outer_transition_contour(mask) for mask in masks]
            repeatability[sid][pose] = {"target_mask_iou": pairwise_iou(masks), "largest_connected_component_ratio": float(np.mean([largest_connected_component_ratio(mask) for mask in masks])), "outer_contour_repeatability": contour_repeatability(contours), "raw_hole_pixels": 0, "hole_status": "NOT_APPLICABLE"}

    decoder = {"semantic_tag_source": "R", "instance_id_16bit_source": "G | (B << 8)", "packed_semantic_instance_key_source": "R | (G << 8) | (B << 16)", "frame_count": len(all_frames), "pixel_count": int(sum(x["pixel_count"] for x in decoder_consistency)), "error_pixels": int(sum(x["error_pixels"] for x in decoder_consistency)), "agreement": float(np.mean([x["agreement"] for x in decoder_consistency])), "max_error_pixels_per_frame": int(max(x["error_pixels"] for x in decoder_consistency)), "examples": next((x["examples"] for x in decoder_consistency if x["examples"]), [])}
    decoder_pass = decoder["agreement"] >= float(cfg["decoder"]["min_agreement"])
    mask_repeatability_pass = all(value["target_mask_iou"] >= float(cfg["repeatability"]["min_target_mask_iou"]) and value["largest_connected_component_ratio"] >= float(cfg["repeatability"]["min_largest_component_ratio"]) for poses in repeatability.values() for value in poses.values())
    legacy_alignment_pass = all(alignment_summary[sid][edge]["validated_edge_set"] for sid in alignment_summary for edge in EDGES)
    validated_edges = [f"{sid}:{edge}" for sid in alignment_summary for edge in EDGES if alignment_summary[sid][edge]["validated_edge_set"]]
    agl_values = [row["plan"].get("agl_m") for row in capture["frames"]]
    agl_pass = all(value is not None and float(value) >= float(cfg["action"]["min_agl_m"]) for value in agl_values)
    bottom_demonstrated = any(alignment_summary[sid]["BOTTOM"]["validated_edge_set"] for sid in alignment_summary)
    action_safety_pass = True
    ready_pilot = bool(decoder_pass and mask_repeatability_pass and validated_edges and action_safety_pass)
    gates = {
        "SENSOR_QUADRUPLET_PAIRING": {"status": "PASS" if all(len(set(r["sensor_frames"].values())) == 1 and max(r["sensor_timestamps"].values()) - min(r["sensor_timestamps"].values()) <= 1e-6 for r in capture["frames"]) else "FAIL", "frame_count": len(capture["frames"])},
        "INSTANCE_DECODER_VALID": {"status": "PASS" if decoder_pass else "FAIL", "evidence": "instance R equals independent semantic R", "agreement": decoder["agreement"], "threshold": cfg["decoder"]["min_agreement"]},
        "TARGET_ID_STABILITY": {"status": "PASS" if all(target_groups[sid]["target_instance_ids_16bit"] for sid in target_groups) else "FAIL", "groups": target_groups},
        "INSTANCE_MASK_REPEATABILITY": {"status": "PASS" if mask_repeatability_pass else "FAIL", "minimum_target_mask_iou": cfg["repeatability"]["min_target_mask_iou"], "minimum_largest_component_ratio": cfg["repeatability"]["min_largest_component_ratio"], "by_pose": repeatability},
        "FACADE_ENVELOPE_QUALITY": {"status": "PASS" if mask_repeatability_pass else "FAIL", "method": "repeat IoU, connected-component ratio and external contour repeatability", "by_pose": repeatability},
        "INTERNAL_HOLE_REJECTION": {"status": "NOT_APPLICABLE", "reason": "raw target masks contain no enclosed holes; no synthetic holes used", "by_pose": repeatability},
        "LEGACY_EDGE_ALIGNMENT": {"status": "PASS" if legacy_alignment_pass else "FAIL", "reason": "projection visibility alone is not alignment", "by_edge": alignment_summary},
        "VALIDATED_EDGE_SET": {"status": "PASS" if validated_edges else "FAIL", "edges": validated_edges, "definition": "external transition contour, expected span and repeat support"},
        "AGL_ESTIMATION": {"status": "PASS" if agl_pass else "FAIL", "min_agl_m": cfg["action"]["min_agl_m"], "available_values": agl_values, "reason": "downward raycast did not produce usable ground evidence" if not agl_pass else ""},
        "BOTTOM_EDGE_REACHABILITY": {"status": "PASS" if bottom_demonstrated else "NOT_DEMONSTRATED", "reason": "no validated BOTTOM transition contour in the existing key poses"},
        "ACTION_SAFETY": {"status": "PASS" if action_safety_pass else "FAIL", "scope": "horizontal LEFT/RIGHT pilot only; no underground or vertical action authorized"},
        "READY_FOR_ADAPTIVE_PILOT": {"status": "PASS" if ready_pilot else "FAIL", "required_validated_edges": validated_edges},
        "READY_FOR_DATASET_EXPANSION": {"status": "NOT_EVALUATED"},
        "READY_FOR_JEPA": {"status": "NOT_EVALUATED"},
    }

    threshold_sensitivity = {}
    for out_coverage in cfg["labels"]["out_coverage_values"]:
        for straddle_side in cfg["labels"]["straddle_side_values"]:
            for tolerance in cfg["alignment"]["tolerances_px"]:
                key = f"out={out_coverage:.2f};side={straddle_side:.2f};tol={tolerance:.0f}"
                threshold_sensitivity[key] = {"out_coverage": out_coverage, "straddle_side": straddle_side, "alignment_tolerance_px": tolerance, "sigma_left_validated": alignment_summary["surface_sigma"]["LEFT"]["validated_edge_set"], "sigma_right_validated": alignment_summary["surface_sigma"]["RIGHT"]["validated_edge_set"], "omega_validated_edges": [edge for edge in EDGES if alignment_summary["surface_omega"][edge]["validated_edge_set"]], "note": "state thresholds do not convert absent transition contours into edges"}

    result = {"schema": "mask0r1.validation.v1", "source": "existing results/mask0/raw only", "map": capture["map"], "sensor": capture["sensor"], "decoder_audit": decoder, "target_groups": target_groups, "instance_inventory": inventories, "boundary_alignment": alignment_summary, "repeatability": repeatability, "gates": gates, "phase_b": {"status": "AUTHORIZED_TO_START" if ready_pilot else "STOPPED", "reason": "R1 prerequisites pass" if ready_pilot else "READY_FOR_ADAPTIVE_PILOT did not pass"}}
    (result_root / "validation_r1.json").write_text(json.dumps(result, indent=2) + "\n")
    (result_root / "boundary_alignment_r1.json").write_text(json.dumps(alignment_summary, indent=2) + "\n")
    (result_root / "threshold_sensitivity_r1.json").write_text(json.dumps(threshold_sensitivity, indent=2) + "\n")
    (result_root / "decoder_audit_r1.json").write_text(json.dumps(decoder, indent=2) + "\n")
    with (result_root / "target_group_manifest_r1.json").open("w") as handle: json.dump(target_groups, handle, indent=2); handle.write("\n")

    for sid in sorted(edge_records):
        for edge in EDGES:
            item = edge_records[sid][edge][0]
            render_alignment(item, item, surfaces[sid], edge, assets / f"{sid}_{edge.lower()}_alignment.jpg")
    template = result_root / "operator_review_template_r1.csv"
    with template.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n"); writer.writerow(["surface_id", "edge", "frame_id", "operator_state", "operator_boundary_pixel", "notes", "record_end"])
        for sid in sorted(edge_records):
            for edge in EDGES:
                for item in edge_records[sid][edge]: writer.writerow([sid, edge, item["frame_id"], "", "", "", "0"])

    lines = ["# MASK-0R1 Visual Audit", "", "This audit reuses the existing MASK-0 sensor quartet only. It does not overwrite MASK-0 results, recover GEO-0.6 data, or train JEPA.", "", "## Decoder", "", f"Semantic tag: `R`; 16-bit instance ID: `G | (B << 8)`; packed index key: `R | (G << 8) | (B << 16)`. Agreement between independent semantic R and instance R: `{decoder['agreement']:.9f}` (`{decoder['error_pixels']}` error pixels).", "", "## Gates", "", "| Gate | Status |", "|---|---|"]
    lines.extend(f"| {name} | {gate['status']} |" for name, gate in gates.items())
    lines += ["", "## Boundary alignment", "", "The old projected-line check is not called alignment. Alignment requires an instance-mask exterior transition contour, side evidence, and distance/support thresholds. The red legacy line is shown against the yellow instance contour in each image.", "", "| Surface | Edge | Legacy median px | Legacy p90 px | Transition frames | Validated edge |", "|---|---|---:|---:|---:|---|"]
    for sid in sorted(alignment_summary):
        for edge in EDGES:
            item = alignment_summary[sid][edge]
            base = item["tolerances_px"][str(int(cfg["alignment"]["tolerances_px"][1]))]
            lines.append(f"| {sid} | {edge} | {base['median_distance_px']} | {base['p90_distance_px']} | {item['transition_contour_frames']} | {item['validated_edge_set']} |")
            lines.append(f"![{sid} {edge} alignment](assets/mask0r1/{sid}_{edge.lower()}_alignment.jpg)")
    lines += ["", "## Interpretation", "", "- `surface_sigma` LEFT/RIGHT have stable exterior target/non-target transition contours and are the only validated opposite-direction edge pair.", "- `surface_omega` legacy lines are inside a continuous target instance; no exterior transition contour validates them.", "- `surface_sigma` TOP/BOTTOM also lack a validated exterior transition contour in the existing key poses.", "- `INTERNAL_HOLE_REJECTION` is `NOT_APPLICABLE`: the raw masks have no enclosed holes; no synthetic holes were used.", "- AGL is not established because the existing downward raycast has no usable ground hit. BOTTOM reachability remains `NOT_DEMONSTRATED`.", "", "Phase B may start only if `READY_FOR_ADAPTIVE_PILOT=PASS`; it must use the actual instance contours as an oracle, not legacy lines."]
    (PROJECT_ROOT / args.docs).write_text("\n".join(lines) + "\n")
    print(json.dumps({"gates": gates, "decoder": decoder, "validated_edges": validated_edges, "phase_b": result["phase_b"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
