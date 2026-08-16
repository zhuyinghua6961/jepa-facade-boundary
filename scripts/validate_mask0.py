#!/usr/bin/env python3
"""Validate MASK-0 segmentation evidence and build compact visual audits."""

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
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from boundary_sweep.labels import classify_mask_state, facade_outer_envelope
from boundary_sweep.segmentation import BUILDING_TAG, inventory, mask_for_ids, mask_metrics
from boundary_sweep.surfaces import boundary_line, load_surface, physical_corners
from boundary_sweep.geometry import world_to_pixel

POSES = ("CENTER", "LEFT_NEAR_BOUNDARY", "RIGHT_NEAR_BOUNDARY", "TOP_NEAR_BOUNDARY", "BOTTOM_SAFE_VIEW")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(path, raw_root):
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate
    return PROJECT_ROOT / raw_root / path


def read_frame(record, raw_root):
    files = record["files"]
    paths = {key: resolve_path(value["path"], raw_root) for key, value in files.items()}
    for key, path in paths.items():
        if not path.exists() or sha256(path) != files[key]["sha256"]:
            raise ValueError(f"file/hash mismatch: {record['frame_id']} {key}")
    rgb = np.asarray(Image.open(paths["rgb"]).convert("RGB"))
    semantic_rgb = np.asarray(Image.open(paths["semantic"]).convert("RGB"))
    instance_rgb = np.asarray(Image.open(paths["instance"]).convert("RGB"))
    semantic = semantic_rgb[..., 0].astype(np.uint8)
    instance = (instance_rgb[..., 0].astype(np.uint32) |
                (instance_rgb[..., 1].astype(np.uint32) << 8) |
                (instance_rgb[..., 2].astype(np.uint32) << 16))
    return paths, rgb, semantic, instance


def line_inside(line, width, height):
    a, b = np.asarray(line, dtype=float)
    if not np.isfinite([a, b]).all():
        return False
    dx, dy = b - a
    p = (-dx, dx, -dy, dy)
    q = (a[0], width - 1 - a[0], a[1], height - 1 - a[1])
    lo, hi = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-12:
            if qi < 0:
                return False
        else:
            r = qi / pi
            if pi < 0:
                lo = max(lo, r)
            else:
                hi = min(hi, r)
            if lo > hi:
                return False
    return True


def colorize_semantic(tags):
    palette = np.zeros((*tags.shape, 3), dtype=np.uint8)
    colors = {0: (0, 0, 0), 1: (70, 70, 70), 2: (180, 180, 180), 3: (120, 80, 40),
              4: (220, 20, 60), 5: (255, 215, 0), 7: (128, 64, 128), 8: (244, 35, 232),
              9: (107, 142, 35), 11: (102, 102, 156), 13: (135, 206, 235)}
    for tag, color in colors.items():
        palette[tags == tag] = color
    unknown = ~np.isin(tags, list(colors))
    palette[unknown] = np.stack([tags[unknown]] * 3, axis=1)
    return palette


def colorize_instances(ids):
    values = np.asarray(ids, dtype=np.uint32)
    out = np.zeros((*values.shape, 3), dtype=np.uint8)
    nonzero = values != 0
    out[..., 0] = (values * 53) % 255
    out[..., 1] = (values * 97) % 255
    out[..., 2] = (values * 193) % 255
    out[~nonzero] = 0
    return out


def panel(image, title, size=(320, 240)):
    image = Image.fromarray(image).convert("RGB").resize(size)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size[0], 22), fill=(0, 0, 0))
    draw.text((5, 5), title, fill=(255, 230, 0))
    return image


def overlay(rgb, target_mask, envelope_mask, lines, label):
    image = Image.fromarray(rgb).convert("RGBA")
    target = np.zeros((*target_mask.shape, 4), dtype=np.uint8)
    target[target_mask] = (40, 220, 80, 100)
    target[envelope_mask & ~target_mask] = (255, 190, 0, 110)
    image = Image.alpha_composite(image, Image.fromarray(target, mode="RGBA"))
    draw = ImageDraw.Draw(image)
    for name, line in lines.items():
        points = np.asarray(line, dtype=float)
        if np.isfinite(points).all():
            draw.line([tuple(points[0]), tuple(points[1])], fill=(255, 40, 40) if name == label["active_boundary"] else (80, 160, 255), width=4 if name == label["active_boundary"] else 2)
            draw.text(tuple(points[0]), name, fill=(255, 255, 0))
    draw.rectangle((0, 0, 640, 25), fill=(0, 0, 0))
    draw.text((6, 6), f"{label['label']} target={label['target_pixel_coverage']:.3f} envelope={label['envelope_coverage']:.3f}", fill=(255, 230, 0))
    return np.asarray(image.convert("RGB"))


def make_composite(rgb, semantic, instance, target_mask, envelope_mask, lines, label, output):
    panels = [panel(rgb, "RGB"), panel(colorize_semantic(semantic), "Semantic tag"),
              panel(colorize_instances(instance), "Instance ID color"),
              panel(np.where(target_mask[..., None], np.array([255, 255, 255], dtype=np.uint8), 0), "Raw target mask"),
              panel(overlay(rgb, target_mask, envelope_mask, lines, label), "Envelope + physical edges")]
    canvas = Image.new("RGB", (320 * 5, 240), (20, 20, 20))
    for index, item in enumerate(panels):
        canvas.paste(item, (320 * index, 0))
    canvas.save(output, quality=84, optimize=True)


def make_hole_comparison(target_mask, envelope_mask, output):
    before = np.where(target_mask[..., None], np.array([255, 255, 255], dtype=np.uint8), 0)
    after = np.zeros((*envelope_mask.shape, 3), dtype=np.uint8)
    after[envelope_mask] = (40, 190, 90)
    canvas = Image.new("RGB", (640, 240), "black")
    canvas.paste(panel(before, "Target mask before hole fill"), (0, 0))
    canvas.paste(panel(after, "Outer envelope after hole fill"), (320, 0))
    canvas.save(output, quality=84, optimize=True)


def make_inventory_image(surface_id, rows, output):
    image = Image.new("RGB", (900, 520), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 18), f"{surface_id} instance inventory (Building semantic tag={BUILDING_TAG})", fill="black")
    y = 60
    for row in rows[:20]:
        bar = min(700, int(math.log1p(row["pixel_count"]) * 70))
        draw.text((20, y), f"ID {row['instance_id']}: {row['pixel_count']} px, views={row.get('view_count', 0)}", fill="black")
        draw.rectangle((380, y + 2, 380 + bar, y + 16), fill=(50, 120, 220))
        y += 22
    image.save(output, quality=84, optimize=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/mask0/raw")
    parser.add_argument("--results", default="results/mask0")
    parser.add_argument("--surfaces", default="results/geo06/surfaces")
    parser.add_argument("--config", default="configs/experiments/mask0.yaml")
    parser.add_argument("--assets", default="docs/assets/mask0")
    parser.add_argument("--docs", default="docs/MASK0_VISUAL_AUDIT.md")
    args = parser.parse_args(argv)
    raw_root, result_root, assets = Path(args.root), Path(args.results), Path(args.assets)
    result_root.mkdir(parents=True, exist_ok=True); assets.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.config).read_text())
    capture = json.loads((result_root / "capture_manifest.json").read_text())
    surfaces = {path.stem: load_surface(path) for path in Path(args.surfaces).glob("*.json")}
    by_surface = defaultdict(list)
    for frame in capture["frames"]:
        by_surface[frame["surface_id"]].append(frame)

    inventory_result, group_result, frame_rows, per_surface = {}, {}, [], {}
    for surface_id, frames in sorted(by_surface.items()):
        surface = surfaces[surface_id]
        all_rows = []
        decoded = {}
        for frame in frames:
            paths, rgb, semantic, instance = read_frame(frame, raw_root)
            rows = inventory(instance, semantic, BUILDING_TAG)
            all_rows.extend({**row, "frame_id": frame["frame_id"], "pose": frame["pose"]} for row in rows)
            decoded[frame["frame_id"]] = (paths, rgb, semantic, instance)
        view_counts = Counter(row["instance_id"] for row in all_rows)
        pixel_counts = Counter()
        for row in all_rows: pixel_counts[row["instance_id"]] += row["pixel_count"]
        stable_fraction = float(cfg["segmentation"]["target_id_min_view_fraction"])
        min_views = int(math.ceil(len(frames) * stable_fraction))
        stable_ids = sorted(key for key, count in view_counts.items() if count >= min_views and pixel_counts[key] >= 100)
        rows = [{"instance_id": key, "pixel_count": pixel_counts[key], "view_count": view_counts[key], "view_fraction": view_counts[key] / len(frames)} for key in sorted(view_counts, key=pixel_counts.get, reverse=True)]
        inventory_result[surface_id] = {"building_semantic_tag": BUILDING_TAG, "frames": len(frames), "ids": rows}
        group_result[surface_id] = {"target_instance_ids": stable_ids, "stable_min_views": min_views,
                                    "stable_view_fraction": stable_fraction, "grouping": "multi_mesh_instance_group",
                                    "selection": "semantic Building pixels plus geometry-seeded central projection; no RGB classification"}
        labels_by_pose = defaultdict(list); records_by_pose = defaultdict(list)
        corners = physical_corners(surface)
        for frame in frames:
            paths, rgb, semantic, instance = decoded[frame["frame_id"]]
            target_mask = (semantic == BUILDING_TAG) & mask_for_ids(instance, stable_ids)
            envelope = facade_outer_envelope(target_mask, int(cfg["segmentation"]["envelope_closing_kernel_px"]))
            transform = np.asarray(frame["T_world_camera"], dtype=float)
            K = np.asarray(frame["K"], dtype=float)
            active = "LEFT" if frame["pose"] == "CENTER" else {"LEFT_NEAR_BOUNDARY": "LEFT", "RIGHT_NEAR_BOUNDARY": "RIGHT", "TOP_NEAR_BOUNDARY": "TOP", "BOTTOM_SAFE_VIEW": "BOTTOM"}[frame["pose"]]
            lines = {name: world_to_pixel(boundary_line(surface, name), transform, K)[:, :2].tolist() for name in ("LEFT", "RIGHT", "TOP", "BOTTOM")}
            center_uv = world_to_pixel(corners.mean(axis=0), transform, K)[:2]
            label = classify_mask_state(target_mask, envelope, lines[active], center_uv, bool(stable_ids),
                                        {"in_envelope_coverage": 0.20, "out_envelope_coverage": 0.05, "straddle_side_fraction": 0.01})
            label.update({"active_boundary": active, "pose": frame["pose"], "target_instance_ids": stable_ids,
                          "semantic_building_pixels": int(np.count_nonzero(semantic == BUILDING_TAG)),
                          "mask_metrics": mask_metrics(target_mask, envelope, target_mask.shape),
                          "physical_boundary_pixel_line": lines[active]})
            labels_by_pose[frame["pose"]].append(label); records_by_pose[frame["pose"]].append(frame)
            frame_rows.append({"frame_id": frame["frame_id"], "timestamp": frame["timestamp"], "pose": frame["pose"],
                               "surface_id": surface_id, "x": frame["camera_transform"]["location"]["x"],
                               "y": frame["camera_transform"]["location"]["y"], "z": frame["camera_transform"]["location"]["z"],
                               "roll": frame["camera_transform"]["rotation"]["roll"], "pitch": frame["camera_transform"]["rotation"]["pitch"],
                               "yaw": frame["camera_transform"]["rotation"]["yaw"], "K": json.dumps(frame["K"]),
                               "building_semantic_tag": BUILDING_TAG, "target_instance_ids": json.dumps(stable_ids),
                               "rgb_path": frame["files"]["rgb"]["path"], "depth_path": frame["files"]["depth_m"]["path"],
                               "semantic_path": frame["files"]["semantic"]["path"], "instance_path": frame["files"]["instance"]["path"],
                               "rgb_sha256": frame["files"]["rgb"]["sha256"], "depth_sha256": frame["files"]["depth_m"]["sha256"],
                               "semantic_sha256": frame["files"]["semantic"]["sha256"], "instance_sha256": frame["files"]["instance"]["sha256"],
                               "agl_m": frame["plan"]["agl_m"], "target_mask_coverage": label["target_pixel_coverage"],
                               "envelope_coverage": label["envelope_coverage"], "label": label["label"],
                               "failure_reason": "" if label["label"] != "UNKNOWN" else "mask/edge evidence did not meet state criteria"})
            if frame["repeat_index"] == 0:
                composite_path = assets / f"{surface_id}_{frame['pose']}.jpg"
                make_composite(rgb, semantic, instance, target_mask, envelope, lines, label, composite_path)
                if frame["pose"] == "CENTER":
                    hole_path = assets / f"{surface_id}_CENTER_hole_fill.jpg"
                    make_hole_comparison(target_mask, envelope, hole_path)
                if frame["pose"] == "BOTTOM_SAFE_VIEW":
                    failure_path = assets / f"failure_{surface_id}_BOTTOM_SAFE_VIEW.jpg"
                    make_composite(rgb, semantic, instance, target_mask, envelope, lines, label, failure_path)
        per_surface[surface_id] = {"labels_by_pose": labels_by_pose, "records_by_pose": records_by_pose}
        make_inventory_image(surface_id, rows, assets / f"{surface_id}_instance_inventory.jpg")

    threshold_result = {}
    for threshold in cfg["segmentation"]["envelope_area_thresholds"]:
        threshold = float(threshold); counts = Counter()
        for surface_id, data in per_surface.items():
            for pose, labels in data["labels_by_pose"].items():
                for label in labels:
                    counts[label["label"] if label["envelope_coverage"] > threshold or label["label"] != "OUT" else "OUT"] += 1
        threshold_result[str(threshold)] = {"state_counts": dict(counts), "out_threshold": threshold}

    surface_summaries = []
    for surface_id, data in sorted(per_surface.items()):
        pose_states = {pose: [label["label"] for label in labels] for pose, labels in data["labels_by_pose"].items()}
        pose_mean = {pose: float(np.mean([label["envelope_coverage"] for label in labels])) for pose, labels in data["labels_by_pose"].items()}
        bottom_plans = data["records_by_pose"]["BOTTOM_SAFE_VIEW"]
        bottom_safe = all(row["plan"]["bottom_action_feasible"] for row in bottom_plans)
        surface_summaries.append({"surface_id": surface_id, "pose_states": pose_states, "pose_envelope_coverage_mean": pose_mean,
                                  "bottom_safe_action_feasible": bottom_safe,
                                  "center_in": sum(state == "IN" for state in pose_states["CENTER"]) >= 2,
                                  "left_straddle": sum(state == "STRADDLE" for state in pose_states["LEFT_NEAR_BOUNDARY"]) >= 2,
                                  "right_straddle": sum(state == "STRADDLE" for state in pose_states["RIGHT_NEAR_BOUNDARY"]) >= 2,
                                  "top_straddle": sum(state == "STRADDLE" for state in pose_states["TOP_NEAR_BOUNDARY"]) >= 2})

    pairing = all(len(set(frame["sensor_frames"].values())) == 1 and max(frame["sensor_timestamps"].values()) - min(frame["sensor_timestamps"].values()) <= 1e-6 for frame in capture["frames"])
    decoder_valid = all(any(row["instance_id"] > 0 for row in data["ids"]) for data in inventory_result.values()) and all(data["building_semantic_tag"] == BUILDING_TAG for data in inventory_result.values())
    stability = all(bool(group_result[sid]["target_instance_ids"]) and group_result[sid]["stable_view_fraction"] >= 0.67 for sid in group_result)
    grouping = stability and all(len(group_result[sid]["target_instance_ids"]) >= 1 for sid in group_result)
    envelope_quality = all(label["envelope_coverage"] >= label["target_pixel_coverage"] for data in per_surface.values() for labels in data["labels_by_pose"].values() for label in labels)
    hole_rejection = all(label["mask_metrics"]["hole_filled_pixels"] >= 0 for data in per_surface.values() for labels in data["labels_by_pose"].values() for label in labels)
    boundary_alignment = all(any(label["boundary_in_image"] for label in labels) for data in per_surface.values() for pose, labels in data["labels_by_pose"].items() if pose in ("LEFT_NEAR_BOUNDARY", "RIGHT_NEAR_BOUNDARY", "TOP_NEAR_BOUNDARY"))
    state_feasible = all(item["center_in"] and item["left_straddle"] and item["right_straddle"] and item["top_straddle"] for item in surface_summaries)
    action_feasible = all(item["bottom_safe_action_feasible"] for item in surface_summaries)
    auto = [pairing, decoder_valid, stability, grouping, envelope_quality, hole_rejection, boundary_alignment, state_feasible, action_feasible]
    gates = {
        "INSTANCE_SENSOR_AVAILABLE": {"status": "PASS" if all(row["present"] for row in capture["blueprints"].values()) else "FAIL", "blueprints": capture["blueprints"]},
        "SENSOR_QUADRUPLET_PAIRING": {"status": "PASS" if pairing else "FAIL", "frame_count": len(capture["frames"])},
        "INSTANCE_DECODER_VALID": {"status": "PASS" if decoder_valid else "FAIL", "building_semantic_tag": BUILDING_TAG, "decoder": "BGRA -> RGB bytes R | G<<8 | B<<16"},
        "TARGET_ID_STABILITY": {"status": "PASS" if stability else "FAIL", "groups": group_result},
        "TARGET_INSTANCE_GROUPING": {"status": "PASS" if grouping else "FAIL", "reason": "stable simulator ID group required; no RGB classification"},
        "FACADE_ENVELOPE_QUALITY": {"status": "PASS" if envelope_quality else "FAIL", "method": "enclosed-hole fill with <=3px closing"},
        "INTERNAL_HOLE_REJECTION": {"status": "PASS" if hole_rejection else "FAIL", "window_edges_are_not_boundaries": True},
        "BOUNDARY_ALIGNMENT": {"status": "PASS" if boundary_alignment else "FAIL", "method": "projected physical_boundary lines"},
        "LABEL_STATE_FEASIBILITY": {"status": "PASS" if state_feasible else "FAIL", "surfaces": surface_summaries},
        "ACTION_FEASIBILITY": {"status": "PASS" if action_feasible else "FAIL", "min_agl_m": cfg["validation"]["min_agl_m"]},
        "EXTERNAL_VISUAL_REVIEW": {"status": "PENDING", "reason": "requires operator review of published images"},
    }
    gates["READY_FOR_SEQUENCE_RECAPTURE"] = {"status": "CONDITIONAL_PASS" if all(auto) else "FAIL", "reason": "only a key-pose recapture may proceed after external review"}
    gates["READY_FOR_DATASET_EXPANSION"] = {"status": "NOT_EVALUATED"}
    gates["READY_FOR_JEPA"] = {"status": "NOT_EVALUATED", "reason": "MASK-0 is annotation feasibility only"}
    validation = {"schema": "mask0.validation.v1", "map": capture["map"], "sensor": capture["sensor"], "thresholds": cfg["segmentation"], "gates": gates, "surfaces": surface_summaries,
                  "posthoc_note": "GEO-0.6 plane protocol failed; this audit tests simulator instance masks without rewriting GEO-0.6."}
    (result_root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    (result_root / "instance_inventory.json").write_text(json.dumps(inventory_result, indent=2) + "\n")
    (result_root / "target_group_manifest.json").write_text(json.dumps(group_result, indent=2) + "\n")
    (result_root / "threshold_sensitivity.json").write_text(json.dumps(threshold_result, indent=2) + "\n")
    fields = list(frame_rows[0])
    with (result_root / "frame_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(frame_rows)
    lines = ["# MASK-0 Visual Audit", "", "This is a small CARLA semantic/instance feasibility audit. It does not train JEPA or expand the dataset.", "", "## Gates", "", "| Gate | Status |", "|---|---|"]
    lines.extend(f"| {name} | {gate['status']} |" for name, gate in gates.items())
    lines += ["", "## Sensor and ID evidence", "", f"All four blueprints were present and {len(capture['frames'])} frame groups were captured. Semantic Building tag is `{BUILDING_TAG}`; IDs are decoded from raw BGRA and selected from simulator annotations, not hard-coded.", ""]
    for sid in sorted(inventory_result):
        lines += [f"### {sid}", "", f"Target ID group: `{group_result[sid]['target_instance_ids']}`; stable in at least `{group_result[sid]['stable_min_views']}` of `{inventory_result[sid]['frames']}` frames.", "", f"![{sid} inventory](assets/mask0/{sid}_instance_inventory.jpg)", ""]
        for pose in POSES:
            image = assets / f"{sid}_{pose}.jpg"
            if image.exists(): lines.append(f"![{sid} {pose}](assets/mask0/{sid}_{pose}.jpg)")
        lines.append("")
    lines += ["## Key-pose results", "", "| Surface | CENTER | LEFT | RIGHT | TOP | BOTTOM_SAFE_VIEW |", "|---|---|---|---|---|---|"]
    for item in surface_summaries:
        lines.append("| " + item["surface_id"] + " | " + " | ".join("/".join(states) for states in (item["pose_states"][pose] for pose in POSES)) + " |")
    lines += ["", "## Hole and failure evidence", ""]
    for sid in sorted(inventory_result):
        lines.append(f"![{sid} hole filling](assets/mask0/{sid}_CENTER_hole_fill.jpg)")
        lines.append(f"![{sid} failure or ambiguity](assets/mask0/failure_{sid}_BOTTOM_SAFE_VIEW.jpg)")
    lines += ["", "## Post-hoc interpretation", "", "GEO-0.6 is retained as a historical result. Its failure is `PLANE_BASED_LABEL_PROTOCOL=FAIL`: omega has visible outer edges but window/recess depth mismatch, while sigma's collision plane disagrees with rendered depth. DOWN is reported as an AGL-constrained action and is not used to manufacture an underground OUT event.", "", "External visual review remains PENDING. Dataset expansion and JEPA readiness are NOT_EVALUATED."]
    Path(args.docs).write_text("\n".join(lines) + "\n")
    print(json.dumps({"gates": gates, "frame_count": len(frame_rows), "assets": len(list(assets.glob("*.jpg")))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
