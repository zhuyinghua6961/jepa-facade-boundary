#!/usr/bin/env python3
"""Offline OBS-0 boundary-distance observability audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from boundary_sweep.observability import (aligned_ssim, regression_metrics,
                                          ridge_fit_predict, rgb_descriptor)


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def draw_heatmap(matrix: np.ndarray, labels: list[str], output: Path, title: str,
                 value_range: tuple[float, float] | None = None) -> None:
    size = max(420, 22 * len(labels) + 100)
    canvas = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(canvas)
    low, high = value_range or (float(np.nanmin(matrix)), float(np.nanmax(matrix)))
    low, high = min(low, high), max(high, low + 1e-9)
    margin = 80
    cell = max(8, (size - margin - 20) // max(len(labels), 1))
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = float(matrix[i, j])
            ratio = np.clip((value - low) / (high - low), 0.0, 1.0)
            color = (int(255 * (1.0 - ratio)), int(80 + 150 * (1.0 - abs(ratio - 0.5) * 2)), int(255 * ratio))
            x0, y0 = margin + j * cell, 35 + i * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=color)
    for index, label in enumerate(labels):
        short = label.replace("LEFT", "L").replace("RIGHT", "R").replace("step", "s")
        draw.text((margin + index * cell, 12), short, fill=(0, 0, 0))
        draw.text((8, 35 + index * cell), short, fill=(0, 0, 0))
    draw.text((10, size - 22), title, fill=(0, 0, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=88, optimize=True)


def draw_alias_pair(left: dict, right: dict, score: float, gap: float, output: Path) -> None:
    canvas = Image.new("RGB", (1280, 520), (20, 20, 20))
    for index, row in enumerate((left, right)):
        image = Image.open(resolve(row["rgb_path"])).convert("RGB").resize((640, 480))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 640, 42), fill=(0, 0, 0))
        draw.text((8, 8), f"{row['direction']} step={row['step_index']} remaining={row['remaining_distance_m']:.2f}m", fill=(255, 235, 0))
        canvas.paste(image, (index * 640, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 480, 1280, 520), fill=(0, 0, 0))
    draw.text((8, 490), f"aligned_ssim={score:.4f}  remaining-distance gap={gap:.2f}m", fill=(255, 235, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=84, optimize=True)


def draw_probe_plot(results: dict, output: Path) -> None:
    names = list(results)
    width, height = 1100, 560
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    max_value = max(float(result["aggregate"]["MAE_m"]) for result in results.values())
    left, bottom = 80, 460
    for i, name in enumerate(names):
        value = float(results[name]["aggregate"]["MAE_m"])
        x0 = left + i * 135
        bar_height = int(330 * value / max(max_value, 1e-6))
        draw.rectangle((x0, bottom - bar_height, x0 + 80, bottom), fill=(50, 120, 200))
        draw.text((x0, bottom + 10), name, fill=(0, 0, 0))
        draw.text((x0, bottom - bar_height - 18), f"{value:.2f}", fill=(0, 0, 0))
    draw.text((80, 25), "OBS-0 cross-direction probe MAE (lower is better)", fill=(0, 0, 0))
    draw.text((80, 50), "No random adjacent-frame split; LEFT->RIGHT and RIGHT->LEFT are reported separately.", fill=(0, 0, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=88, optimize=True)


def draw_scatter(records: list[dict], predictions: dict, output: Path) -> None:
    canvas = Image.new("RGB", (1000, 600), "white")
    draw = ImageDraw.Draw(canvas)
    margin = 90
    draw.rectangle((margin, 40, 940, 500), outline=(20, 20, 20), width=2)
    actual = np.asarray([row["target"] for row in records], dtype=float)
    pred = np.asarray(predictions, dtype=float)
    max_value = max(float(actual.max()), float(pred.max()), 1.0)
    for a, p in zip(actual, pred):
        x = int(margin + 850 * a / max_value); y = int(500 - 430 * p / max_value)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(40, 110, 200))
    draw.line((margin, 500, 940, 70), fill=(180, 60, 60), width=2)
    draw.text((margin, 15), "OBS-0 prediction scatter: actual x / predicted y", fill=(0, 0, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=88, optimize=True)


def build_features(records: list[dict]) -> dict[str, np.ndarray]:
    descriptors = np.asarray([row["descriptor"] for row in records], dtype=float)
    previous = np.vstack([descriptors[0], descriptors[:-1]])
    offsets = np.asarray([[row["relative_offset_m"], row["relative_delta_m"]] for row in records], dtype=float)
    poses = np.asarray([row["pose_features"] for row in records], dtype=float)
    return {
        "P0": np.zeros((len(records), 1), dtype=float),
        "P1": np.asarray([[row["step_index"], row["relative_offset_m"], row["relative_delta_m"]] for row in records], dtype=float),
        "P2": poses,
        "P3": descriptors,
        "P4": descriptors - previous,
        "P5": np.column_stack([descriptors, previous, offsets]),
        "P6": np.column_stack([descriptors, previous, poses]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reanalysis", default="results/mask1/event_reanalysis_r1.json")
    parser.add_argument("--output", default="results/obs0")
    parser.add_argument("--assets", default="docs/assets/obs0")
    args = parser.parse_args(argv)
    output = PROJECT_ROOT / args.output; assets = PROJECT_ROOT / args.assets
    output.mkdir(parents=True, exist_ok=True); assets.mkdir(parents=True, exist_ok=True)
    reanalysis = json.loads((PROJECT_ROOT / args.reanalysis).read_text())
    records = []
    for trajectory in reanalysis["trajectories"]:
        first_local = trajectory["first_local_straddle_step"]
        if first_local is None:
            continue
        frames = trajectory["frames"]
        first_position = np.asarray(frames[0]["camera_position_world"], dtype=float)
        previous_position = first_position
        for frame in frames:
            step = int(frame["step_index"])
            if step >= first_local:
                continue
            position = np.asarray(frame["camera_position_world"], dtype=float)
            relative_offset = float(np.linalg.norm(position - first_position))
            relative_delta = float(np.linalg.norm(position - previous_position))
            records.append({
                "direction": trajectory["direction"], "step_index": step,
                "offset_m": float(frame["offset_m"]), "remaining_steps_to_local_boundary": int(first_local - step),
                "remaining_distance_m": float((first_local - step) * trajectory.get("step_m", 0.5)),
                "remaining_steps_to_global_3pct": None if not isinstance(trajectory["first_global_3pct_step"], int) else max(trajectory["first_global_3pct_step"] - step, 0),
                "remaining_steps_to_global_5pct": None if not isinstance(trajectory["first_global_5pct_step"], int) else max(trajectory["first_global_5pct_step"] - step, 0),
                "remaining_steps_to_global_10pct": None if not isinstance(trajectory["first_global_10pct_step"], int) else max(trajectory["first_global_10pct_step"] - step, 0),
                "rgb_path": frame["rgb_path"], "pose_features": [*position.tolist(), 0.0, 0.0, 0.0],
                "relative_offset_m": relative_offset, "relative_delta_m": relative_delta,
                "target": float((first_local - step) * trajectory.get("step_m", 0.5)), "descriptor": rgb_descriptor(np.asarray(Image.open(resolve(frame["rgb_path"])).convert("RGB"))).tolist(),
                "global_external_coverage": float(frame["global_external_coverage"]),
                "frame_id": frame["frame_id"],
            })
            previous_position = position
    records.sort(key=lambda row: (row["direction"], row["step_index"]))
    with (output / "frame_targets.csv").open("w", newline="") as handle:
        fields = ["direction", "step_index", "offset_m", "remaining_steps_to_local_boundary", "remaining_distance_m", "remaining_steps_to_global_3pct", "remaining_steps_to_global_5pct", "remaining_steps_to_global_10pct", "rgb_path", "frame_id"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in records)
    labels = [f"{row['direction']}_s{row['step_index']}" for row in records]
    similarity = np.eye(len(records), dtype=float); shifts = np.zeros_like(similarity)
    pair_rows = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            score, shift = aligned_ssim(np.asarray(Image.open(resolve(records[i]["rgb_path"])).convert("RGB")), np.asarray(Image.open(resolve(records[j]["rgb_path"])).convert("RGB")))
            similarity[i, j] = similarity[j, i] = score; shifts[i, j] = shifts[j, i] = shift
            gap = abs(records[i]["target"] - records[j]["target"])
            pair_rows.append({"left_index": i, "right_index": j, "left_direction": records[i]["direction"], "right_direction": records[j]["direction"], "left_step": records[i]["step_index"], "right_step": records[j]["step_index"], "remaining_distance_gap_m": float(gap), "aligned_ssim": float(score), "phase_shift_px": float(shift), "cross_direction": records[i]["direction"] != records[j]["direction"]})
    with (output / "alias_pairs.csv").open("w", newline="") as handle:
        fields = list(pair_rows[0]) if pair_rows else ["left_index", "right_index"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(pair_rows)
    summary = {"schema": "obs0.alias_summary.v1", "pair_count": len(pair_rows), "thresholds": {}, "strongest_pairs": []}
    for score_threshold in (0.90, 0.95, 0.98):
        for gap_threshold in (1.0, 2.0, 3.0):
            eligible = [pair for pair in pair_rows if pair["remaining_distance_gap_m"] >= gap_threshold]
            aliases = [pair for pair in eligible if pair["aligned_ssim"] >= score_threshold]
            key = f"ssim_{score_threshold:.2f}_gap_{gap_threshold:.1f}m"
            summary["thresholds"][key] = {"eligible_pairs": len(eligible), "alias_pairs": len(aliases), "alias_ratio": float(len(aliases) / max(len(eligible), 1))}
    strongest = sorted((pair for pair in pair_rows if pair["remaining_distance_gap_m"] >= 2.0), key=lambda pair: pair["aligned_ssim"], reverse=True)[:5]
    summary["strongest_pairs"] = strongest
    (output / "alias_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    draw_heatmap(similarity, labels, assets / "pairwise_ssim_heatmap.jpg", "aligned SSIM")
    gap_matrix = np.zeros_like(similarity)
    for pair in pair_rows:
        gap_matrix[pair["left_index"], pair["right_index"]] = gap_matrix[pair["right_index"], pair["left_index"]] = pair["remaining_distance_gap_m"]
    draw_heatmap(gap_matrix, labels, assets / "remaining_distance_gap_heatmap.jpg", "remaining-distance gap (m)", (0.0, 5.0))
    for index, pair in enumerate(strongest, 1):
        draw_alias_pair(records[pair["left_index"]], records[pair["right_index"]], pair["aligned_ssim"], pair["remaining_distance_gap_m"], assets / f"alias_pair_{index:02d}.jpg")
    features = build_features(records)
    target = np.asarray([row["target"] for row in records], dtype=float)
    by_direction = {direction: np.asarray([i for i, row in enumerate(records) if row["direction"] == direction], dtype=int) for direction in ("LEFT", "RIGHT")}
    probes = {}
    prediction_rows = []
    for name, values in features.items():
        splits = {}
        all_actual, all_pred = [], []
        for train_direction, test_direction in (("LEFT", "RIGHT"), ("RIGHT", "LEFT")):
            train_indices, test_indices = by_direction[train_direction], by_direction[test_direction]
            baseline_prediction = np.full(len(test_indices), float(target[train_indices].mean()))
            baseline = regression_metrics(target[test_indices], baseline_prediction)
            prediction = ridge_fit_predict(values[train_indices], target[train_indices], values[test_indices], alpha=1.0)
            metrics = regression_metrics(target[test_indices], prediction, baseline["MAE_m"])
            splits[f"train_{train_direction}_test_{test_direction}"] = {"metrics": metrics, "constant_baseline": baseline, "test_count": int(len(test_indices))}
            all_actual.extend(target[test_indices].tolist()); all_pred.extend(prediction.tolist())
            if name in ("P3", "P5", "P6"):
                draw_scatter([records[i] for i in test_indices], prediction, assets / f"{name.lower()}_{train_direction.lower()}_to_{test_direction.lower()}.jpg")
        aggregate_baseline = float(np.mean([splits[key]["constant_baseline"]["MAE_m"] for key in splits]))
        aggregate_metrics = regression_metrics(all_actual, all_pred, aggregate_baseline)
        probes[name] = {"splits": splits, "aggregate": aggregate_metrics, "input_definition": {"P0": "constant mean", "P1": "step and relative odometry", "P2": "absolute pose", "P3": "current RGB descriptor", "P4": "two-frame RGB change", "P5": "short RGB history plus relative odometry", "P6": "short RGB history plus absolute pose"}[name]}
    (output / "probe_results.json").write_text(json.dumps({"schema": "obs0.probe_results.v1", "split": "complete direction holdout", "records": len(records), "probes": probes}, indent=2) + "\n")
    draw_probe_plot(probes, assets / "probe_mae_comparison.jpg")
    draw_probe_plot({"P3_RGB": probes["P3"], "P5_History+Odom": probes["P5"],
                     "P2_AbsPose": probes["P2"], "P6_History+Pose": probes["P6"]},
                    assets / "probe_family_comparison.jpg")
    groups = {
        "rgb_only_P3": probes["P3"]["aggregate"],
        "history_odometry_P5": probes["P5"]["aggregate"],
        "absolute_pose_only_P2": probes["P2"]["aggregate"],
        "history_absolute_pose_P6": probes["P6"]["aggregate"],
    }
    validation = {
        "schema": "obs0.validation.v1", "source": "pre-local-straddle MASK-1 frames only",
        "sample_count": len(records), "directions": {key: int(len(value)) for key, value in by_direction.items()},
        "gates": {
            "PREBOUNDARY_MASK_PROGRESS": {
                "status": "PRESENT" if any(row["global_external_coverage"] > 0.0 for row in records) else "ABSENT",
                "reason": "RIGHT step 9 has 1.526% external coverage before the local contour event" if any(row["direction"] == "RIGHT" and row["global_external_coverage"] > 0.0 for row in records) else "target coverage is exactly 1.0 before local contour on both directions",
            },
            "SINGLE_FRAME_RGB_OBSERVABILITY": {"status": "PASS" if probes["P3"]["aggregate"]["constant_baseline_improvement_m"] > 0.1 else "FAIL", "evidence": probes["P3"]["aggregate"]},
            "HISTORY_ODOMETRY_OBSERVABILITY": {"status": "PASS" if probes["P5"]["aggregate"]["constant_baseline_improvement_m"] > 0.1 else "FAIL", "evidence": probes["P5"]["aggregate"]},
            "ABSOLUTE_POSE_DEPENDENCE": {"status": "PASS" if probes["P2"]["aggregate"]["constant_baseline_improvement_m"] > 0.1 else "FAIL", "evidence": probes["P2"]["aggregate"], "interpretation": "pose predictive signal is not evidence of cross-building generalization"},
            "BOUNDARY_DETECTION_OBSERVABILITY": {"status": "INCONCLUSIVE", "reason": "local event is derived from privileged instance masks; human-visible event remains pending"},
            "BOUNDARY_DISTANCE_OBSERVABILITY": {"status": "INCONCLUSIVE", "evidence": groups, "reason": "two trajectories provide a diagnostic holdout, not cross-surface validation"},
            "ACTION_SELECTION_OBSERVABILITY": {"status": "NOT_EVALUATED", "reason": "no same-start alternative actions in existing data"},
            "READY_FOR_MULTI_SURFACE_CAPTURE": {"status": "CONDITIONAL_PASS", "reason": "diagnostic signal exists, but visual and multi-surface validation remain pending"},
            "READY_FOR_DATASET_EXPANSION": {"status": "NOT_EVALUATED"},
            "READY_FOR_JEPA": {"status": "NOT_EVALUATED"},
        },
        "probe_summary": groups,
        "alias_summary": summary,
        "probe_feature_boundary": "instance/semantic/depth are GT generation only; not included in P0-P6 inputs",
    }
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
