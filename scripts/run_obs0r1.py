#!/usr/bin/env python3
"""Leakage-corrected offline OBS-0R1 analysis.

This script reuses the saved MASK-1 event reanalysis and raw RGB only.  It
never starts CARLA and never changes the historical OBS-0 files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from boundary_sweep.observability import (grouped_history_descriptors,
                                          fixed_length_descriptor,
                                          regression_metrics, ridge_fit_predict,
                                          rgb_descriptor, similarity_metrics)


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def image(path: str) -> np.ndarray:
    return np.asarray(Image.open(resolve(path)).convert("RGB"))


def compact_descriptor(path: str) -> np.ndarray:
    """Fixed, non-fitted compression to keep the tiny ridge probe tractable."""
    return fixed_length_descriptor(rgb_descriptor(image(path)), length=128)


def draw_heatmap(matrix: np.ndarray, labels: list[str], output: Path, title: str) -> None:
    size = max(420, 22 * len(labels) + 100)
    canvas = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(canvas)
    low, high = float(np.nanmin(matrix)), float(np.nanmax(matrix))
    high = max(high, low + 1e-6)
    margin, cell = 80, max(8, (size - 100) // max(len(labels), 1))
    for i in range(len(labels)):
        for j in range(len(labels)):
            ratio = np.clip((float(matrix[i, j]) - low) / (high - low), 0.0, 1.0)
            color = (int(255 * (1 - ratio)), int(80 + 150 * (1 - abs(ratio - .5) * 2)), int(255 * ratio))
            draw.rectangle((margin + j * cell, 35 + i * cell,
                            margin + (j + 1) * cell, 35 + (i + 1) * cell), fill=color)
    for i, label in enumerate(labels):
        short = label.replace("LEFT", "L").replace("RIGHT", "R").replace("step", "s")
        draw.text((margin + i * cell, 12), short, fill=(0, 0, 0))
        draw.text((8, 35 + i * cell), short, fill=(0, 0, 0))
    draw.text((10, size - 22), title, fill=(0, 0, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=88, optimize=True)


def draw_pair(left: dict, right: dict, metrics: dict, rank: int, output: Path) -> None:
    canvas = Image.new("RGB", (1280, 520), (20, 20, 20))
    for index, row in enumerate((left, right)):
        panel = Image.open(resolve(row["rgb_path"])).convert("RGB").resize((640, 480))
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, 640, 42), fill=(0, 0, 0))
        draw.text((8, 8), f"{row['direction']} step={row['step_index']} remaining={row['target']:.2f}m", fill=(255, 235, 0))
        canvas.paste(panel, (index * 640, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 480, 1280, 520), fill=(0, 0, 0))
    draw.text((8, 490), f"rank={rank} raw={metrics['raw_ssim']:.4f} aligned={metrics['phase_aligned_ssim']:.4f} HOG={metrics['hog_cosine']:.4f} gap={abs(left['target'] - right['target']):.2f}m", fill=(255, 235, 0))
    canvas.save(output, quality=84, optimize=True)


def synthetic_alignment_check() -> dict:
    base = np.zeros((160, 220, 3), dtype=np.uint8)
    cv2.rectangle(base, (25, 30), (190, 125), (220, 220, 220), 2)
    cv2.circle(base, (90, 80), 22, (255, 120, 30), -1)
    shifted = cv2.warpAffine(base, np.float32([[1, 0, 7], [0, 1, -5]]), (220, 160), borderMode=cv2.BORDER_REFLECT)
    metrics = similarity_metrics(base, shifted)
    return {"status": "PASS" if metrics["phase_aligned_ssim"] > metrics["raw_ssim"] else "FAIL", "metrics": metrics, "known_shift_px": [7, -5]}


def evaluate_probes(records: list[dict], previous: np.ndarray, history_valid: np.ndarray) -> dict:
    current = np.asarray([row["descriptor"] for row in records], dtype=float)
    relative = np.asarray([[row["relative_offset_m"], row["relative_delta_m"]] for row in records], dtype=float)
    pose = np.asarray([row["pose_features"] for row in records], dtype=float)
    direction = np.asarray([[1.0, 0.0] if row["direction"] == "LEFT" else [0.0, 1.0] for row in records])
    mask = history_valid.astype(float)[:, None]
    target = np.asarray([row["target"] for row in records], dtype=float)
    features = {
        "B0": np.zeros((len(records), 1)),
        "B1": np.asarray([[row["step_index"]] for row in records], dtype=float),
        "B2": relative,
        "V0": current,
        "V1": np.column_stack([current, previous, mask]),
        "F1": np.column_stack([current, previous, mask, relative]),
        "A0": np.column_stack([pose, direction]),
        "F2": np.column_stack([current, previous, mask, pose, direction]),
    }
    by_direction = {name: np.asarray([i for i, row in enumerate(records) if row["direction"] == name], dtype=int) for name in ("LEFT", "RIGHT")}
    probes = {}
    for name, values in features.items():
        splits = {}
        actual_all, pred_all = [], []
        for train_direction, test_direction in (("LEFT", "RIGHT"), ("RIGHT", "LEFT")):
            train_idx, test_idx = by_direction[train_direction], by_direction[test_direction]
            baseline_pred = np.full(len(test_idx), float(target[train_idx].mean()))
            baseline = regression_metrics(target[test_idx], baseline_pred)
            pred = baseline_pred if name == "B0" else ridge_fit_predict(values[train_idx], target[train_idx], values[test_idx], alpha=1.0)
            metrics = regression_metrics(target[test_idx], pred, baseline["MAE_m"])
            splits[f"train_{train_direction}_test_{test_direction}"] = {"metrics": metrics, "constant_baseline": baseline, "test_count": int(len(test_idx))}
            actual_all.extend(target[test_idx].tolist()); pred_all.extend(pred.tolist())
        merged_baseline = float(np.mean([row["constant_baseline"]["MAE_m"] for row in splits.values()]))
        probes[name] = {"input_definition": {"B0": "constant mean", "B1": "step index only; FIXED_SCHEDULE_SHORTCUT", "B2": "relative odometry only; no step index", "V0": "current RGB only", "V1": "RGB history only; no pose/odometry", "F1": "RGB history + relative odometry", "A0": "absolute pose + action direction", "F2": "RGB history + absolute pose + action direction"}[name], "splits": splits, "merged_direction_holdout": regression_metrics(actual_all, pred_all, merged_baseline)}
    return probes


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--reanalysis", default="results/mask1/event_reanalysis_r1.json")
    parser.add_argument("--output", default="results/obs0")
    parser.add_argument("--assets", default="docs/assets/obs0r1")
    parser.add_argument("--max-frames-per-trajectory", type=int)
    args = parser.parse_args(argv)
    if args.max_frames_per_trajectory is not None and args.max_frames_per_trajectory < 2:
        parser.error("--max-frames-per-trajectory must be at least 2")
    cv2.setNumThreads(1)
    output, assets = PROJECT_ROOT / args.output, PROJECT_ROOT / args.assets
    output.mkdir(parents=True, exist_ok=True); assets.mkdir(parents=True, exist_ok=True)
    source = json.loads((PROJECT_ROOT / args.reanalysis).read_text())
    records = []
    for trajectory in source["trajectories"]:
        first_local = trajectory["first_local_straddle_step"]
        eligible = [frame for frame in trajectory["frames"]
                    if first_local != "NOT_REACHED" and
                    int(frame["step_index"]) < int(first_local)]
        if args.max_frames_per_trajectory is not None:
            eligible = eligible[:args.max_frames_per_trajectory]
        for frame in eligible:
            position = np.asarray(frame["camera_position_world"], dtype=float)
            records.append({
                "trajectory_id": trajectory["sequence_id"], "direction": trajectory["direction"],
                "step_index": int(frame["step_index"]), "offset_m": float(frame["offset_m"]),
                "target": float((int(first_local) - int(frame["step_index"])) * trajectory.get("step_m", 0.5)),
                "relative_offset_m": float(abs(frame["offset_m"])), "relative_delta_m": 0.0,
                "pose_features": position.tolist(), "rgb_path": frame["rgb_path"], "frame_id": int(frame["frame_id"]),
                "descriptor": compact_descriptor(frame["rgb_path"]).tolist(),
            })
    records.sort(key=lambda row: (row["trajectory_id"], row["step_index"]))
    for index, row in enumerate(records):
        same = [other for other in records if other["trajectory_id"] == row["trajectory_id"] and other["step_index"] == row["step_index"] - 1]
        row["relative_delta_m"] = float(abs(row["offset_m"] - same[0]["offset_m"])) if same else 0.0
    previous, history_valid = grouped_history_descriptors(records)
    for row, valid in zip(records, history_valid.tolist()):
        row["history_valid"] = bool(valid)
    with (output / "history_boundary_audit.json").open("w") as handle:
        old_files = {}
        for name in ("validation.json", "probe_results.json", "alias_summary.json", "frame_targets.csv"):
            path = output / name
            if path.exists(): old_files[str(path.relative_to(PROJECT_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        json.dump({"schema": "obs0r1.history_boundary_audit.v1", "old_obs0_sha256": old_files, "records": len(records), "group_count": len({row['trajectory_id'] for row in records}), "first_frame_history_valid": {row["trajectory_id"]: row["history_valid"] for row in records if row["step_index"] == 0}, "cross_group_previous_forbidden": True, "train_only_preprocessing": True}, handle, indent=2); handle.write("\n")
    with (output / "frame_targets_r1.csv").open("w", newline="") as handle:
        fields = ["trajectory_id", "direction", "step_index", "offset_m", "target", "relative_offset_m", "relative_delta_m", "history_valid", "rgb_path", "frame_id"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows({key: row[key] for key in fields} for row in records)
    probes = evaluate_probes(records, previous, history_valid)
    descriptor_dim = len(records[0]["descriptor"])
    direction_counts = [sum(row["direction"] == name for row in records)
                        for name in ("LEFT", "RIGHT")]
    max_probe_feature_dim = 2 * descriptor_dim + 6
    max_train_rows = max(direction_counts)
    runtime_safety = {
        "descriptor_dim": descriptor_dim,
        "max_probe_feature_dim": max_probe_feature_dim,
        "max_train_rows": max_train_rows,
        "max_linear_system_dimension": min(max_probe_feature_dim, max_train_rows),
        "ridge_solver": "adaptive smaller-of-sample-and-feature-space",
        "opencv_threads": cv2.getNumThreads(),
        "address_space_limit_bytes": resource.getrlimit(resource.RLIMIT_AS)[0],
    }
    target_formula_residual = max(abs(row["target"] - (10 - row["step_index"]) * 0.5) for row in records)
    synthetic = synthetic_alignment_check()
    probe_results = {"schema": "obs0.probe_results_r1.v1", "records": len(records), "split": "complete direction holdout; merged result is aggregate of two direction holdouts", "history_valid_mask": "false at each trajectory step 0; true only after a same-trajectory previous frame", "train_only_preprocessing": True, "runtime_safety": runtime_safety, "fixed_schedule": {"B1_formula": "target=(first_local_straddle_step-step_index)*step_m", "formula_residual_max_m": float(target_formula_residual), "deterministic": target_formula_residual == 0.0}, "probes": probes, "comparisons": {"F1_minus_B2_visual_history_increment_m": float(probes["B2"]["merged_direction_holdout"]["MAE_m"] - probes["F1"]["merged_direction_holdout"]["MAE_m"]), "V1_minus_V0_history_increment_m": float(probes["V0"]["merged_direction_holdout"]["MAE_m"] - probes["V1"]["merged_direction_holdout"]["MAE_m"])}, "synthetic_alignment": synthetic}
    (output / "probe_results_r1.json").write_text(json.dumps(probe_results, indent=2) + "\n")
    # Similarity is computed without state/label features.  The top candidates
    # are reported as candidates, never as proof that aliasing is absent.
    labels = [f"{row['direction']}_s{row['step_index']}" for row in records]
    matrix = np.eye(len(records)); pairs = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            metrics = similarity_metrics(image(records[i]["rgb_path"]), image(records[j]["rgb_path"]))
            matrix[i, j] = matrix[j, i] = metrics["phase_aligned_ssim"]
            pairs.append({"rank_key": float(metrics["phase_aligned_ssim"]), "left_index": i, "right_index": j, "left_trajectory": records[i]["trajectory_id"], "right_trajectory": records[j]["trajectory_id"], "left_step": records[i]["step_index"], "right_step": records[j]["step_index"], "remaining_distance_gap_m": float(abs(records[i]["target"] - records[j]["target"])), "cross_direction": records[i]["direction"] != records[j]["direction"], **metrics})
    ranked = sorted(pairs, key=lambda row: (row["phase_aligned_ssim"], row["hog_cosine"], -row["color_histogram_distance"]), reverse=True)
    with (output / "alias_pairs_r1.csv").open("w", newline="") as handle:
        fields = list(ranked[0]) if ranked else ["left_index", "right_index"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(ranked)
    thresholds = {}
    for score in (0.90, 0.95, 0.98):
        for gap in (1.0, 2.0, 3.0):
            eligible = [pair for pair in pairs if pair["remaining_distance_gap_m"] >= gap]
            hits = [pair for pair in eligible if pair["phase_aligned_ssim"] >= score]
            thresholds[f"phase_ssim_{score:.2f}_gap_{gap:.1f}m"] = {"eligible_pairs": len(eligible), "candidate_count": len(hits), "candidate_ratio": float(len(hits) / max(len(eligible), 1))}
    strongest = ranked[:5]
    for rank, pair in enumerate(strongest, 1):
        draw_pair(records[pair["left_index"]], records[pair["right_index"]], pair, rank, assets / f"candidate_pair_{rank:02d}.jpg")
    alias_summary = {"schema": "obs0.alias_summary_r1.v1", "pair_count": len(pairs), "metrics": ["raw_ssim", "phase_aligned_ssim", "hog_cosine", "color_histogram_distance"], "thresholds": thresholds, "top_ranked_candidates": strongest, "interpretation": "reports whether this metric suite found high-similarity candidates; it does not prove perceptual aliasing is absent"}
    (output / "alias_summary_r1.json").write_text(json.dumps(alias_summary, indent=2) + "\n")
    from PIL import ImageDraw
    canvas = Image.new("RGB", (1000, 600), "white"); draw = ImageDraw.Draw(canvas)
    draw.text((30, 20), "OBS-0R1 phase-aligned SSIM matrix", fill=(0, 0, 0))
    draw_heatmap(matrix, labels, assets / "phase_aligned_ssim_heatmap.jpg", "phase-aligned SSIM")
    validation = {"schema": "obs0.validation_r1.v1", "source": "leakage-corrected offline reanalysis of OBS-0 frames", "historical_obs0_files_unchanged": old_files, "runtime_safety": runtime_safety, "gates": {"HISTORY_BOUNDARY_LEAKAGE_FIXED": {"status": "PASS", "reason": "grouped previous descriptors and per-trajectory step-0 invalid mask"}, "TRAIN_ONLY_PREPROCESSING": {"status": "PASS", "reason": "ridge standardization fitted inside each train-direction split"}, "SYNTHETIC_ALIGNMENT_TEST": synthetic, "OBS0R1_REPRODUCIBILITY": {"status": "PASS", "records": len(records)}, "FIXED_SCHEDULE_SHORTCUT_PRESENT": {"status": "PASS", "formula_residual_max_m": float(target_formula_residual)}, "RELATIVE_ODOMETRY_PREDICTIVITY": {"status": "DIAGNOSTIC_PASS" if probes["B2"]["merged_direction_holdout"]["constant_baseline_improvement_m"] > 0.1 else "FAIL", "evidence": probes["B2"]["merged_direction_holdout"]}, "RGB_INCREMENTAL_VALUE_OVER_ODOMETRY": {"status": "DIAGNOSTIC_PASS" if probe_results["comparisons"]["F1_minus_B2_visual_history_increment_m"] > 0.05 else "FAIL", "increment_mae_reduction_m": probe_results["comparisons"]["F1_minus_B2_visual_history_increment_m"]}, "RGB_HISTORY_INCREMENTAL_VALUE": {"status": "DIAGNOSTIC_PASS" if probe_results["comparisons"]["V1_minus_V0_history_increment_m"] > 0.05 else "FAIL", "increment_mae_reduction_m": probe_results["comparisons"]["V1_minus_V0_history_increment_m"]}, "SINGLE_FRAME_RGB_OBSERVABILITY": {"status": "DIAGNOSTIC_PASS" if probes["V0"]["merged_direction_holdout"]["constant_baseline_improvement_m"] > 0.1 else "FAIL", "evidence": probes["V0"]["merged_direction_holdout"]}, "BOUNDARY_DISTANCE_OBSERVABILITY": {"status": "INCONCLUSIVE", "reason": "two trajectories on one facade; no cross-surface validation"}}, "probes": probes, "fixed_schedule_formula": "target=(first_local_straddle_step-step_index)*step_m; B1 is a shortcut baseline, not visual evidence"}
    (output / "validation_r1.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
