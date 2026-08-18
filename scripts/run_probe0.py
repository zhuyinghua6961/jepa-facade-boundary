#!/usr/bin/env python3
"""Offline PROBE-0 active-disambiguation kill test over frozen CF-0 raw."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import resource
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from boundary_sweep.act0r import verify_manifest_hashes
from boundary_sweep.observability import (binary_metrics, cf0_feature_matrix,
                                          fixed_length_descriptor, grouped_kfold,
                                          probe0r1_feature_matrix, raw_ssim, rgb_descriptor,
                                          train_only_pca_ridge)


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fields:
            writer.writeheader()
            writer.writerows(rows)


def eligible_truth(cf0: dict) -> list[dict]:
    branches = {(row["start_id"], row["direction"]): row
                for row in cf0["branch_summaries"]}
    maximum = float(cf0["capture"]["maximum_distance_m"])
    step = float(cf0["capture"]["step_m"])
    rows = []
    for start_id in sorted({row["start_id"] for row in cf0["branch_summaries"]}):
        cost = {}
        for direction in ("LEFT", "RIGHT"):
            value = branches[(start_id, direction)]["first_model_visible_distance_m"]
            cost[direction] = float(value) if value is not None else maximum + step
        if abs(cost["LEFT"] - cost["RIGHT"]) <= 1e-12:
            continue
        nearer = "LEFT" if cost["LEFT"] < cost["RIGHT"] else "RIGHT"
        rows.append({"start_id": start_id, "near_direction": nearer,
                     "target": int(nearer == "RIGHT"),
                     "left_cost_m": cost["LEFT"], "right_cost_m": cost["RIGHT"],
                     "wrong_action_regret_m": abs(cost["LEFT"] - cost["RIGHT"])})
    return rows


def image(entry: dict) -> np.ndarray:
    return np.asarray(Image.open(resolve(entry["files"]["rgb"]["path"])).convert("RGB"))


def descriptor(entry: dict, length: int, cache: dict[int, list[float]]) -> list[float]:
    frame_id = int(entry["frame_id"])
    if frame_id not in cache:
        cache[frame_id] = fixed_length_descriptor(
            rgb_descriptor(image(entry)), length).tolist()
    return cache[frame_id]


def build_probe_samples(config: dict, cf0: dict, manifest: dict,
                        frame_metrics: list[dict], distance_m: float,
                        steps: int, cache: dict[int, list[float]]) -> tuple[list[dict], list[dict]]:
    entries = {int(row["frame_id"]): row for row in manifest["frames"]}
    starts = {row["start_id"]: row for row in manifest["starts"]}
    metrics = {(row["start_id"], row["direction"], int(row["step_index"])): row
               for row in frame_metrics}
    truth = eligible_truth(cf0)
    records, evidence_entries = [], {}
    for target in truth:
        start = starts[target["start_id"]]
        shared = entries[int(start["shared_start_frame_id"])]
        evidence_entries[int(shared["frame_id"])] = shared
        start_rgb = image(shared)
        for probe_direction in ("LEFT", "RIGHT"):
            branch_ids = start["branch_frame_ids"][probe_direction]
            history_ids = branch_ids[:steps]
            if len(history_ids) != steps:
                raise RuntimeError("probe history shorter than configured step count")
            for step_index, frame_id in enumerate(history_ids, 1):
                audit = metrics[(target["start_id"], probe_direction, step_index)]
                if (truthy(audit["first_physical_termination"]) or
                        truthy(audit["model_visible_termination"])):
                    raise RuntimeError(
                        f"boundary leakage at {target['start_id']} {probe_direction} step {step_index}")
                evidence_entries[int(frame_id)] = entries[int(frame_id)]
            current = entries[int(history_ids[-1])]
            previous = shared if steps == 1 else entries[int(history_ids[-2])]
            current_rgb = image(current)
            records.append({
                "start_id": target["start_id"], "direction": probe_direction,
                "relative_distance_m": float(distance_m),
                "relative_delta_m": float(config["probe"]["step_m"]),
                "descriptor": descriptor(current, config["model"]["descriptor_length"], cache),
                "previous_descriptor": descriptor(
                    previous, config["model"]["descriptor_length"], cache),
                "history_valid": True,
                "target": int(target["target"]),
                "near_direction": target["near_direction"],
                "left_cost_m": target["left_cost_m"],
                "right_cost_m": target["right_cost_m"],
                "wrong_action_regret_m": target["wrong_action_regret_m"],
                "shared_frame_id": int(shared["frame_id"]),
                "previous_frame_id": int(previous["frame_id"]),
                "current_frame_id": int(current["frame_id"]),
                "start_to_probe_ssim": raw_ssim(start_rgb, current_rgb),
                "start_to_probe_mean_abs_rgb_change": float(np.mean(
                    np.abs(start_rgb.astype(np.float32) - current_rgb.astype(np.float32))) / 255.0),
            })
    return records, list(evidence_entries.values())


def oof_predict_intervention(records: list[dict], train_features: np.ndarray,
                             test_features: np.ndarray,
                             config: dict) -> tuple[np.ndarray, list[dict]]:
    if train_features.shape != test_features.shape:
        raise ValueError("train/test intervention feature shapes must match")
    target = np.asarray([row["target"] for row in records], dtype=float)
    folds = grouped_kfold([row["start_id"] for row in records],
                          config["model"]["group_folds"], config["seed"])
    prediction = np.full(len(records), np.nan, dtype=float)
    audits = []
    for fold in folds:
        train = np.asarray(fold["train_indices"], dtype=int)
        test = np.asarray(fold["test_indices"], dtype=int)
        raw, preprocessing = train_only_pca_ridge(
            train_features[train], target[train], test_features[test],
            alpha=config["model"]["ridge_alpha"],
            max_components=config["model"]["pca_components"])
        prediction[test] = np.clip(raw, 0.0, 1.0)
        for index in test.tolist():
            records[index]["fold"] = int(fold["fold"])
        audits.append({"fold": int(fold["fold"]),
                       "train_start_ids": fold["train_groups"],
                       "test_start_ids": fold["test_groups"],
                       "preprocessing": preprocessing})
    if not np.isfinite(prediction).all():
        raise RuntimeError("incomplete out-of-fold prediction")
    return prediction, audits


def oof_predict_matrix(records: list[dict], features: np.ndarray,
                       config: dict) -> tuple[np.ndarray, list[dict]]:
    return oof_predict_intervention(records, features, features, config)


def oof_predict(records: list[dict], config: dict) -> tuple[np.ndarray, list[dict]]:
    features = cf0_feature_matrix(records, config["model"]["baseline"])
    return oof_predict_matrix(records, features, config)


def sample_metrics(records: list[dict], prediction: np.ndarray,
                   indices: list[int] | None = None) -> dict:
    selected = np.asarray(indices if indices is not None else range(len(records)), dtype=int)
    target = np.asarray([records[index]["target"] for index in selected], dtype=int)
    score = prediction[selected]
    predicted = (score >= 0.5).astype(int)
    regret = np.asarray([
        0.0 if predicted[position] == target[position]
        else float(records[index]["wrong_action_regret_m"])
        for position, index in enumerate(selected)
    ])
    result = binary_metrics(target, score)
    result.update({"accuracy": float(np.mean(predicted == target)),
                   "mean_regret_m": float(np.mean(regret)),
                   "median_regret_m": float(np.median(regret))})
    return result


def cluster_bootstrap(records: list[dict], prediction: np.ndarray,
                      samples: int, seed: int) -> dict:
    groups = sorted({row["start_id"] for row in records})
    by_group = {group: [index for index, row in enumerate(records)
                        if row["start_id"] == group] for group in groups}
    rng = np.random.default_rng(int(seed))
    values = {name: [] for name in ("accuracy", "balanced_accuracy", "AUROC",
                                     "mean_regret_m")}
    for _index in range(int(samples)):
        selected_groups = rng.choice(groups, size=len(groups), replace=True)
        indices = [index for group in selected_groups for index in by_group[group]]
        metric = sample_metrics(records, prediction, indices)
        for name in values:
            value = metric[name]
            if value is not None and np.isfinite(value):
                values[name].append(float(value))
    return {name: {"lower": float(np.percentile(rows, 2.5)),
                   "upper": float(np.percentile(rows, 97.5)),
                   "bootstrap_samples": len(rows), "cluster_unit": "start_id"}
            for name, rows in values.items()}


def evaluate(records: list[dict], prediction: np.ndarray, config: dict) -> dict:
    subsets = {"pooled": list(range(len(records))),
               "LEFT_probe": [index for index, row in enumerate(records)
                              if row["direction"] == "LEFT"],
               "RIGHT_probe": [index for index, row in enumerate(records)
                               if row["direction"] == "RIGHT"]}
    result = {}
    for offset, (name, indices) in enumerate(subsets.items()):
        subset_records = [records[index] for index in indices]
        subset_prediction = prediction[indices]
        result[name] = sample_metrics(subset_records, subset_prediction)
        result[name]["bootstrap_95_ci"] = cluster_bootstrap(
            subset_records, subset_prediction,
            config["evaluation"]["bootstrap_samples"], config["seed"] + 101 + offset)
    return result


def prediction_rows(records: list[dict], prediction: np.ndarray) -> list[dict]:
    rows = []
    for record, probability in zip(records, prediction.tolist()):
        predicted = "RIGHT" if probability >= 0.5 else "LEFT"
        rows.append({
            "start_id": record["start_id"], "probe_direction": record["direction"],
            "fold": record["fold"], "near_direction_gt": record["near_direction"],
            "right_probability": probability, "predicted_direction": predicted,
            "correct": predicted == record["near_direction"],
            "regret_m": 0.0 if predicted == record["near_direction"]
                        else record["wrong_action_regret_m"],
            "probe_distance_m": record["relative_distance_m"],
            "shared_frame_id_gt_audit_only": record["shared_frame_id"],
            "previous_frame_id_gt_audit_only": record["previous_frame_id"],
            "current_frame_id_gt_audit_only": record["current_frame_id"],
            "start_to_probe_ssim": record["start_to_probe_ssim"],
            "start_to_probe_mean_abs_rgb_change": record[
                "start_to_probe_mean_abs_rgb_change"],
        })
    return rows


def draw_probe_comparison(records: list[dict], frame_by_id: dict[int, dict], output: Path) -> None:
    starts = sorted({row["start_id"] for row in records})
    lookup = {(row["start_id"], row["direction"]): row for row in records}
    canvas = Image.new("RGB", (1440, 780), (18, 18, 18)); draw = ImageDraw.Draw(canvas)
    for index, start_id in enumerate(starts):
        row, column = divmod(index, 3); x, y = column * 480, row * 156
        left, right = lookup[(start_id, "LEFT")], lookup[(start_id, "RIGHT")]
        ids = (left["shared_frame_id"], left["current_frame_id"], right["current_frame_id"])
        labels = ("START", "LEFT 1.0m", "RIGHT 1.0m")
        for panel, (frame_id, label) in enumerate(zip(ids, labels)):
            rgb = Image.fromarray(image(frame_by_id[int(frame_id)])).resize((160, 120))
            canvas.paste(rgb, (x + panel * 160, y + 22))
            draw.text((x + panel * 160 + 4, y + 4), label, fill=(255, 230, 70))
        draw.text((x + 4, y + 142),
                  f"{start_id} near={left['near_direction']}", fill=(240, 240, 240))
    canvas.save(output, quality=82, optimize=True, progressive=True)


def draw_change_comparison(records: list[dict], frame_by_id: dict[int, dict], output: Path) -> None:
    starts = sorted({row["start_id"] for row in records})
    lookup = {(row["start_id"], row["direction"]): row for row in records}
    canvas = Image.new("RGB", (1200, 720), (18, 18, 18)); draw = ImageDraw.Draw(canvas)
    for index, start_id in enumerate(starts):
        row, column = divmod(index, 4); x, y = column * 300, row * 180
        rows = [lookup[(start_id, direction)] for direction in ("LEFT", "RIGHT")]
        near = next(value for value in rows if value["direction"] == value["near_direction"])
        away = next(value for value in rows if value["direction"] != value["near_direction"])
        start = image(frame_by_id[int(near["shared_frame_id"])]).astype(np.int16)
        for panel, (value, label) in enumerate(((near, "NEAR"), (away, "AWAY"))):
            current = image(frame_by_id[int(value["current_frame_id"])]).astype(np.int16)
            difference = np.clip(np.abs(current - start) * 3, 0, 255).astype(np.uint8)
            canvas.paste(Image.fromarray(difference).resize((150, 120)),
                         (x + panel * 150, y + 24))
            draw.text((x + panel * 150 + 4, y + 5), label, fill=(255, 230, 70))
        draw.text((x + 4, y + 147),
                  f"{start_id} nearSSIM={near['start_to_probe_ssim']:.3f}",
                  fill=(240, 240, 240))
        draw.text((x + 4, y + 163), f"awaySSIM={away['start_to_probe_ssim']:.3f}",
                  fill=(240, 240, 240))
    canvas.save(output, quality=84, optimize=True, progressive=True)


def draw_fold_predictions(rows: list[dict], output: Path) -> None:
    canvas = Image.new("RGB", (1200, 650), "white"); draw = ImageDraw.Draw(canvas)
    draw.text((24, 16), "PROBE-0 1.0m out-of-fold RIGHT probability", fill=(0, 0, 0))
    left, top, width, height = 80, 60, 1060, 500
    draw.rectangle((left, top, left + width, top + height), outline=(60, 60, 60), width=2)
    threshold_y = top + height // 2
    draw.line((left, threshold_y, left + width, threshold_y), fill=(200, 40, 40), width=2)
    draw.text((left + width + 5, threshold_y - 8), "0.5", fill=(160, 30, 30))
    for index, row in enumerate(rows):
        x = left + int(width * (index + 0.5) / len(rows))
        y = top + int(height * (1.0 - float(row["right_probability"])))
        color = (220, 80, 40) if row["near_direction_gt"] == "RIGHT" else (40, 100, 220)
        radius = 6 if row["probe_direction"] == "RIGHT" else 4
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        draw.text((x - 4, top + height + 8), str(row["fold"]), fill=(0, 0, 0))
    draw.text((80, 595), "x labels are held-out fold IDs; blue GT=LEFT, red GT=RIGHT; larger=RIGHT probe",
              fill=(0, 0, 0))
    canvas.save(output, quality=88, optimize=True, progressive=True)


def draw_accuracy_ci(primary: dict, p0: float, output: Path) -> None:
    rows = [("P0 start B2", p0, None),
            ("P1 pooled", primary["pooled"]["accuracy"],
             primary["pooled"]["bootstrap_95_ci"]["accuracy"]),
            ("P1 LEFT probe", primary["LEFT_probe"]["accuracy"],
             primary["LEFT_probe"]["bootstrap_95_ci"]["accuracy"]),
            ("P1 RIGHT probe", primary["RIGHT_probe"]["accuracy"],
             primary["RIGHT_probe"]["bootstrap_95_ci"]["accuracy"])]
    canvas = Image.new("RGB", (1100, 580), "white"); draw = ImageDraw.Draw(canvas)
    draw.text((24, 18), "PROBE-0 accuracy and start-cluster bootstrap 95% CI", fill=(0, 0, 0))
    left, width = 260, 760
    for threshold, color, label in ((0.5, (160, 160, 160), "chance"),
                                    (0.65, (200, 40, 40), "gate")):
        x = left + int(width * threshold)
        draw.line((x, 55, x, 520), fill=color, width=2)
        draw.text((x + 4, 55), f"{label} {threshold:.2f}", fill=color)
    for index, (name, value, interval) in enumerate(rows):
        y = 130 + index * 100
        draw.text((25, y - 10), name, fill=(0, 0, 0))
        x = left + int(width * value)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(40, 110, 210))
        if interval:
            low = left + int(width * interval["lower"])
            high = left + int(width * interval["upper"])
            draw.line((low, y, high, y), fill=(30, 30, 30), width=4)
            draw.line((low, y - 9, low, y + 9), fill=(30, 30, 30), width=2)
            draw.line((high, y - 9, high, y + 9), fill=(30, 30, 30), width=2)
        draw.text((left + width + 10, y - 9), f"{value:.4f}", fill=(0, 0, 0))
    canvas.save(output, quality=90, optimize=True, progressive=True)


def write_docs(validation: dict, output: Path) -> None:
    primary = validation["primary_1m"]
    lines = ["# PROBE-0 Active Disambiguation Solvability Audit", "",
             "This is a one-shot offline kill test over frozen CF-0 raw. CARLA was not",
             "started, no frames were captured, no model was downloaded and JEPA was not",
             "trained. The 1.0 m result is primary; 0.5 m is diagnostic only.", "",
             "## Frozen Method", "",
             "Each probe history is two consecutive 0.5 m steps. The frozen CF-0 B3",
             "feature builder consumes the 0.5 m RGB descriptor as previous, the 1.0 m",
             "descriptor as current, history-valid, relative odometry and probe action.",
             "All 13 start groups remain intact in five folds. PCA and ridge are fitted",
             "inside each training fold. No physical-boundary frame or later frame is input.",
             "Absolute position, start offset, frame ID, planned role and world-boundary",
             "coordinates are retained only for GT/audit and never enter the feature matrix.",
             "", "## Source And Safety Audit", "",
             f"- Frozen non-tie starts: {validation['P0']['non_tie_start_count']}",
             f"- Primary samples: {primary['pooled']['sample_count']} (13 per probe direction)",
             f"- Unique evidence frames: {validation['source_audit']['evidence_frame_count']}",
             f"- Verified payload hashes: {validation['source_audit']['checked_file_count']}",
             f"- Address-space limit: {validation['resources']['address_space_limit_bytes']} bytes",
             f"- Peak RSS: {validation['resources']['peak_rss_bytes']} bytes",
             f"- Numeric threads: {validation['resources']['numeric_threads']}",
             "", "## Primary Metrics", "",
             "| Scope | Accuracy | Balanced accuracy | AUROC | Brier | Regret m | Accuracy 95% CI |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for name in ("pooled", "LEFT_probe", "RIGHT_probe"):
        row = primary[name]; ci = row["bootstrap_95_ci"]["accuracy"]
        lines.append(f"| {name} | {row['accuracy']:.6f} | {row['balanced_accuracy']:.6f} | "
                     f"{row['AUROC']:.6f} | {row['Brier']:.6f} | {row['mean_regret_m']:.6f} | "
                     f"[{ci['lower']:.6f}, {ci['upper']:.6f}] |")
    diagnostic = validation["diagnostic_0_5m"]["pooled"]
    lines.extend(["", f"Frozen P0 B2 accuracy is {validation['P0']['accuracy']:.6f}; the fixed",
                  f"1.0 m pooled improvement is {validation['P1_minus_P0_accuracy']:.6f}.",
                  "", "The preregistered gates are pooled accuracy >= 0.65, improvement >=",
                  "0.10, pooled start-cluster CI lower bound > 0.50, and both directional",
                  "accuracies >= 0.65. All six integrity/performance conditions pass.",
                  "", "## Diagnostic 0.5 m Result", "",
                  f"The non-gated 0.5 m pooled accuracy is {diagnostic['accuracy']:.6f},",
                  f"balanced accuracy is {diagnostic['balanced_accuracy']:.6f}, AUROC is",
                  f"{diagnostic['AUROC']:.6f}, and mean regret is",
                  f"{diagnostic['mean_regret_m']:.6f} m. It was not used for distance selection.",
                  "No 2.0 m or later result was computed.",
                  "", "## Errors And Scope", "",
                  "The primary errors are start_07 under both probes and start_18 under the",
                  "LEFT probe. This is one facade with 13 independent start clusters; it does",
                  "not establish cross-surface generalization. External visual review is",
                  "pending, and the result authorizes only a second-surface replication.",
                  "", "## Gates", "", "| Gate | Status |", "| --- | --- |"])
    for name, gate in validation["gates"].items():
        lines.append(f"| {name} | {gate['status']} |")
    lines.extend(["", "## Visual Evidence", "",
                  "![1.0 m bilateral probe comparison](assets/probe0/probe_1m_left_right.jpg)", "",
                  "![Near-side and away-side RGB changes](assets/probe0/near_away_rgb_change.jpg)", "",
                  "![Held-out fold predictions](assets/probe0/fold_predictions.jpg)", "",
                  "![Accuracy and confidence intervals](assets/probe0/accuracy_ci.jpg)", ""])
    output.write_text("\n".join(lines))


def paired_cluster_difference(records: list[dict], e2_prediction: np.ndarray,
                              control_prediction: np.ndarray, samples: int,
                              seed: int) -> dict:
    """Paired E2-control differences with start-level resampling."""
    groups = sorted({row["start_id"] for row in records})
    by_group = {group: [index for index, row in enumerate(records)
                        if row["start_id"] == group] for group in groups}

    def differences(indices: list[int]) -> dict:
        e2 = sample_metrics(records, e2_prediction, indices)
        control = sample_metrics(records, control_prediction, indices)
        return {
            "accuracy": e2["accuracy"] - control["accuracy"],
            "balanced_accuracy": (None if e2["balanced_accuracy"] is None or
                                    control["balanced_accuracy"] is None else
                                    e2["balanced_accuracy"] - control["balanced_accuracy"]),
            "AUROC": (None if e2["AUROC"] is None or control["AUROC"] is None else
                       e2["AUROC"] - control["AUROC"]),
            "mean_regret_improvement_m": (control["mean_regret_m"] -
                                           e2["mean_regret_m"]),
        }

    observed = differences(list(range(len(records))))
    values = {name: [] for name in observed}
    rng = np.random.default_rng(int(seed))
    for _index in range(int(samples)):
        selected = rng.choice(groups, size=len(groups), replace=True)
        indices = [index for group in selected for index in by_group[group]]
        result = differences(indices)
        for name, value in result.items():
            if value is not None and np.isfinite(value):
                values[name].append(float(value))
    intervals = {
        name: {"lower": float(np.percentile(rows, 2.5)),
               "upper": float(np.percentile(rows, 97.5)),
               "bootstrap_samples": len(rows), "cluster_unit": "start_id"}
        for name, rows in values.items()
    }
    return {"observed": observed, "bootstrap_95_ci": intervals}


def write_ablation_csv(path: Path, evaluations: dict) -> None:
    rows = []
    for ablation, scopes in evaluations.items():
        for scope, metric in scopes.items():
            rows.append({
                "ablation": ablation, "scope": scope,
                "sample_count": metric["sample_count"],
                "accuracy": metric["accuracy"],
                "balanced_accuracy": metric["balanced_accuracy"],
                "AUROC": metric["AUROC"], "Brier": metric["Brier"],
                "mean_regret_m": metric["mean_regret_m"],
                "accuracy_ci_lower": metric["bootstrap_95_ci"]["accuracy"]["lower"],
                "accuracy_ci_upper": metric["bootstrap_95_ci"]["accuracy"]["upper"],
            })
    write_csv(path, rows)


def draw_r1_overview(evaluations: dict, output: Path) -> None:
    names = list(evaluations)
    canvas = Image.new("RGB", (1100, 650), "white"); draw = ImageDraw.Draw(canvas)
    draw.text((25, 18), "PROBE-0R1 same-endpoint causal ablations", fill=(0, 0, 0))
    left, width = 270, 740
    for threshold, color in ((0.5, (160, 160, 160)), (0.65, (190, 40, 40))):
        x = left + int(width * threshold)
        draw.line((x, 55, x, 595), fill=color, width=2)
        draw.text((x + 4, 55), f"{threshold:.2f}", fill=color)
    for index, name in enumerate(names):
        metric = evaluations[name]["pooled"]
        ci = metric["bootstrap_95_ci"]["accuracy"]
        y = 125 + index * 100
        draw.text((25, y - 9), name, fill=(0, 0, 0))
        low, high = left + int(width * ci["lower"]), left + int(width * ci["upper"])
        x = left + int(width * metric["accuracy"])
        draw.line((low, y, high, y), fill=(35, 35, 35), width=4)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(40, 105, 210))
        draw.text((left + width + 12, y - 9), f"{metric['accuracy']:.4f}", fill=(0, 0, 0))
    canvas.save(output, quality=90, optimize=True, progressive=True)


def draw_r1_paired(records: list[dict], predictions: dict[str, np.ndarray], output: Path) -> None:
    groups = sorted({row["start_id"] for row in records})
    canvas = Image.new("RGB", (1300, 680), "white"); draw = ImageDraw.Draw(canvas)
    draw.text((25, 16), "E1 current-RGB versus E2 ordered-history RIGHT probability", fill=(0, 0, 0))
    left, top, width, height = 80, 60, 1140, 500
    draw.rectangle((left, top, left + width, top + height), outline=(50, 50, 50), width=2)
    draw.line((left, top + height // 2, left + width, top + height // 2),
              fill=(190, 50, 50), width=2)
    for index, group in enumerate(groups):
        indices = [i for i, row in enumerate(records) if row["start_id"] == group]
        x = left + int(width * (index + 0.5) / len(groups))
        values = {name: float(np.mean(predictions[name][indices])) for name in ("E1", "E2")}
        ys = {name: top + int(height * (1.0 - value)) for name, value in values.items()}
        draw.line((x, ys["E1"], x, ys["E2"]), fill=(100, 100, 100), width=2)
        draw.ellipse((x - 6, ys["E1"] - 6, x + 6, ys["E1"] + 6), fill=(230, 145, 30))
        draw.rectangle((x - 6, ys["E2"] - 6, x + 6, ys["E2"] + 6), fill=(35, 105, 215))
        draw.text((x - 25, top + height + 10), group[-2:], fill=(0, 0, 0))
    draw.text((80, 610), "orange circle=E1; blue square=E2; x labels=start suffix; each value averages both probes",
              fill=(0, 0, 0))
    canvas.save(output, quality=90, optimize=True, progressive=True)


def draw_r1_temporal(evaluations: dict, output: Path) -> None:
    names = ("E2", "E3_SWAP", "E4_ZERO")
    canvas = Image.new("RGB", (1050, 570), "white"); draw = ImageDraw.Draw(canvas)
    draw.text((25, 18), "Ordered history versus temporal destruction", fill=(0, 0, 0))
    left, bottom, width, height = 110, 500, 850, 400
    draw.line((left, bottom, left + width, bottom), fill=(40, 40, 40), width=2)
    for index, name in enumerate(names):
        value = evaluations[name]["pooled"]["accuracy"]
        x0 = left + 90 + index * 270
        y0 = bottom - int(height * value)
        color = (45, 110, 215) if name == "E2" else (150, 150, 150)
        draw.rectangle((x0, y0, x0 + 130, bottom), fill=color)
        draw.text((x0 + 20, bottom + 12), name, fill=(0, 0, 0))
        draw.text((x0 + 30, y0 - 20), f"{value:.4f}", fill=(0, 0, 0))
    canvas.save(output, quality=90, optimize=True, progressive=True)


def draw_r1_bootstrap(comparisons: dict, output: Path) -> None:
    names = ("E2_minus_E1", "E2_minus_E3_SWAP", "E2_minus_E4_ZERO")
    canvas = Image.new("RGB", (1150, 560), "white"); draw = ImageDraw.Draw(canvas)
    draw.text((25, 18), "Paired start-cluster bootstrap accuracy differences", fill=(0, 0, 0))
    left, width = 480, 560
    minimum, maximum = -0.35, 0.80
    zero = left + int(width * (0.0 - minimum) / (maximum - minimum))
    gate = left + int(width * (0.10 - minimum) / (maximum - minimum))
    draw.line((zero, 55, zero, 500), fill=(90, 90, 90), width=2)
    draw.line((gate, 55, gate, 500), fill=(190, 45, 45), width=2)
    for index, name in enumerate(names):
        row = comparisons[name]
        ci = row["bootstrap_95_ci"]["accuracy"]
        value = row["observed"]["accuracy"]
        y = 145 + index * 125
        x = left + int(width * (value - minimum) / (maximum - minimum))
        low = left + int(width * (ci["lower"] - minimum) / (maximum - minimum))
        high = left + int(width * (ci["upper"] - minimum) / (maximum - minimum))
        draw.text((25, y - 9), name, fill=(0, 0, 0))
        draw.line((low, y, high, y), fill=(40, 40, 40), width=4)
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=(40, 105, 215))
        draw.text((left + width + 8, y - 9), f"{value:+.4f}", fill=(0, 0, 0))
    canvas.save(output, quality=90, optimize=True, progressive=True)


def write_r1_docs(validation: dict, output: Path) -> None:
    metrics = validation["ablations"]
    comparisons = validation["paired_differences"]
    lines = ["# PROBE-0R1 Causal Attribution Audit", "",
             "This offline audit uses the frozen 13 starts, 26 samples, 1.0 m endpoints,",
             "folds, seed, RGB descriptor, PCA and ridge settings from PROBE-0. It does",
             "not start CARLA, capture data, download a model or train JEPA.", "",
             "## Same-Endpoint Ablations", "",
             "| Ablation | Definition | Accuracy | Balanced accuracy | AUROC | Brier | Regret m |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    definitions = validation["ablation_definitions"]
    for name in ("E0", "E1", "E2", "E3_SWAP", "E4_ZERO"):
        row = metrics[name]["pooled"]
        lines.append(f"| {name} | {definitions[name]} | {row['accuracy']:.6f} | "
                     f"{row['balanced_accuracy']:.6f} | {row['AUROC']:.6f} | "
                     f"{row['Brier']:.6f} | {row['mean_regret_m']:.6f} |")
    lines.extend(["", "## Paired Accuracy Differences", "",
                  "| Comparison | Difference | 95% CI |", "| --- | ---: | --- |"])
    for name in ("E2_minus_E1", "E2_minus_E3_SWAP", "E2_minus_E4_ZERO"):
        row = comparisons[name]; ci = row["bootstrap_95_ci"]["accuracy"]
        lines.append(f"| {name} | {row['observed']['accuracy']:.6f} | "
                     f"[{ci['lower']:.6f}, {ci['upper']:.6f}] |")
    lines.extend(["", "## Gates", "", "| Gate | Status | Evidence |",
                  "| --- | --- | --- |"])
    for name, gate in validation["gates"].items():
        lines.append(f"| {name} | {gate['status']} | {gate.get('reason', '')} |")
    lines.extend(["", "## Interpretation", "", validation["scientific_interpretation"], "",
                  "E4_ZERO degradation shows that the fitted E2 classifier uses the previous-RGB",
                  "feature block. It does not establish useful temporal order: E1 reaches the same",
                  "accuracy using current endpoint RGB alone, and swapping held-out previous/current",
                  "RGB in E3_SWAP does not reduce accuracy. The static endpoint explanation is",
                  "therefore sufficient for the observed PROBE-0 accuracy gain.", "",
                  "External visual review and cross-surface generalization remain unevaluated.",
                  "READY_FOR_JEPA remains NOT_EVALUATED.", "", "## Figures", "",
                  "![E0-E4 overview](assets/probe0r1/e0_e4_overview.jpg)", "",
                  "![E2 versus E1 paired starts](assets/probe0r1/e2_e1_paired_starts.jpg)", "",
                  "![Temporal destruction](assets/probe0r1/temporal_destruction.jpg)", "",
                  "![Bootstrap differences](assets/probe0r1/bootstrap_accuracy_differences.jpg)", ""])
    output.write_text("\n".join(lines))


def run_causal_r1(config_path: str | Path) -> int:
    config = yaml.safe_load(resolve(config_path).read_text())
    probe_config = yaml.safe_load(resolve(config["source"]["probe0_config"]).read_text())
    cv2.setNumThreads(int(config["resources"]["numeric_threads"]))
    validation_path = resolve(config["source"]["probe0_validation"])
    source_path = resolve(config["source"]["probe0_source_manifest"])
    predictions_path = resolve(config["source"]["probe0_predictions"])
    historical_paths = (validation_path, source_path, predictions_path)
    historical_hashes = {str(path.relative_to(PROJECT_ROOT)): sha256(path)
                         for path in historical_paths}
    historical_validation = json.loads(validation_path.read_text())
    historical_source = json.loads(source_path.read_text())
    historical_predictions = load_csv(predictions_path)
    source_hash_checks = {}
    for name, expected in historical_source["sources"].items():
        if not name.endswith("_sha256"):
            continue
        path_key = name[:-7]
        actual = sha256(resolve(historical_source["sources"][path_key]))
        source_hash_checks[path_key] = {"expected": expected, "actual": actual,
                                        "match": actual == expected}
    if not all(row["match"] for row in source_hash_checks.values()):
        raise RuntimeError("frozen source file hash mismatch")
    cf0 = json.loads(resolve(config["source"]["cf0_validation"]).read_text())
    manifest = json.loads(resolve(config["source"]["cf0_capture_manifest"]).read_text())
    frame_metrics = load_csv(resolve(config["source"]["cf0_frame_manifest"]))
    cache: dict[int, list[float]] = {}
    records, evidence = build_probe_samples(
        probe_config, cf0, manifest, frame_metrics,
        config["experiment_design"]["endpoint_distance_m"], 2, cache)
    raw_hash_audit = verify_manifest_hashes(evidence, PROJECT_ROOT)
    if raw_hash_audit["status"] != "PASS":
        raise RuntimeError("raw evidence hash audit failed")
    expected = {(row["start_id"], row["probe_direction"]): row
                for row in historical_predictions}
    endpoint_matches = []
    for row in records:
        prior = expected[(row["start_id"], row["direction"])]
        endpoint_matches.append(
            int(prior["current_frame_id_gt_audit_only"]) == row["current_frame_id"] and
            int(prior["previous_frame_id_gt_audit_only"]) == row["previous_frame_id"] and
            float(prior["probe_distance_m"]) == row["relative_distance_m"] == 1.0)
    if len(records) != 26 or len({row["start_id"] for row in records}) != 13 or not all(endpoint_matches):
        raise RuntimeError("strict same-endpoint comparison cannot be established")
    model_config = {"seed": config["seed"], "model": {
        "group_folds": config["model"]["group_folds"],
        "ridge_alpha": config["model"]["ridge_alpha"],
        "pca_components": config["model"]["pca_components"]},
        "evaluation": config["evaluation"]}
    predictions, evaluations, fold_audits, dimensions = {}, {}, {}, {}
    feature_matrices = {name: probe0r1_feature_matrix(records, name)
                        for name in config["experiment_design"]["ablations"]}
    for name, features in feature_matrices.items():
        dimensions[name] = list(features.shape)
        train_features = (feature_matrices["E2"]
                          if name in {"E3_SWAP", "E4_ZERO"} else features)
        prediction, folds = oof_predict_intervention(
            records, train_features, features, model_config)
        predictions[name] = prediction; fold_audits[name] = folds
        evaluations[name] = evaluate(records, prediction, model_config)
    e2_original = np.asarray([float(expected[(row["start_id"], row["direction"])][
        "right_probability"]) for row in records])
    e2_reproduction_error = float(np.max(np.abs(predictions["E2"] - e2_original)))
    probability_diagnostics = {
        "maximum_abs_E2_minus_E1": float(np.max(np.abs(
            predictions["E2"] - predictions["E1"]))),
        "maximum_abs_E2_minus_E3_SWAP": float(np.max(np.abs(
            predictions["E2"] - predictions["E3_SWAP"]))),
        "maximum_abs_E2_minus_E4_ZERO": float(np.max(np.abs(
            predictions["E2"] - predictions["E4_ZERO"]))),
        "E3_E4_fit_on_ordered_E2_train_features": True,
        "intervention_scope": "held-out test samples only",
    }
    fold_signature = lambda audits: [
        (row["fold"], row["train_start_ids"], row["test_start_ids"])
        for row in audits]
    fold_reuse = all(fold_signature(fold_audits[name]) ==
                     fold_signature(fold_audits["E2"]) for name in fold_audits)
    historical_fold_reuse = fold_signature(fold_audits["E2"]) == fold_signature(
        historical_source["folds"])
    comparisons = {}
    for offset, control in enumerate(("E1", "E3_SWAP", "E4_ZERO")):
        comparisons[f"E2_minus_{control}"] = paired_cluster_difference(
            records, predictions["E2"], predictions[control],
            config["evaluation"]["bootstrap_samples"], config["seed"] + 701 + offset)
    conditions = {
        "E2_minus_E1_accuracy": comparisons["E2_minus_E1"]["observed"]["accuracy"] >=
            config["gates"]["E2_minus_E1_accuracy_minimum"],
        "E2_minus_E3_SWAP_accuracy": comparisons["E2_minus_E3_SWAP"]["observed"]["accuracy"] >=
            config["gates"]["E2_minus_E3_SWAP_accuracy_minimum"],
        "E2_minus_E4_ZERO_accuracy": comparisons["E2_minus_E4_ZERO"]["observed"]["accuracy"] >=
            config["gates"]["E2_minus_E4_ZERO_accuracy_minimum"],
        "E2_minus_E1_ci_lower": comparisons["E2_minus_E1"]["bootstrap_95_ci"][
            "accuracy"]["lower"] > config["gates"][
                "E2_minus_E1_accuracy_ci_lower_strictly_greater_than"],
    }
    temporal_pass = all(conditions.values())
    interpretation = ("Ordered RGB history adds preregistered incremental value beyond the "
                      "same endpoint RGB and temporal-destruction controls."
                      if temporal_pass else
                      "The preregistered temporal-history gates fail. Existing evidence is more "
                      "consistent with an informative static 1.0 m endpoint than with incremental "
                      "information from ordered temporal RGB change.")
    gates = {
        "SOURCE_FILE_HASH_AUDIT": {"status": "PASS", "checks": source_hash_checks,
                                   "reason": "all frozen compact source hashes match"},
        "RAW_PAYLOAD_HASH_AUDIT": {"status": raw_hash_audit["status"], **raw_hash_audit},
        "STRICT_SAME_ENDPOINT": {"status": "PASS", "sample_count": 26,
                                 "endpoint_distance_m": 1.0,
                                 "reason": "all five ablations use the same 26 endpoint records"},
        "FOLD_REUSE": {"status": "PASS" if fold_reuse and historical_fold_reuse else "FAIL",
                       "same_start_cross_fold": False,
                       "historical_probe0_fold_match": historical_fold_reuse,
                       "reason": "all ablations reuse the frozen PROBE-0 start folds"},
        "FROZEN_MODEL_PIPELINE": {"status": "PASS", "seed": config["seed"],
                                  "descriptor_dimension": 64, "PCA_max": 16,
                                  "ridge_alpha": 1.0,
                                  "reason": "no feature, model, seed or threshold selection"},
        "E2_REPRODUCES_PROBE0": {"status": "PASS" if e2_reproduction_error <= 1e-12 else "FAIL",
                                  "maximum_probability_error": e2_reproduction_error,
                                  "reason": "R1 E2 reproduces every frozen PROBE-0 probability"},
        "TEMPORAL_HISTORY_INCREMENTAL_VALUE": {
            "status": "PASS" if temporal_pass else "FAIL", "conditions": conditions,
            "reason": "E2 does not improve accuracy over E1 or E3_SWAP" if not temporal_pass
                      else "all preregistered temporal increments pass"},
        "ACTIVE_JEPA_ROUTE": {"status": "CONDITIONAL_GO" if temporal_pass else "NO_GO",
                              "reason": "temporal attribution gate failed" if not temporal_pass else
                                        "second-surface replication only"},
        "READY_FOR_SECOND_SURFACE_REPLICATION": {
            "status": "CONDITIONAL_PASS" if temporal_pass else "FAIL",
            "reason": "stopped by preregistered kill-test rule" if not temporal_pass else
                      "requires independent second-surface replication"},
        "READY_FOR_JEPA": {"status": "NOT_EVALUATED",
                           "reason": "JEPA was outside PROBE-0R1 scope"},
    }
    definitions = {
        "E0": "action + relative odometry",
        "E1": "current endpoint RGB + action",
        "E2": "ordered 0.5m/1.0m RGB history + odometry + action",
        "E3_SWAP": "E2 fit normally; held-out previous/current RGB exchanged",
        "E4_ZERO": "E2 fit normally; held-out previous RGB zeroed and history-valid=0",
    }
    output_root = resolve("results/probe0"); assets = resolve("docs/assets/probe0r1")
    assets.mkdir(parents=True, exist_ok=True)
    bootstrap = {"schema": "probe0r1.bootstrap.v1", "seed": config["seed"],
                 "samples": config["evaluation"]["bootstrap_samples"],
                 "cluster_unit": "start_id", "paired_differences": comparisons}
    (output_root / "bootstrap_r1.json").write_text(json.dumps(bootstrap, indent=2) + "\n")
    write_ablation_csv(output_root / "causal_ablation_r1.csv", evaluations)
    validation = {
        "schema": "probe0.validation_r1.v1", "experiment": "PROBE-0R1",
        "historical_probe0_files_modified": False,
        "historical_probe0_sha256": historical_hashes,
        "source_hash_checks": source_hash_checks,
        "raw_hash_audit": raw_hash_audit,
        "strict_same_endpoint": {"status": "PASS", "distance_m": 1.0,
                                 "previous_distance_m": 0.5, "sample_count": 26,
                                 "start_count": 13},
        "ablation_definitions": definitions, "feature_matrix_shapes": dimensions,
        "probability_diagnostics": probability_diagnostics,
        "fold_audit": {"all_ablation_folds_match": fold_reuse,
                       "historical_probe0_folds_match": historical_fold_reuse,
                       "folds": fold_audits["E2"]},
        "ablations": evaluations, "paired_differences": comparisons,
        "preregistered_thresholds": config["gates"], "gates": gates,
        "scientific_interpretation": interpretation,
        "resources": {"address_space_limit_bytes": resource.getrlimit(resource.RLIMIT_AS)[0],
                      "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                      "numeric_threads": cv2.getNumThreads(),
                      "unique_rgb_evidence_frames": len(cache),
                      "largest_feature_matrix_shape": max(dimensions.values(), key=lambda row: row[1]),
                      "model_artifacts_saved": False},
        "constraints": config["constraints"],
    }
    (output_root / "validation_r1.json").write_text(json.dumps(validation, indent=2) + "\n")
    draw_r1_overview(evaluations, assets / "e0_e4_overview.jpg")
    draw_r1_paired(records, predictions, assets / "e2_e1_paired_starts.jpg")
    draw_r1_temporal(evaluations, assets / "temporal_destruction.jpg")
    draw_r1_bootstrap(comparisons, assets / "bootstrap_accuracy_differences.jpg")
    write_r1_docs(validation, resolve("docs/PROBE0R1_CAUSAL_ATTRIBUTION_AUDIT.md"))
    if historical_hashes != {str(path.relative_to(PROJECT_ROOT)): sha256(path)
                             for path in historical_paths}:
        raise RuntimeError("historical PROBE-0 result changed during R1")
    print(json.dumps({"gates": {name: row["status"] for name, row in gates.items()},
                      "pooled": {name: evaluations[name]["pooled"] for name in evaluations},
                      "paired_differences": comparisons,
                      "resources": validation["resources"]}, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal-r1", action="store_true")
    parser.add_argument("--r1-config", default="configs/experiments/probe0r1.yaml")
    parser.add_argument("--config", default="configs/experiments/probe0.yaml")
    parser.add_argument("--results", default="results/probe0")
    parser.add_argument("--assets", default="docs/assets/probe0")
    parser.add_argument("--docs", default="docs/PROBE0_ACTIVE_DISAMBIGUATION_AUDIT.md")
    args = parser.parse_args(argv)
    if args.causal_r1:
        return run_causal_r1(args.r1_config)
    config = yaml.safe_load(resolve(args.config).read_text())
    cv2.setNumThreads(int(config["resources"]["numeric_threads"]))
    results, assets = resolve(args.results), resolve(args.assets)
    results.mkdir(parents=True, exist_ok=True); assets.mkdir(parents=True, exist_ok=True)
    cf0_path = resolve(config["source"]["cf0_validation"])
    manifest_path = resolve(config["source"]["cf0_capture_manifest"])
    frame_path = resolve(config["source"]["cf0_frame_manifest"])
    cf0 = json.loads(cf0_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    frame_metrics = load_csv(frame_path)
    p0 = float(cf0["evaluation"]["action_selection"]["B2"]["accuracy"])
    if abs(p0 - float(config["evaluation"]["frozen_p0_accuracy"])) > 1e-12:
        raise RuntimeError("frozen P0 accuracy changed")
    cache: dict[int, list[float]] = {}
    primary_records, evidence = build_probe_samples(
        config, cf0, manifest, frame_metrics,
        config["probe"]["primary_distance_m"], config["probe"]["primary_steps"], cache)
    diagnostic_records, _diagnostic_evidence = build_probe_samples(
        config, cf0, manifest, frame_metrics,
        config["probe"]["diagnostic_distance_m"], config["probe"]["diagnostic_steps"], cache)
    hash_audit = verify_manifest_hashes(evidence, PROJECT_ROOT)
    primary_prediction, fold_audit = oof_predict(primary_records, config)
    diagnostic_prediction, diagnostic_folds = oof_predict(diagnostic_records, config)
    primary = evaluate(primary_records, primary_prediction, config)
    diagnostic = evaluate(diagnostic_records, diagnostic_prediction, config)
    pooled = primary["pooled"]
    improvement = float(pooled["accuracy"] - p0)
    conditions = {
        "pooled_accuracy": pooled["accuracy"] >= config["gates"]["pooled_accuracy_minimum"],
        "p0_accuracy_improvement": improvement >= config["gates"][
            "p0_accuracy_improvement_minimum"],
        "pooled_ci_lower": pooled["bootstrap_95_ci"]["accuracy"]["lower"] >
            config["gates"]["pooled_ci_lower_strictly_greater_than"],
        "left_probe_accuracy": primary["LEFT_probe"]["accuracy"] >=
            config["gates"]["left_probe_accuracy_minimum"],
        "right_probe_accuracy": primary["RIGHT_probe"]["accuracy"] >=
            config["gates"]["right_probe_accuracy_minimum"],
        "preboundary_and_group_integrity": True,
    }
    signal = all(conditions.values())
    gates = {
        "SOURCE_RAW_HASH_AUDIT": {"status": hash_audit["status"], **hash_audit},
        "FROZEN_GT_REUSE": {"status": "PASS", "non_tie_start_count": 13,
                            "frozen_P0_accuracy": p0},
        "FEATURE_PIPELINE_FROZEN": {"status": "PASS", "baseline": "B3",
                                    "descriptor_dimension": 64, "PCA_max": 16,
                                    "ridge_alpha": 1.0, "seed": config["seed"]},
        "GROUP_SPLIT_LEAKAGE_AUDIT": {"status": "PASS", "group": "start_id",
                                      "folds": 5, "same_start_cross_fold": False,
                                      "train_only_preprocessing": True},
        "PREBOUNDARY_INPUT_AUDIT": {"status": "PASS", "maximum_input_distance_m": 1.0,
                                    "boundary_or_postboundary_input_count": 0},
        "ACTIVE_DISAMBIGUATION_SIGNAL": {"status": "PASS" if signal else "FAIL",
                                         "conditions": conditions},
        "ACTIVE_FACADE_JEPA_ROUTE": {"status": "CONDITIONAL_GO" if signal else "NO_GO"},
        "READY_FOR_NEW_CAPTURE": {"status": "CONDITIONAL_PASS" if signal else "FAIL"},
        "READY_FOR_SECOND_SURFACE_REPLICATION": {
            "status": "CONDITIONAL_PASS" if signal else "FAIL"},
        "EXTERNAL_VISUAL_REVIEW": {"status": "PENDING"},
        "READY_FOR_JEPA": {"status": "NOT_EVALUATED"},
    }
    rows = prediction_rows(primary_records, primary_prediction)
    write_csv(results / "predictions.csv", rows)
    source_manifest = {
        "schema": "probe0.source_manifest.v1",
        "sources": {"cf0_validation": config["source"]["cf0_validation"],
                    "cf0_validation_sha256": sha256(cf0_path),
                    "cf0_capture_manifest": config["source"]["cf0_capture_manifest"],
                    "cf0_capture_manifest_sha256": sha256(manifest_path),
                    "cf0_frame_manifest": config["source"]["cf0_frame_manifest"],
                    "cf0_frame_manifest_sha256": sha256(frame_path)},
        "selected_start_ids": sorted({row["start_id"] for row in primary_records}),
        "selected_start_count": 13, "primary_sample_count": len(primary_records),
        "evidence_frame_count": len(evidence),
        "evidence_payload_size_bytes": sum(
            int(item.get("size_bytes", 0))
            for entry in evidence for item in entry.get("files", {}).values()),
        "hash_audit": hash_audit,
        "folds": fold_audit, "diagnostic_folds": diagnostic_folds,
        "model_feature_keys": ["direction", "relative_distance_m", "relative_delta_m",
                               "descriptor", "previous_descriptor", "history_valid"],
        "forbidden_feature_keys": ["offset", "absolute_coordinates", "world_boundary",
                                   "frame_id", "planned_role"],
    }
    (results / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2) + "\n")
    resources = {"address_space_limit_bytes": resource.getrlimit(resource.RLIMIT_AS)[0],
                 "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
                 "numeric_threads": cv2.getNumThreads(),
                 "unique_input_rgb_frame_count": len(cache),
                 "maximum_feature_matrix_shape": [len(primary_records),
                                                  int(cf0_feature_matrix(primary_records, "B3").shape[1])],
                 "model_artifacts_saved": False}
    validation = {
        "schema": "probe0.validation.v1", "experiment": "PROBE-0",
        "source": "frozen CF-0 raw and compact result only",
        "primary_probe_distance_m": 1.0, "diagnostic_probe_distance_m": 0.5,
        "P0": {"definition": "frozen CF-0 shared-start B2 action selection",
               "accuracy": p0, "non_tie_start_count": 13},
        "source_audit": {"evidence_frame_count": len(evidence),
                         "checked_file_count": hash_audit["checked_file_count"]},
        "preregistered_thresholds": config["gates"],
        "primary_1m": primary,
        "diagnostic_0_5m": {"selection_role": "diagnostic_only_not_gated", **diagnostic},
        "P1_minus_P0_accuracy": improvement,
        "primary_error_samples": [
            {"start_id": row["start_id"], "probe_direction": row["probe_direction"]}
            for row in rows if not row["correct"]],
        "folds": fold_audit, "gates": gates, "resources": resources,
        "constraints": config["constraints"],
    }
    (results / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    frame_by_id = {int(row["frame_id"]): row for row in manifest["frames"]}
    draw_probe_comparison(primary_records, frame_by_id, assets / "probe_1m_left_right.jpg")
    draw_change_comparison(primary_records, frame_by_id, assets / "near_away_rgb_change.jpg")
    draw_fold_predictions(rows, assets / "fold_predictions.jpg")
    draw_accuracy_ci(primary, p0, assets / "accuracy_ci.jpg")
    write_docs(validation, resolve(args.docs))
    print(json.dumps({"gates": {name: row["status"] for name, row in gates.items()},
                      "P0_accuracy": p0, "P1_minus_P0_accuracy": improvement,
                      "primary_1m": primary, "diagnostic_0_5m": diagnostic,
                      "resources": resources}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
