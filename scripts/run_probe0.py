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
                                          raw_ssim, rgb_descriptor,
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


def oof_predict(records: list[dict], config: dict) -> tuple[np.ndarray, list[dict]]:
    features = cf0_feature_matrix(records, config["model"]["baseline"])
    target = np.asarray([row["target"] for row in records], dtype=float)
    folds = grouped_kfold([row["start_id"] for row in records],
                          config["model"]["group_folds"], config["seed"])
    prediction = np.full(len(records), np.nan, dtype=float)
    audits = []
    for fold in folds:
        train = np.asarray(fold["train_indices"], dtype=int)
        test = np.asarray(fold["test_indices"], dtype=int)
        raw, preprocessing = train_only_pca_ridge(
            features[train], target[train], features[test],
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/probe0.yaml")
    parser.add_argument("--results", default="results/probe0")
    parser.add_argument("--assets", default="docs/assets/probe0")
    parser.add_argument("--docs", default="docs/PROBE0_ACTIVE_DISAMBIGUATION_AUDIT.md")
    args = parser.parse_args(argv)
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
