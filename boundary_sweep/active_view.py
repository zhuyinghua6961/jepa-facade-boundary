"""AVS-0 cross-surface active-view-selection evaluation primitives."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from .observability import cf0_feature_matrix, train_only_pca_ridge


def surface_leave_one_out(records: Sequence[Mapping], pca_components: int = 16,
                          ridge_alpha: float = 1.0) -> tuple[np.ndarray, list[dict]]:
    """Evaluate frozen E1 features with train-only processing per held-out surface."""
    rows = []
    for source in records:
        row = dict(source)
        descriptor = np.asarray(row["descriptor"], dtype=float)
        row.setdefault("previous_descriptor", np.zeros_like(descriptor))
        row.setdefault("history_valid", False)
        rows.append(row)
    surfaces = sorted({str(row["surface_id"]) for row in rows})
    if len(surfaces) < 2:
        raise ValueError("surface leave-one-out requires at least two surfaces")
    features = cf0_feature_matrix(rows, "B2")
    target = np.asarray([int(row["target"]) for row in rows], dtype=float)
    prediction = np.full(len(rows), np.nan, dtype=float)
    audits = []
    for held_out in surfaces:
        test = np.asarray([i for i, row in enumerate(rows)
                           if str(row["surface_id"]) == held_out], dtype=int)
        train = np.asarray([i for i, row in enumerate(rows)
                            if str(row["surface_id"]) != held_out], dtype=int)
        if not len(train) or not len(test):
            raise ValueError("empty surface leave-one-out partition")
        raw, preprocessing = train_only_pca_ridge(
            features[train], target[train], features[test], alpha=ridge_alpha,
            max_components=pca_components)
        prediction[test] = np.clip(raw, 0.0, 1.0)
        audits.append({
            "held_out_surface": held_out,
            "train_surfaces": [name for name in surfaces if name != held_out],
            "train_sample_count": int(len(train)),
            "test_sample_count": int(len(test)),
            "preprocessing": preprocessing,
        })
    if not np.isfinite(prediction).all():
        raise RuntimeError("incomplete surface leave-one-out prediction")
    return prediction, audits


def paired_policy_rows(records: Sequence[Mapping], prediction: Sequence[float]) -> list[dict]:
    """Collapse bilateral endpoint predictions to one policy comparison per start."""
    grouped: dict[tuple[str, str], dict[str, tuple[Mapping, float]]] = defaultdict(dict)
    for row, probability in zip(records, prediction):
        key = (str(row["surface_id"]), str(row["start_id"]))
        direction = str(row["direction"])
        if direction not in {"LEFT", "RIGHT"} or direction in grouped[key]:
            raise ValueError("each start must contain one LEFT and one RIGHT endpoint")
        grouped[key][direction] = (row, float(probability))
    output = []
    for (surface_id, start_id), pair in sorted(grouped.items()):
        if set(pair) != {"LEFT", "RIGHT"}:
            raise ValueError("incomplete counterfactual endpoint pair")
        left_row, left_probability = pair["LEFT"]
        right_row, right_probability = pair["RIGHT"]
        if int(left_row["target"]) != int(right_row["target"]):
            raise ValueError("paired actions disagree on ground truth")
        target = int(left_row["target"])
        left_correct = int((left_probability >= 0.5) == bool(target))
        right_correct = int((right_probability >= 0.5) == bool(target))
        regret = float(left_row["wrong_action_regret_m"])
        output.append({
            "surface_id": surface_id,
            "start_id": start_id,
            "near_direction_gt": "RIGHT" if target else "LEFT",
            "left_right_probability": left_probability,
            "right_right_probability": right_probability,
            "fixed_left_correct": left_correct,
            "fixed_right_correct": right_correct,
            "random_expected_correct": (left_correct + right_correct) / 2.0,
            "oracle_correct": max(left_correct, right_correct),
            "unique_best_action": ("LEFT" if left_correct and not right_correct else
                                   "RIGHT" if right_correct and not left_correct else "TIE"),
            "wrong_action_regret_m": regret,
            "fixed_left_regret_m": 0.0 if left_correct else regret,
            "fixed_right_regret_m": 0.0 if right_correct else regret,
            "oracle_regret_m": 0.0 if max(left_correct, right_correct) else regret,
        })
    return output


def policy_summary(rows: Sequence[Mapping]) -> dict:
    """Summarize fixed, random and oracle policy accuracy and preference headroom."""
    values = list(rows)
    if not values:
        raise ValueError("policy summary requires starts")
    accuracy = {
        "FIXED_LEFT": float(np.mean([row["fixed_left_correct"] for row in values])),
        "FIXED_RIGHT": float(np.mean([row["fixed_right_correct"] for row in values])),
        "RANDOM": float(np.mean([row["random_expected_correct"] for row in values])),
        "ORACLE_PER_START": float(np.mean([row["oracle_correct"] for row in values])),
    }
    best_name = max(("FIXED_LEFT", "FIXED_RIGHT"), key=lambda name: accuracy[name])
    left_unique = sum(row["unique_best_action"] == "LEFT" for row in values)
    right_unique = sum(row["unique_best_action"] == "RIGHT" for row in values)
    return {
        "start_count": len(values),
        "accuracy": accuracy,
        "best_fixed_policy": best_name,
        "best_fixed_accuracy": accuracy[best_name],
        "oracle_minus_best_fixed_accuracy": accuracy["ORACLE_PER_START"] - accuracy[best_name],
        "unique_best_action_counts": {"LEFT": left_unique, "RIGHT": right_unique,
                                      "TIE": len(values) - left_unique - right_unique},
        "unique_best_action_fractions": {"LEFT": left_unique / len(values),
                                         "RIGHT": right_unique / len(values)},
    }


def stratified_start_bootstrap(rows: Sequence[Mapping], samples: int,
                               seed: int) -> dict:
    """Bootstrap starts within surfaces and recompute oracle minus best fixed."""
    values = list(rows)
    by_surface: dict[str, list[Mapping]] = defaultdict(list)
    for row in values:
        by_surface[str(row["surface_id"])].append(row)
    rng = np.random.default_rng(int(seed))
    estimates = []
    for _ in range(int(samples)):
        selected = []
        for surface in sorted(by_surface):
            group = by_surface[surface]
            indices = rng.integers(0, len(group), size=len(group))
            selected.extend(group[index] for index in indices)
        estimates.append(policy_summary(selected)["oracle_minus_best_fixed_accuracy"])
    return {
        "lower": float(np.percentile(estimates, 2.5)),
        "upper": float(np.percentile(estimates, 97.5)),
        "bootstrap_samples": len(estimates),
        "cluster_unit": "start_id",
        "surface_stratified": True,
    }


def evaluate_preregistered_gates(summary: Mapping, starts_per_surface: Mapping[str, int],
                                 bootstrap: Mapping, thresholds: Mapping) -> dict:
    """Evaluate AVS-0 preregistered gates without outcome overrides."""
    conditions = {
        "valid_surface_count": len(starts_per_surface) >= int(thresholds["minimum_surface_count"]),
        "valid_starts_per_surface": bool(starts_per_surface) and all(
            count >= int(thresholds["minimum_starts_per_surface"])
            for count in starts_per_surface.values()),
        "oracle_accuracy": float(summary["accuracy"]["ORACLE_PER_START"]) >=
            float(thresholds["oracle_accuracy_minimum"]),
        "oracle_headroom": float(summary["oracle_minus_best_fixed_accuracy"]) >=
            float(thresholds["oracle_minus_best_fixed_minimum"]),
        "headroom_ci_lower": float(bootstrap["lower"]) >
            float(thresholds["headroom_ci_lower_strictly_greater_than"]),
        "left_unique_fraction": float(summary["unique_best_action_fractions"]["LEFT"]) >=
            float(thresholds["unique_action_fraction_minimum"]),
        "right_unique_fraction": float(summary["unique_best_action_fractions"]["RIGHT"]) >=
            float(thresholds["unique_action_fraction_minimum"]),
    }
    passed = all(conditions.values())
    return {"status": "PASS" if passed else "FAIL", "conditions": conditions,
            "thresholds": dict(thresholds)}
