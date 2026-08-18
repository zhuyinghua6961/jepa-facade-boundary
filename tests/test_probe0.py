import importlib.util
import csv
import json
from pathlib import Path

import numpy as np
import yaml

from boundary_sweep.observability import cf0_feature_matrix, grouped_kfold


def _runner_module():
    path = Path("scripts/run_probe0.py")
    spec = importlib.util.spec_from_file_location("run_probe0", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _record(start_id="start_00", direction="LEFT", target=0):
    return {
        "start_id": start_id,
        "direction": direction,
        "relative_distance_m": 1.0,
        "relative_delta_m": 0.5,
        "descriptor": np.arange(64, dtype=float),
        "previous_descriptor": np.arange(64, dtype=float) + 100.0,
        "history_valid": True,
        "target": target,
        "near_direction": "RIGHT" if target else "LEFT",
        "wrong_action_regret_m": 1.5,
        # Deliberately extreme values: none may enter the feature matrix.
        "offset": 1e9,
        "absolute_coordinates": [1e9, 1e9, 1e9],
        "world_boundary": [1e9] * 6,
        "frame_id": 999999999,
        "planned_role": "FORBIDDEN",
    }


def test_probe0_uses_exact_frozen_b3_layout_and_ignores_gt_audit_fields():
    record = _record()
    features = cf0_feature_matrix([record], "B3")
    assert features.shape == (1, 133)
    assert np.array_equal(features[0, :4], [-1.0, 1.0, 1.0, 0.5])
    assert np.array_equal(features[0, 4:68], record["descriptor"])
    assert np.array_equal(features[0, 68:132], record["previous_descriptor"])
    assert features[0, 132] == 1.0
    assert np.max(np.abs(features)) < 1e6


def test_probe0_keeps_bilateral_samples_from_each_start_in_one_fold():
    records = [_record(f"start_{index:02d}", direction, index % 2)
               for index in range(13) for direction in ("LEFT", "RIGHT")]
    folds = grouped_kfold([row["start_id"] for row in records], 5, 20260817)
    seen = set()
    for fold in folds:
        train_groups = set(fold["train_groups"])
        test_groups = set(fold["test_groups"])
        assert train_groups.isdisjoint(test_groups)
        for start_id in test_groups:
            assert sum(records[index]["start_id"] == start_id
                       for index in fold["test_indices"]) == 2
        seen.update(test_groups)
    assert len(seen) == 13


def test_probe0_config_preregisters_one_meter_and_forbids_distance_selection():
    config = yaml.safe_load(Path("configs/experiments/probe0.yaml").read_text())
    assert config["seed"] == 20260817
    assert config["probe"] == {
        "primary_distance_m": 1.0,
        "primary_steps": 2,
        "diagnostic_distance_m": 0.5,
        "diagnostic_steps": 1,
        "step_m": 0.5,
        "forbidden_distance_m": 2.0,
    }
    assert config["model"] == {
        "baseline": "B3", "descriptor_length": 64, "pca_components": 16,
        "ridge_alpha": 1.0, "group_folds": 5,
    }
    assert config["constraints"]["distance_selection_by_performance"] is False


def test_probe0_frozen_truth_excludes_ties_and_reuses_cf0_distances():
    runner = _runner_module()
    cf0 = {
        "capture": {"maximum_distance_m": 4.0, "step_m": 0.5},
        "branch_summaries": [
            {"start_id": "near_left", "direction": "LEFT",
             "first_model_visible_distance_m": 2.5},
            {"start_id": "near_left", "direction": "RIGHT",
             "first_model_visible_distance_m": None},
            {"start_id": "tie", "direction": "LEFT",
             "first_model_visible_distance_m": None},
            {"start_id": "tie", "direction": "RIGHT",
             "first_model_visible_distance_m": None},
        ],
    }
    truth = runner.eligible_truth(cf0)
    assert truth == [{
        "start_id": "near_left", "near_direction": "LEFT", "target": 0,
        "left_cost_m": 2.5, "right_cost_m": 4.5,
        "wrong_action_regret_m": 2.0,
    }]


def test_probe0_cluster_bootstrap_resamples_whole_start_groups():
    runner = _runner_module()
    records = [_record("a", "LEFT", 0), _record("a", "RIGHT", 0),
               _record("b", "LEFT", 1), _record("b", "RIGHT", 1)]
    prediction = np.asarray([0.1, 0.2, 0.8, 0.9])
    result = runner.cluster_bootstrap(records, prediction, samples=25, seed=7)
    assert all(row["cluster_unit"] == "start_id" for row in result.values())
    assert result["accuracy"]["bootstrap_samples"] == 25


def test_probe0_published_inputs_are_exactly_preboundary_steps_one_and_two():
    with Path("results/probe0/predictions.csv").open(newline="") as handle:
        predictions = list(csv.DictReader(handle))
    with Path("results/cf0/frame_manifest.csv").open(newline="") as handle:
        frame_lookup = {int(row["frame_id"]): row for row in csv.DictReader(handle)}
    assert len(predictions) == 26
    for row in predictions:
        previous = frame_lookup[int(row["previous_frame_id_gt_audit_only"])]
        current = frame_lookup[int(row["current_frame_id_gt_audit_only"])]
        assert (int(previous["step_index"]), int(current["step_index"])) == (1, 2)
        assert float(current["relative_distance_m"]) == 1.0
        for evidence in (previous, current):
            assert evidence["first_physical_termination"] == "False"
            assert evidence["model_visible_termination"] == "False"


def test_probe0_compact_result_and_four_public_images_are_complete():
    validation = json.loads(Path("results/probe0/validation.json").read_text())
    source = json.loads(Path("results/probe0/source_manifest.json").read_text())
    assert validation["primary_1m"]["pooled"]["sample_count"] == 26
    assert validation["gates"]["READY_FOR_JEPA"]["status"] == "NOT_EVALUATED"
    assert source["evidence_frame_count"] == 65
    assert source["hash_audit"]["checked_file_count"] == 390
    expected = {"probe_1m_left_right.jpg", "near_away_rgb_change.jpg",
                "fold_predictions.jpg", "accuracy_ci.jpg"}
    assets = {path.name for path in Path("docs/assets/probe0").glob("*.jpg")}
    assert assets == expected
    assert all(0 < (Path("docs/assets/probe0") / name).stat().st_size < 2_000_000
               for name in expected)
