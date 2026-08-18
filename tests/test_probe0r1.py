import importlib.util
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from boundary_sweep.observability import probe0r1_feature_matrix


def _runner_module():
    path = Path("scripts/run_probe0.py")
    spec = importlib.util.spec_from_file_location("run_probe0_r1", path)
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
        "offset": 1e9,
        "absolute_coordinates": [1e9, 1e9, 1e9],
        "world_boundary": [1e9] * 6,
        "frame_id": 999999999,
        "planned_role": "FORBIDDEN",
    }


def test_probe0r1_ablation_shapes_and_frozen_feature_content():
    row = _record()
    matrices = {name: probe0r1_feature_matrix([row], name)
                for name in ("E0", "E1", "E2", "E3_SWAP", "E4_ZERO")}
    assert {name: value.shape for name, value in matrices.items()} == {
        "E0": (1, 4), "E1": (1, 66), "E2": (1, 133),
        "E3_SWAP": (1, 133), "E4_ZERO": (1, 133),
    }
    assert np.array_equal(matrices["E2"][0, 4:68], row["descriptor"])
    assert np.array_equal(matrices["E2"][0, 68:132], row["previous_descriptor"])
    assert np.array_equal(matrices["E3_SWAP"][0, 4:68], row["previous_descriptor"])
    assert np.array_equal(matrices["E3_SWAP"][0, 68:132], row["descriptor"])
    assert np.array_equal(matrices["E4_ZERO"][0, 4:68], row["descriptor"])
    assert np.count_nonzero(matrices["E4_ZERO"][0, 68:132]) == 0
    assert matrices["E4_ZERO"][0, 132] == 0.0
    assert all(np.max(np.abs(value)) < 1e6 for value in matrices.values())


def test_probe0r1_unknown_ablation_fails_closed():
    try:
        probe0r1_feature_matrix([_record()], "SELECT_BEST")
    except ValueError as error:
        assert "unknown PROBE-0R1 ablation" in str(error)
    else:
        raise AssertionError("unknown ablation did not fail")


def test_probe0r1_config_freezes_endpoint_model_and_gates():
    config = yaml.safe_load(Path("configs/experiments/probe0r1.yaml").read_text())
    assert config["seed"] == 20260817
    design = config["experiment_design"]
    assert {name: design[name] for name in ("endpoint_distance_m", "previous_distance_m",
                                             "start_count", "sample_count", "ablations")} == {
        "endpoint_distance_m": 1.0, "previous_distance_m": 0.5,
        "start_count": 13, "sample_count": 26,
        "ablations": ["E0", "E1", "E2", "E3_SWAP", "E4_ZERO"]}
    assert set(design["intervention_protocol"]) == {"E3_SWAP", "E4_ZERO"}
    assert config["model"] == {"descriptor_length": 64, "pca_components": 16,
                               "ridge_alpha": 1.0, "group_folds": 5}
    assert set(config["gates"].values()) == {0.10, 0.0}
    assert config["constraints"]["model_or_threshold_selection"] is False


def test_probe0r1_paired_bootstrap_uses_start_clusters():
    runner = _runner_module()
    records = [_record("a", "LEFT", 0), _record("a", "RIGHT", 0),
               _record("b", "LEFT", 1), _record("b", "RIGHT", 1)]
    e2 = np.asarray([0.1, 0.2, 0.8, 0.9])
    control = np.asarray([0.7, 0.7, 0.3, 0.3])
    result = runner.paired_cluster_difference(records, e2, control, 25, 9)
    assert result["observed"]["accuracy"] == 1.0
    assert all(row["cluster_unit"] == "start_id"
               for row in result["bootstrap_95_ci"].values())
    assert result["bootstrap_95_ci"]["accuracy"]["bootstrap_samples"] == 25


def test_probe0r1_temporal_destruction_is_held_out_only():
    runner = _runner_module()
    records = [_record(f"start_{index:02d}", "LEFT", index % 2)
               for index in range(10)]
    ordered = np.arange(30, dtype=float).reshape(10, 3)
    intervened = ordered + 1000.0
    calls = []
    original = runner.train_only_pca_ridge

    def fake(train_x, train_y, test_x, alpha, max_components):
        calls.append((train_x.copy(), test_x.copy()))
        return np.zeros(len(test_x)), {"input_dimension": train_x.shape[1]}

    runner.train_only_pca_ridge = fake
    try:
        runner.oof_predict_intervention(
            records, ordered, intervened,
            {"seed": 20260817, "model": {"group_folds": 5,
                                           "ridge_alpha": 1.0,
                                           "pca_components": 16}})
    finally:
        runner.train_only_pca_ridge = original
    assert len(calls) == 5
    assert all(np.max(train) < 1000.0 for train, _test in calls)
    assert all(np.min(test) >= 1000.0 for _train, test in calls)


def test_probe0r1_compact_result_fails_temporal_gate_without_changing_history():
    validation = json.loads(Path("results/probe0/validation_r1.json").read_text())
    assert validation["strict_same_endpoint"]["sample_count"] == 26
    assert validation["gates"]["TEMPORAL_HISTORY_INCREMENTAL_VALUE"]["status"] == "FAIL"
    assert validation["gates"]["ACTIVE_JEPA_ROUTE"]["status"] == "NO_GO"
    assert validation["gates"]["READY_FOR_SECOND_SURFACE_REPLICATION"]["status"] == "FAIL"
    assert validation["gates"]["READY_FOR_JEPA"]["status"] == "NOT_EVALUATED"
    assert validation["probability_diagnostics"][
        "E3_E4_fit_on_ordered_E2_train_features"] is True
    for name, expected in validation["historical_probe0_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected


def test_probe0r1_csv_bootstrap_and_public_assets_are_complete():
    with Path("results/probe0/causal_ablation_r1.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    bootstrap = json.loads(Path("results/probe0/bootstrap_r1.json").read_text())
    assert len(rows) == 15
    assert bootstrap["samples"] == 1000
    assert bootstrap["cluster_unit"] == "start_id"
    expected = {"e0_e4_overview.jpg", "e2_e1_paired_starts.jpg",
                "temporal_destruction.jpg", "bootstrap_accuracy_differences.jpg"}
    assets = {path.name for path in Path("docs/assets/probe0r1").glob("*.jpg")}
    assert assets == expected
    assert all(0 < (Path("docs/assets/probe0r1") / name).stat().st_size < 2_000_000
               for name in expected)
