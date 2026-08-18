import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml

from boundary_sweep.active_view import (evaluate_preregistered_gates,
                                         paired_policy_rows, policy_summary,
                                         stratified_start_bootstrap,
                                         surface_leave_one_out)


def records():
    rows = []
    for surface_index in range(3):
        for start_index in range(8):
            target = (surface_index + start_index) % 2
            for direction in ("LEFT", "RIGHT"):
                descriptor = np.zeros(64)
                descriptor[target] = 1.0
                descriptor[2 + surface_index] = 0.1
                rows.append({
                    "surface_id": f"surface_{surface_index}",
                    "start_id": f"start_{start_index}", "direction": direction,
                    "relative_distance_m": 1.0, "relative_delta_m": 0.5,
                    "descriptor": descriptor, "target": target,
                    "wrong_action_regret_m": 1.0,
                })
    return rows


def test_surface_leave_one_out_is_complete_and_train_only():
    prediction, audits = surface_leave_one_out(records(), 16, 1.0)
    assert prediction.shape == (48,)
    assert np.isfinite(prediction).all()
    assert {row["held_out_surface"] for row in audits} == {
        "surface_0", "surface_1", "surface_2"}
    assert all(row["preprocessing"]["preprocessing_fit_scope"] ==
               "training_fold_only" for row in audits)


def test_policy_pairing_and_unique_action_counts():
    rows = records()[:4]
    prediction = [0.1, 0.9, 0.1, 0.9]
    paired = paired_policy_rows(rows, prediction)
    assert len(paired) == 2
    assert {row["unique_best_action"] for row in paired} == {"LEFT", "RIGHT"}
    summary = policy_summary(paired)
    assert summary["accuracy"]["ORACLE_PER_START"] == 1.0
    assert summary["unique_best_action_fractions"] == {"LEFT": 0.5, "RIGHT": 0.5}


def test_oracle_headroom_bootstrap_resamples_starts_within_surfaces():
    paired = []
    for surface in range(3):
        for start in range(8):
            left = int(start % 2 == 0); right = 1 - left
            paired.append({"surface_id": f"s{surface}", "start_id": f"x{start}",
                           "fixed_left_correct": left, "fixed_right_correct": right,
                           "random_expected_correct": 0.5, "oracle_correct": 1,
                           "unique_best_action": "LEFT" if left else "RIGHT"})
    result = stratified_start_bootstrap(paired, 100, 7)
    assert result["cluster_unit"] == "start_id"
    assert result["surface_stratified"] is True
    assert result["bootstrap_samples"] == 100


def test_preregistered_gate_fails_when_one_action_lacks_headroom():
    summary = {
        "accuracy": {"ORACLE_PER_START": 0.8},
        "oracle_minus_best_fixed_accuracy": 0.2,
        "unique_best_action_fractions": {"LEFT": 0.25, "RIGHT": 0.10},
    }
    thresholds = {"minimum_surface_count": 3, "minimum_starts_per_surface": 8,
                  "oracle_accuracy_minimum": 0.7,
                  "oracle_minus_best_fixed_minimum": 0.15,
                  "headroom_ci_lower_strictly_greater_than": 0.0,
                  "unique_action_fraction_minimum": 0.2}
    gate = evaluate_preregistered_gates(
        summary, {"a": 8, "b": 8, "c": 8}, {"lower": 0.01}, thresholds)
    assert gate["status"] == "FAIL"
    assert gate["conditions"]["right_unique_fraction"] is False


def test_config_freezes_candidate_order_limits_and_e1_model():
    config = yaml.safe_load(Path("configs/experiments/avs0.yaml").read_text())
    assert config["candidate_order"] == [7, 8, 10, 19]
    assert config["capture"]["starts_per_new_surface"] == 8
    assert config["capture"]["maximum_saved_quartets"] == 48
    assert config["evaluation"] == {"descriptor_length": 64, "pca_components": 16,
                                    "ridge_alpha": 1.0, "bootstrap_samples": 1000}
    assert config["resources"]["numeric_threads"] == 1
    assert config["constraints"]["policy_training"] is False
    assert config["constraints"]["jepa_training"] is False


def test_capture_driver_reuses_existing_sensor_and_boundary_modules():
    path = Path("scripts/run_avs0.py")
    source = path.read_text()
    assert "SynchronousRGBDSeg" in source
    assert "_cf0_sample_metric" in source
    assert "config_outcome_override_audit" in source
    spec = importlib.util.spec_from_file_location("run_avs0", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pose = module.candidate_pose(yaml.safe_load(
        Path("configs/experiments/avs0.yaml").read_text())["candidates"][7])
    assert np.allclose(pose[:3, 3], [-11.27066612, 256.42282104, 10.27672386])


def test_current_session_instance_is_resolved_from_center_pixels():
    path = Path("scripts/run_avs0.py")
    spec = importlib.util.spec_from_file_location("run_avs0_target", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    target_id = 39241
    bgra = np.zeros((4, 6, 4), dtype=np.uint8)
    bgra[..., 0] = target_id >> 8
    bgra[..., 1] = target_id & 255
    bgra[..., 2] = 3
    class ImageData:
        height, width, raw_data = 4, 6, bgra.tobytes()
    result = module.resolve_center_target_instance({"data": {"instance": ImageData()}})
    assert result["status"] == "PASS"
    assert result["target_instance_id"] == target_id
    assert result["method"] == "optical_center_building_pixel"


def test_published_avs0_result_is_fail_closed_and_assets_exist():
    path = Path("results/avs0/validation.json")
    if not path.exists():
        return
    result = json.loads(path.read_text())
    gates = result["gates"]
    assert result["selected_new_candidates"] == [7, 8]
    assert result["valid_starts_per_surface"] == {
        "candidate_1": 8, "candidate_7": 8, "candidate_8": 8}
    assert gates["PHYSICAL_BOUNDARY_AND_SAFETY"]["status"] == "PASS"
    assert gates["SURFACE_LEAVE_ONE_OUT"]["status"] == "PASS"
    assert gates["ACTIVE_VIEW_SELECTION_HEADROOM"]["status"] == "FAIL"
    assert gates["READY_FOR_POLICY_PILOT"]["status"] == "FAIL"
    assert gates["READY_FOR_JEPA"]["status"] == "NOT_EVALUATED"
    assert result["policy_summary"]["oracle_minus_best_fixed_accuracy"] == 0.0
    assert result["bootstrap_95_ci"]["lower"] == 0.0
    assets = {entry.name for entry in Path("docs/assets/avs0").glob("*.jpg")}
    assert assets == {"candidate_1_probe_contact_sheet.jpg",
                      "candidate_7_probe_contact_sheet.jpg",
                      "candidate_8_probe_contact_sheet.jpg",
                      "fixed_vs_oracle.jpg", "action_preference_distribution.jpg"}
