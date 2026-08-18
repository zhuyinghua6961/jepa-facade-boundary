#!/usr/bin/env python3
"""Validate the compact public GEO-0.5R2 result set and its gate semantics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml

from scripts.validate_geo05r2 import depth_gate, reprojection_stats


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/geo05r2")
    args = parser.parse_args(argv)
    root = Path(args.results)
    required = ["geo05r2_validation.json", "trajectory_transition_audit_r2.json", "geometry_reference_audit.json", "depth_metric_v2.json", "r2_dataset_manifest.json"]
    missing = [name for name in required if not (root / name).exists()]
    validation = json.loads((root / "geo05r2_validation.json").read_text()) if not missing else {}
    depth = json.loads((root / "depth_metric_v2.json").read_text()) if not missing else {}
    reference = json.loads((root / "geometry_reference_audit.json").read_text()) if not missing else {}
    surfaces = sorted((root / "surfaces_v3").glob("*.json")) if (root / "surfaces_v3").exists() else []
    gates = validation.get("gates", {})
    surface_checks = []
    for path in surfaces:
        data = json.loads(path.read_text())
        actual = reprojection_stats(data.get("reprojection_errors_px", {}))
        reported = gates.get("PHYSICAL_BOUNDARY_GROUND_TRUTH", {}).get("surfaces", [])
        match = next((row for row in reported if row.get("surface_id") == data.get("surface_id", path.stem)), {})
        surface_checks.append(actual["sample_count"] > 0 and abs(actual["median_px"] - match.get("median_px", -1)) < 1e-12 and abs(actual["max_px"] - match.get("max_px", -1)) < 1e-12)
    forbidden = []
    for path in Path(".").rglob("*"):
        if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts and path.suffix != ".pyc" and path.stat().st_size < 2_000_000:
            try:
                text = path.read_text(errors="ignore")
            except Exception:
                continue
            if ("/" + "mnt/fast18/") in text or ("/" + "Users/") in text:
                forbidden.append(str(path))
    depth_result = depth_gate(depth, "results/geo05r2/depth_metric_v2.json")
    r1_path = Path("results/mask0/validation_r1.json")
    mask1_path = Path("results/mask1/validation.json")
    mask1_r1_path = Path("results/mask1/validation_r1.json")
    event_r1_path = Path("results/mask1/event_reanalysis_r1.json")
    repeatability_r1_path = Path("results/mask1/world_boundary_repeatability_r1.json")
    obs0_path = Path("results/obs0/validation.json")
    probe_path = Path("results/obs0/probe_results.json")
    obs0r1_path = Path("results/obs0/validation_r1.json")
    obs0r1_probe_path = Path("results/obs0/probe_results_r1.json")
    obs0r1_alias_path = Path("results/obs0/alias_summary_r1.json")
    obs0r1_history_path = Path("results/obs0/history_boundary_audit.json")
    obs0r1_frames_path = Path("results/obs0/frame_targets_r1.csv")
    act0_validation_path = Path("results/act0/validation.json")
    act0_manifest_path = Path("results/act0/scout_manifest.json")
    act0s_path = Path("results/act0/screening_audit_v2.json")
    act0s_matrix_path = Path("results/act0/candidate_gate_matrix_v2.csv")
    act0s_coverage_path = Path("results/act0/scout_coverage_audit.json")
    act0r_validation_path = Path("results/act0r/validation.json")
    act0r_search_path = Path("results/act0r/search_plan_checkpoint.json")
    act0r_frame_manifest_path = Path("results/act0r/frame_manifest.csv")
    act0r_operator_path = Path("results/act0r/operator_review_template.csv")
    cap0_path = Path("results/cap0/validation.json")
    cap0_probe_path = Path("results/cap0/as2_probe.json")
    cap0_root_cause_path = Path("results/cap0/root_cause.json")
    cap0_matrix_path = Path("results/cap0/health_matrix.csv")
    cap0_manifest_path = Path("results/cap0/diagnostic_manifest.json")
    act0r1_path = Path("results/act0r1/validation.json")
    act0r1_offline_path = Path("results/act0r1/offline_boundary_audit.json")
    act0r1_offline_frames_path = Path("results/act0r1/offline_frame_metrics.csv")
    act0r1_offline_hashes_path = Path("results/act0r1/offline_raw_hash_audit.csv")
    act0r2_path = Path("results/act0r2/validation.json")
    act0r2_manifest_path = Path("results/act0r2/capture_manifest.json")
    act0r2_frames_path = Path("results/act0r2/frame_metrics.csv")
    cf0_path = Path("results/cf0/validation.json")
    cf0_manifest_path = Path("results/cf0/capture_manifest.json")
    cf0_frame_path = Path("results/cf0/frame_manifest.csv")
    cf0_branch_path = Path("results/cf0/branch_summary.csv")
    cf0_fold_path = Path("results/cf0/fold_assignments.csv")
    cf0_action_path = Path("results/cf0/action_selection.csv")
    probe0_path = Path("results/probe0/validation.json")
    probe0_source_path = Path("results/probe0/source_manifest.json")
    probe0_predictions_path = Path("results/probe0/predictions.csv")
    probe0_config_path = Path("configs/experiments/probe0.yaml")
    probe0r1_path = Path("results/probe0/validation_r1.json")
    probe0r1_bootstrap_path = Path("results/probe0/bootstrap_r1.json")
    probe0r1_ablation_path = Path("results/probe0/causal_ablation_r1.csv")
    probe0r1_config_path = Path("configs/experiments/probe0r1.yaml")
    r1 = json.loads(r1_path.read_text()) if r1_path.exists() else {}
    mask1 = json.loads(mask1_path.read_text()) if mask1_path.exists() else {}
    mask1_r1 = json.loads(mask1_r1_path.read_text()) if mask1_r1_path.exists() else {}
    event_r1 = json.loads(event_r1_path.read_text()) if event_r1_path.exists() else {}
    repeatability_r1 = json.loads(repeatability_r1_path.read_text()) if repeatability_r1_path.exists() else {}
    obs0 = json.loads(obs0_path.read_text()) if obs0_path.exists() else {}
    probes = json.loads(probe_path.read_text()) if probe_path.exists() else {}
    obs0r1 = json.loads(obs0r1_path.read_text()) if obs0r1_path.exists() else {}
    obs0r1_probes = json.loads(obs0r1_probe_path.read_text()) if obs0r1_probe_path.exists() else {}
    obs0r1_alias = json.loads(obs0r1_alias_path.read_text()) if obs0r1_alias_path.exists() else {}
    obs0r1_history = json.loads(obs0r1_history_path.read_text()) if obs0r1_history_path.exists() else {}
    act0_validation = json.loads(act0_validation_path.read_text()) if act0_validation_path.exists() else {}
    act0_manifest = json.loads(act0_manifest_path.read_text()) if act0_manifest_path.exists() else {}
    act0s = json.loads(act0s_path.read_text()) if act0s_path.exists() else {}
    act0s_coverage = json.loads(act0s_coverage_path.read_text()) if act0s_coverage_path.exists() else {}
    act0r = json.loads(act0r_validation_path.read_text()) if act0r_validation_path.exists() else {}
    act0r_search = json.loads(act0r_search_path.read_text()) if act0r_search_path.exists() else {}
    cap0 = json.loads(cap0_path.read_text()) if cap0_path.exists() else {}
    cap0_probe = json.loads(cap0_probe_path.read_text()) if cap0_probe_path.exists() else {}
    cap0_root_cause = json.loads(cap0_root_cause_path.read_text()) if cap0_root_cause_path.exists() else {}
    act0r1 = json.loads(act0r1_path.read_text()) if act0r1_path.exists() else {}
    act0r1_offline = json.loads(act0r1_offline_path.read_text()) if act0r1_offline_path.exists() else {}
    act0r2 = json.loads(act0r2_path.read_text()) if act0r2_path.exists() else {}
    act0r2_manifest = json.loads(act0r2_manifest_path.read_text()) if act0r2_manifest_path.exists() else {}
    cf0 = json.loads(cf0_path.read_text()) if cf0_path.exists() else {}
    cf0_manifest = json.loads(cf0_manifest_path.read_text()) if cf0_manifest_path.exists() else {}
    probe0 = json.loads(probe0_path.read_text()) if probe0_path.exists() else {}
    probe0_source = json.loads(probe0_source_path.read_text()) if probe0_source_path.exists() else {}
    probe0_config = yaml.safe_load(probe0_config_path.read_text()) if probe0_config_path.exists() else {}
    probe0r1 = json.loads(probe0r1_path.read_text()) if probe0r1_path.exists() else {}
    probe0r1_bootstrap = json.loads(probe0r1_bootstrap_path.read_text()) if probe0r1_bootstrap_path.exists() else {}
    probe0r1_config = yaml.safe_load(probe0r1_config_path.read_text()) if probe0r1_config_path.exists() else {}
    act0s_matrix_count = 0
    if act0s_matrix_path.exists():
        with act0s_matrix_path.open(newline="") as handle:
            act0s_matrix_count = sum(1 for _row in csv.DictReader(handle))
    act0s_assets = sorted(Path("docs/assets/act0_screening").glob("candidate_*_screening.jpg"))
    act0s_gates = act0s.get("gates", {})
    act0r_gates = act0r.get("gates", {})
    cap0_gates = cap0.get("gates", {})
    act0r1_gates = act0r1.get("gates", {})
    act0r1_offline_gates = act0r1_offline.get("gates", {})
    act0r2_gates = act0r2.get("gates", {})
    cf0_gates = cf0.get("gates", {})
    probe0_gates = probe0.get("gates", {})
    probe0r1_gates = probe0r1.get("gates", {})
    act0r1_offline_frame_count = 0
    if act0r1_offline_frames_path.exists():
        with act0r1_offline_frames_path.open(newline="") as handle:
            act0r1_offline_frame_count = sum(1 for _row in csv.DictReader(handle))
    act0r1_offline_hash_count = 0
    if act0r1_offline_hashes_path.exists():
        with act0r1_offline_hashes_path.open(newline="") as handle:
            act0r1_offline_hash_count = sum(1 for _row in csv.DictReader(handle))
    act0r1_offline_assets = sorted(
        Path("docs/assets/act0r1").glob("offline_*.jpg"))
    act0r2_assets = sorted(Path("docs/assets/act0r2").glob("*.jpg"))
    cf0_assets = sorted(Path("docs/assets/cf0").glob("*.jpg"))
    cf0_csv_counts = {}
    for name, path in (("frames", cf0_frame_path), ("branches", cf0_branch_path),
                       ("folds", cf0_fold_path), ("actions", cf0_action_path)):
        if path.exists():
            with path.open(newline="") as handle:
                cf0_csv_counts[name] = sum(1 for _row in csv.DictReader(handle))
    probe0_predictions = []
    if probe0_predictions_path.exists():
        with probe0_predictions_path.open(newline="") as handle:
            probe0_predictions = list(csv.DictReader(handle))
    cf0_frame_lookup = {}
    if cf0_frame_path.exists():
        with cf0_frame_path.open(newline="") as handle:
            cf0_frame_lookup = {int(row["frame_id"]): row for row in csv.DictReader(handle)}
    probe0_assets = sorted(Path("docs/assets/probe0").glob("*.jpg"))
    probe0r1_assets = sorted(Path("docs/assets/probe0r1").glob("*.jpg"))
    probe0r1_ablation_rows = []
    if probe0r1_ablation_path.exists():
        with probe0r1_ablation_path.open(newline="") as handle:
            probe0r1_ablation_rows = list(csv.DictReader(handle))
    probe0_folds = probe0_source.get("folds", [])
    probe0_fold_groups = [set(row.get("test_start_ids", [])) for row in probe0_folds]
    probe0_fold_integrity = bool(probe0_folds) and all(
        set(row.get("train_start_ids", [])).isdisjoint(row.get("test_start_ids", []))
        for row in probe0_folds)
    probe0_fold_integrity = probe0_fold_integrity and len(
        set().union(*probe0_fold_groups)) == 13 and sum(
        len(group) for group in probe0_fold_groups) == 13
    probe0_prediction_integrity = len(probe0_predictions) == 26
    if probe0_prediction_integrity:
        grouped_predictions = {}
        for row in probe0_predictions:
            grouped_predictions.setdefault(row["start_id"], []).append(row)
        probe0_prediction_integrity = (
            len(grouped_predictions) == 13 and
            all({item["probe_direction"] for item in rows} == {"LEFT", "RIGHT"}
                and len({item["fold"] for item in rows}) == 1
                and all(float(item["probe_distance_m"]) == 1.0 for item in rows)
                for rows in grouped_predictions.values()))
    probe0_preboundary_inputs = bool(probe0_predictions)
    for row in probe0_predictions:
        previous = cf0_frame_lookup.get(int(row["previous_frame_id_gt_audit_only"]), {})
        current = cf0_frame_lookup.get(int(row["current_frame_id_gt_audit_only"]), {})
        probe0_preboundary_inputs = probe0_preboundary_inputs and (
            int(previous.get("step_index", -1)) == 1 and
            int(current.get("step_index", -1)) == 2 and
            previous.get("first_physical_termination") == "False" and
            current.get("first_physical_termination") == "False" and
            previous.get("model_visible_termination") == "False" and
            current.get("model_visible_termination") == "False")
    probe0_primary = probe0.get("primary_1m", {})
    probe0_pooled = probe0_primary.get("pooled", {})
    probe0_left = probe0_primary.get("LEFT_probe", {})
    probe0_right = probe0_primary.get("RIGHT_probe", {})
    probe0_thresholds = probe0_config.get("gates", {})
    probe0_p0 = probe0.get("P0", {}).get("accuracy")
    probe0_expected_conditions = {
        "pooled_accuracy": probe0_pooled.get("accuracy", -1) >= probe0_thresholds.get(
            "pooled_accuracy_minimum", float("inf")),
        "p0_accuracy_improvement": probe0.get("P1_minus_P0_accuracy", -1) >=
            probe0_thresholds.get("p0_accuracy_improvement_minimum", float("inf")),
        "pooled_ci_lower": probe0_pooled.get("bootstrap_95_ci", {}).get(
            "accuracy", {}).get("lower", -1) > probe0_thresholds.get(
                "pooled_ci_lower_strictly_greater_than", float("inf")),
        "left_probe_accuracy": probe0_left.get("accuracy", -1) >= probe0_thresholds.get(
            "left_probe_accuracy_minimum", float("inf")),
        "right_probe_accuracy": probe0_right.get("accuracy", -1) >= probe0_thresholds.get(
            "right_probe_accuracy_minimum", float("inf")),
        "preboundary_and_group_integrity": probe0_preboundary_inputs and
            probe0_fold_integrity and probe0_prediction_integrity,
    }
    probe0r1_metrics = probe0r1.get("ablations", {})
    probe0r1_differences = probe0r1.get("paired_differences", {})
    probe0r1_thresholds = probe0r1_config.get("gates", {})
    def r1_difference(name):
        return probe0r1_differences.get(name, {}).get("observed", {}).get("accuracy", -999.0)
    probe0r1_expected_conditions = {
        "E2_minus_E1_accuracy": r1_difference("E2_minus_E1") >=
            probe0r1_thresholds.get("E2_minus_E1_accuracy_minimum", float("inf")),
        "E2_minus_E3_SWAP_accuracy": r1_difference("E2_minus_E3_SWAP") >=
            probe0r1_thresholds.get("E2_minus_E3_SWAP_accuracy_minimum", float("inf")),
        "E2_minus_E4_ZERO_accuracy": r1_difference("E2_minus_E4_ZERO") >=
            probe0r1_thresholds.get("E2_minus_E4_ZERO_accuracy_minimum", float("inf")),
        "E2_minus_E1_ci_lower": probe0r1_differences.get("E2_minus_E1", {}).get(
            "bootstrap_95_ci", {}).get("accuracy", {}).get("lower", -999.0) >
            probe0r1_thresholds.get(
                "E2_minus_E1_accuracy_ci_lower_strictly_greater_than", float("inf")),
    }
    probe0r1_historical_hashes_match = bool(probe0r1.get("historical_probe0_sha256"))
    for name, expected_hash in probe0r1.get("historical_probe0_sha256", {}).items():
        path = Path(name)
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        probe0r1_historical_hashes_match = (
            probe0r1_historical_hashes_match and actual_hash == expected_hash)
    act0r2_frame_count = 0
    if act0r2_frames_path.exists():
        with act0r2_frames_path.open(newline="") as handle:
            act0r2_frame_count = sum(1 for _row in csv.DictReader(handle))
    act0r_frame_count = 0
    if act0r_frame_manifest_path.exists():
        with act0r_frame_manifest_path.open(newline="") as handle:
            act0r_frame_count = sum(1 for _row in csv.DictReader(handle))
    act0r_operator_count = 0
    if act0r_operator_path.exists():
        with act0r_operator_path.open(newline="") as handle:
            act0r_operator_count = sum(1 for _row in csv.DictReader(handle))
    act0r_assets = sorted(Path("docs/assets/act0r").glob("*.jpg"))
    obs0r1_frame_count = 0
    if obs0r1_frames_path.exists():
        with obs0r1_frames_path.open(newline="") as handle:
            obs0r1_frame_count = sum(1 for _row in csv.DictReader(handle))
    old_hashes_match = bool(obs0r1.get("historical_obs0_files_unchanged"))
    for name, expected in obs0r1.get("historical_obs0_files_unchanged", {}).items():
        path = Path(name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        old_hashes_match = old_hashes_match and actual == expected
    obs0r1_required_gates = ("HISTORY_BOUNDARY_LEAKAGE_FIXED",
                             "TRAIN_ONLY_PREPROCESSING",
                             "SYNTHETIC_ALIGNMENT_TEST",
                             "OBS0R1_REPRODUCIBILITY")
    obs0r1_gates = obs0r1.get("gates", {})
    runtime_safety = obs0r1.get("runtime_safety", {})
    history_first = obs0r1_history.get("first_frame_history_valid", {})
    obs0r1_assets = sorted(Path("docs/assets/obs0r1").glob("*.jpg"))
    r1_gates = r1.get("gates", {})
    mask1_gates = mask1.get("gates", {})
    mask1_required = ("MASK1_CAPTURE_PAIRING", "NORMAL_LOCK", "INITIAL_IN",
                      "STRADDLE_CONFIRMATION", "ADAPTIVE_STOP",
                      "COVERAGE_MONOTONICITY", "WORLD_BOUNDARY_ESTIMATE")
    checks = {
        "required_files": not missing,
        "geometry_reference_schema": reference.get("schema") == "geo05r2.geometry_reference_audit.v1",
        "geometry_reference_count": reference.get("geometry_reference_count", 0) >= 60 and len(reference.get("frames", [])) >= 60,
        "operator_visual_review_pending": reference.get("operator_visual_review_completed") is False and reference.get("operator_visual_review_required") is True,
        "not_manual_audit": not any("manual" in key.lower() for key in reference.keys()),
        "depth_metric_evidence": depth_result["status"] == "PASS",
        "surface_stats_consistent": bool(surface_checks) and all(surface_checks),
        "boundary_semantics_fail": gates.get("BOUNDARY_SEMANTICS", {}).get("status") == "FAIL",
        "event_coverage_fail": gates.get("EVENT_COVERAGE", {}).get("status") == "FAIL",
        "ready_for_jepa_fail": gates.get("READY_FOR_JEPA", {}).get("status") == "FAIL",
        "no_private_paths": not forbidden,
        "validation_schema": validation.get("schema") == "geo05r2.validation.v2",
        "mask0r1_decoder": r1.get("decoder_audit", {}).get("agreement", 0.0) >= 0.9999 and r1.get("decoder_audit", {}).get("error_pixels", 1) == 0,
        "mask0r1_edge_gate": r1_gates.get("READY_FOR_ADAPTIVE_PILOT", {}).get("status") == "PASS" and r1_gates.get("LEGACY_EDGE_ALIGNMENT", {}).get("status") == "FAIL",
        "mask1_pilot_validation": bool(mask1.get("trajectories")) and all(mask1_gates.get(name, {}).get("status") == "PASS" for name in mask1_required),
        "mask1_jepa_not_evaluated": mask1_gates.get("READY_FOR_JEPA", {}).get("status") == "NOT_EVALUATED",
        "mask1r1_reanalysis": mask1_r1.get("schema") == "mask1.validation_r1.v1" and mask1_r1.get("historical_result_files_modified") is False,
        "mask1r1_event_gates": event_r1.get("gates", {}).get("FIRST_STRADDLE_DETECTION", {}).get("status") == "PASS" and event_r1.get("gates", {}).get("STOP_OVERSHOOT", {}).get("status") == "FAIL",
        "mask1r1_repeatability": repeatability_r1.get("gates", {}).get("WORLD_BOUNDARY_ABSOLUTE_ACCURACY", {}).get("status") == "NOT_EVALUATED",
        "obs0_schema": obs0.get("schema") == "obs0.validation.v1" and obs0.get("sample_count") == 20,
        "obs0_complete_direction_holdout": probes.get("split") == "complete direction holdout" and set(probes.get("probes", {})) == {"P0", "P1", "P2", "P3", "P4", "P5", "P6"},
        "obs0_action_selection_not_evaluated": obs0.get("gates", {}).get("ACTION_SELECTION_OBSERVABILITY", {}).get("status") == "NOT_EVALUATED",
        "obs0_jepa_not_evaluated": obs0.get("gates", {}).get("READY_FOR_JEPA", {}).get("status") == "NOT_EVALUATED",
        "obs0r1_required_files": all(path.exists() for path in
                                      (obs0r1_path, obs0r1_probe_path,
                                       obs0r1_alias_path, obs0r1_history_path,
                                       obs0r1_frames_path)),
        "obs0r1_phase_a_gates": all(obs0r1_gates.get(name, {}).get("status") == "PASS"
                                    for name in obs0r1_required_gates),
        "obs0r1_history_is_group_local": obs0r1_history.get("cross_group_previous_forbidden") is True and
                                         bool(history_first) and not any(history_first.values()),
        "obs0r1_train_only_preprocessing": obs0r1_probes.get("train_only_preprocessing") is True,
        "obs0r1_historical_files_unchanged": old_hashes_match,
        "obs0r1_manifest_consistency": obs0r1_frame_count == 20 and
                                      obs0r1_probes.get("records") == 20 and
                                      obs0r1_alias.get("pair_count") == 190,
        "obs0r1_memory_safety": runtime_safety.get("descriptor_dim") == 128 and
                               runtime_safety.get("max_linear_system_dimension", 10**9) <= 20 and
                               runtime_safety.get("opencv_threads") == 1 and
                               runtime_safety.get("address_space_limit_bytes") == 2 * 1024 ** 3,
        "obs0r1_synthetic_alignment": obs0r1_probes.get("synthetic_alignment", {}).get("status") == "PASS" and
                                     obs0r1_probes.get("synthetic_alignment", {}).get("metrics", {}).get("phase_aligned_ssim", -1) >
                                     obs0r1_probes.get("synthetic_alignment", {}).get("metrics", {}).get("raw_ssim", 1),
        "obs0r1_audit_assets": len(obs0r1_assets) >= 6,
        "act0_historical_zero_of_twelve_preserved": act0_validation.get("selected_surface_count") == 0 and
                                                     act0_validation.get("candidate_count") == 12 and
                                                     act0_manifest.get("selected_count") == 0 and
                                                     act0_manifest.get("candidate_count") == 12,
        "act0s_required_files": all(path.exists() for path in
                                     (act0s_path, act0s_matrix_path, act0s_coverage_path,
                                      Path("docs/ACT0_SCREENING_AUDIT.md"))),
        "act0s_candidate_consistency": act0s.get("schema") == "act0.screening_audit.v2" and
                                        len(act0s.get("candidates", [])) == 12 and
                                        act0s_matrix_count == 12,
        "act0s_public_assets": len(act0s_assets) == 12 and
                               Path("docs/assets/act0_screening/candidate_overview.jpg").exists(),
        "act0s_not_observed_not_scene_fail": act0s.get("rejection_ablation", {}).get("scout_edge_not_observed") == 2 and
                                             act0s.get("rejection_ablation", {}).get("visual_semantics_fail") == 2,
        "act0s_tiers_independent": act0s_gates.get("SCREENING_DEFINITION_VALID", {}).get("tier_p_can_veto_tier_v") is False and
                                   act0s_gates.get("VISUAL_EVENT_SURFACE_COUNT", {}).get("count") == 6 and
                                   act0s_gates.get("PHYSICAL_PLANE_SURFACE_COUNT", {}).get("count") == 3,
        "act0s_metric_fails_closed": act0s_gates.get("METRIC_REPEATABLE_SURFACE_COUNT", {}).get("status") == "NOT_EVALUATED" and
                                     act0s_gates.get("METRIC_REPEATABLE_SURFACE_COUNT", {}).get("evaluable_count") == 0 and
                                     all(row.get("tier_m", {}).get("uses_legacy_plane_or_bbox") is False
                                         for row in act0s.get("candidates", [])),
        "act0s_evidence_gap_disclosed": act0s_gates.get("PUBLIC_ACT0_EVIDENCE", {}).get("status") == "FAIL" and
                                        act0s_coverage.get("persisted_actual_rgb_views") == 12 and
                                        act0s_coverage.get("expected_actual_rgb_views") == 36 and
                                        act0s_coverage.get("raw_sensor_data_size_bytes") == 0,
        "act0s_resource_limits": act0s.get("resource_limits", {}).get("configured_address_space_limit_bytes") == 4294967296 and
                                  act0s.get("resource_limits", {}).get("actual_outer_address_space_limit_bytes") == 2147483648,
        "act0s_terminal_gates": act0s_gates.get("READY_FOR_ADAPTIVE_RESCOUT", {}).get("status") == "CONDITIONAL_PASS" and
                                act0s_gates.get("READY_FOR_COUNTERFACTUAL_ROLLOUT", {}).get("status") == "FAIL" and
                                act0s_gates.get("READY_FOR_DATASET_EXPANSION", {}).get("status") == "NOT_EVALUATED" and
                                act0s_gates.get("READY_FOR_JEPA", {}).get("status") == "NOT_EVALUATED" and
                                act0s_gates.get("OPERATOR_VISUAL_REVIEW", {}).get("status") == "PENDING",
        "act0r_required_compact_files": all(path.exists() for path in
                                              (act0r_validation_path, act0r_search_path,
                                               act0r_frame_manifest_path, act0r_operator_path,
                                               Path("docs/ACT0R_VISUAL_AUDIT.md"),
                                               Path("configs/experiments/act0r.yaml"))),
        "act0r_incomplete_capture_disclosed": act0r.get("schema") == "act0r.validation.v1" and
                                                act0r.get("run_status") == "INCOMPLETE_CAPTURE" and
                                                act0r.get("saved_frame_count") == 1 and
                                                act0r.get("failure", {}).get("search_plan_complete") is True,
        "act0r_search_plan_complete": act0r_search.get("schema") == "act0r.search_plan_checkpoint.v1" and
                                       act0r_search.get("complete") is True and
                                       act0r_search.get("candidate_count") == 4 and
                                       [row.get("candidate_index") for row in act0r_search.get("plans", [])] == [1, 7, 10, 19] and
                                       all(set(row.get("searches", {})) == {"LEFT", "RIGHT"}
                                           for row in act0r_search.get("plans", [])),
        "act0r_raw_gate_fails_closed": act0r_gates.get("SENSOR_QUADRUPLET_PAIRING", {}).get("status") == "FAIL" and
                                       act0r_gates.get("SENSOR_QUADRUPLET_PAIRING", {}).get("available_pairing_valid") is True and
                                       act0r_gates.get("SENSOR_QUADRUPLET_PAIRING", {}).get("available_rgb_visual_integrity") == "FAIL" and
                                       act0r_gates.get("RAW_PIXEL_EVIDENCE_AVAILABLE", {}).get("status") == "FAIL" and
                                       act0r_gates.get("RAW_PIXEL_EVIDENCE_AVAILABLE", {}).get("available_frames") == 1 and
                                       act0r_gates.get("RAW_PIXEL_EVIDENCE_AVAILABLE", {}).get("expected_frames") == 60,
        "act0r_no_outcome_override": act0r_gates.get("CONFIG_OUTCOME_OVERRIDE_ABSENT", {}).get("status") == "PASS" and
                                     act0r_gates.get("TIER_V_RECOMPUTED_FROM_PIXELS", {}).get("uses_act0s_outcomes") is False,
        "act0r_no_false_tier_claim": act0r_gates.get("TIER_V_RECOMPUTED_FROM_PIXELS", {}).get("status") == "FAIL" and
                                    act0r_gates.get("TIER_V_RECOMPUTED_FROM_PIXELS", {}).get("recomputed_sides") == 0 and
                                    act0r_gates.get("OFFICIAL_TIER_M", {}).get("status") == "FAIL" and
                                    act0r_gates.get("OFFICIAL_TIER_M", {}).get("uses_plane") is False and
                                    act0r_gates.get("OFFICIAL_TIER_M", {}).get("uses_bbox") is False,
        "act0r_terminal_gates": act0r_gates.get("READY_FOR_COUNTERFACTUAL_ROLLOUT", {}).get("status") == "FAIL" and
                                act0r_gates.get("READY_FOR_DATASET_EXPANSION", {}).get("status") == "NOT_EVALUATED" and
                                act0r_gates.get("READY_FOR_JEPA", {}).get("status") == "NOT_EVALUATED" and
                                act0r_gates.get("OPERATOR_VISUAL_REVIEW", {}).get("status") == "PENDING",
        "act0r_compact_manifests": act0r_frame_count == 1 and act0r_operator_count == 8,
        "act0r_public_failure_assets": len(act0r_assets) == 7 and
                                        act0r_gates.get("PUBLIC_VISUAL_EVIDENCE", {}).get("status") == "FAIL" and
                                        Path("docs/assets/act0r/candidate_01_current_center.jpg").exists() and
                                        Path("docs/assets/act0r/candidate_19_right_unresolved.jpg").exists(),
        "cap0_required_compact_files": all(path.exists() for path in
                                            (cap0_path, cap0_probe_path, cap0_root_cause_path,
                                             cap0_matrix_path, cap0_manifest_path,
                                             Path("docs/CAP0_SENSOR_AUDIT.md"),
                                             Path("configs/experiments/cap0.yaml"))),
        "cap0_matrix_complete": cap0.get("schema") == "cap0.validation.v1" and
                                cap0.get("saved_frame_count") == 15 and
                                set(cap0.get("tests", {})) == {"H1", "H2", "H3", "H4", "H5"} and
                                all(row.get("capture_status") == "PASS" and
                                    len(row.get("frames", [])) == 3
                                    for row in cap0.get("tests", {}).values()),
        "cap0_health_gates": all(cap0_gates.get(name, {}).get("status") == "PASS"
                                 for name in ("TICK_FAIL_FAST", "QUEUE_DEADLINE",
                                              "RAW_BUFFER_OWNERSHIP", "RAW_LENGTH_AND_HASH",
                                              "GPU_WARMUP_COMPLETE",
                                              "KNOWN_GOOD_POSE_RGB_INTEGRITY",
                                              "QUARTET_PAIRING_HEALTH",
                                              "POST_TELEPORT_HEALTH", "RENDER_INTEGRITY",
                                              "ROOT_CAUSE_CLASSIFIED", "RSS_WATCHDOG")),
        "cap0_address_space_probe_disclosed": cap0_probe.get("status") == "FAIL" and
                                              cap0_probe.get("script_completed") is False and
                                              cap0_probe.get("termination") == "EXTERNAL_TIMEOUT" and
                                              cap0_probe.get("outer_exit_code") == 124 and
                                              cap0_probe.get("actual_outer_address_space_limit_bytes") == 2147483648,
        "cap0_root_cause": cap0_root_cause.get("status") == "PASS" and
                           cap0_root_cause.get("classification") == "PYTHON_ADDRESS_SPACE_LIMIT_FAILURE" and
                           cap0_root_cause.get("confidence") == "CONFIRMED",
        "cap0_resource_limits": cap0.get("resources", {}).get("configured_python_as_limit_bytes") == 4294967296 and
                                cap0.get("resources", {}).get("actual_outer_python_as_limit_bytes") == 4294967296 and
                                cap0.get("resources", {}).get("rss_watchdog_exceeded") is False,
        "act0r1_required_compact_files": all(path.exists() for path in
                                              (act0r1_path, Path("docs/ACT0R1_PILOT_AUDIT.md"),
                                               Path("docs/assets/act0r1/left_all_roles.jpg"),
                                               Path("docs/assets/act0r1/straddle_raw_vs_png.jpg"),
                                               Path("docs/assets/act0r1/straddle_sensors.jpg"))),
        "act0r1_capture_complete": act0r1.get("schema") == "act0r1.validation.v1" and
                                    len(act0r1.get("frames", [])) == 8 and
                                    [row.get("capture_role") for row in act0r1.get("frames", [])] ==
                                    ["CENTER", "INSIDE", "PRE_EDGE", "STRADDLE",
                                     "STRADDLE_REPEAT_1", "STRADDLE_REPEAT_2",
                                     "STRADDLE_REPEAT_3", "POST_EDGE"],
        "act0r1_capture_gates": all(act0r1_gates.get(name, {}).get("status") == "PASS"
                                    for name in ("CAPTURE_STACK_RECOVERED",
                                                 "CANDIDATE1_LEFT_CAPTURE_COMPLETE",
                                                 "SENSOR_QUADRUPLET_PAIRING",
                                                 "RGB_VISUAL_INTEGRITY",
                                                 "SAME_POSE_CONFIRMATION",
                                                 "TARGET_INSTANCE_STABILITY",
                                                 "READY_TO_RESUME_ACT0R")),
        "act0r1_ssim_scope": act0r1_gates.get("RGB_VISUAL_INTEGRITY", {}).get(
                                "threshold_application") ==
                                "consecutive SSIM is gated only within frozen-pose groups" and
                                min(act0r1_gates.get("RGB_VISUAL_INTEGRITY", {}).get(
                                    "same_pose_consecutive_ssim", [0.0])) >= 0.9,
        "act0r1_terminal_gates": act0r1_gates.get(
                                    "READY_FOR_COUNTERFACTUAL_ROLLOUT", {}).get(
                                    "status") == "NOT_EVALUATED" and
                                    act0r1_gates.get("READY_FOR_JEPA", {}).get(
                                    "status") == "NOT_EVALUATED" and
                                    act0r1.get("constraints", {}).get("rollout_run") is False and
                                    act0r1.get("constraints", {}).get("jepa_training_run") is False,
        "act0r1_offline_required_files": all(path.exists() for path in
                                              (act0r1_offline_path,
                                               act0r1_offline_frames_path,
                                               act0r1_offline_hashes_path,
                                               Path("docs/ACT0R1_OFFLINE_BOUNDARY_AUDIT.md"),
                                               Path("scripts/audit_act0r1_offline.py"))),
        "act0r1_offline_raw_provenance": act0r1_offline.get("schema") ==
                                          "act0r1.offline_boundary_audit.v1" and
                                          act0r1_offline_frame_count == 8 and
                                          act0r1_offline_hash_count == 56 and
                                          act0r1_offline_gates.get(
                                              "RAW_HASH_AUDIT", {}).get("status") == "PASS" and
                                          act0r1_offline_gates.get(
                                              "RAW_HASH_AUDIT", {}).get("checked_file_count") == 56 and
                                          act0r1_offline_gates.get(
                                              "SENSOR_PAIRING", {}).get("paired_frame_count") == 8,
        "act0r1_offline_historical_inputs_unchanged":
                                          act0r1_offline.get("source", {}).get(
                                              "validation_sha256") ==
                                          hashlib.sha256(act0r1_path.read_bytes()).hexdigest() and
                                          act0r1_offline.get("source", {}).get(
                                              "checkpoint", {}).get("status") == "PASS" and
                                          hashlib.sha256(act0r_search_path.read_bytes()).hexdigest() ==
                                          "a56310883bb15513ea25c97c919d7faf14edb217b1a05fb0c4e12b060c664f73",
        "act0r1_offline_role_independence": act0r1_offline_gates.get(
                                              "ROLE_LABEL_INDEPENDENCE", {}).get(
                                              "status") == "PASS" and
                                              act0r1_offline.get(
                                                  "role_label_policy", {}).get(
                                                  "capture_roles_used_as_ground_truth") is False and
                                              act0r1_offline.get(
                                                  "repeated_pose_group", {}).get(
                                                  "uses_role_labels") is False,
        "act0r1_offline_physical_boundary": act0r1_offline_gates.get(
                                                "TARGET_MASK_PIXEL_VALID", {}).get(
                                                "status") == "PASS" and
                                                act0r1_offline_gates.get(
                                                "LEFT_BOUNDARY_TYPE_RESOLVED", {}).get(
                                                "status") == "PASS" and
                                                act0r1_offline_gates.get(
                                                "LEFT_BOUNDARY_TYPE_RESOLVED", {}).get(
                                                "boundary_type") == "PHYSICAL_TERMINATION" and
                                                act0r1_offline.get(
                                                "boundary_consensus", {}).get(
                                                "consensus_count") == 4,
        "act0r1_offline_tier_gates": act0r1_offline_gates.get(
                                        "TIER_V", {}).get("status") == "PASS" and
                                        act0r1_offline_gates.get(
                                        "TIER_V", {}).get("uses_role_labels") is False and
                                        act0r1_offline_gates.get(
                                        "OFFICIAL_TIER_M", {}).get("status") == "PASS" and
                                        act0r1_offline_gates.get(
                                        "OFFICIAL_TIER_M", {}).get("spread_m", 1.0) <= 0.25 and
                                        act0r1_offline_gates.get(
                                        "OFFICIAL_TIER_M", {}).get("uses_plane") is False and
                                        act0r1_offline_gates.get(
                                        "OFFICIAL_TIER_M", {}).get("uses_bbox") is False and
                                        act0r1_offline_gates.get(
                                        "SAME_POSE_CONFIRMATION", {}).get("status") == "PASS",
        "act0r1_offline_terminal_gates": act0r1_offline_gates.get(
                                             "EXTERNAL_VISUAL_REVIEW", {}).get(
                                             "status") == "PENDING" and
                                             act0r1_offline_gates.get(
                                             "READY_FOR_CANDIDATE1_RIGHT", {}).get(
                                             "status") == "CONDITIONAL_PASS" and
                                             act0r1_offline_gates.get(
                                             "READY_FOR_COUNTERFACTUAL_ROLLOUT", {}).get(
                                             "status") == "NOT_EVALUATED" and
                                             act0r1_offline_gates.get(
                                             "READY_FOR_JEPA", {}).get(
                                             "status") == "NOT_EVALUATED",
        "act0r1_offline_fault_attribution": act0r1_offline.get(
                                                "fault_attribution", {}).get(
                                                "TWO_GIB_ADDRESS_SPACE_FAILURE", {}).get(
                                                "status") == "CONFIRMED" and
                                                act0r1_offline.get(
                                                "fault_attribution", {}).get(
                                                "HISTORICAL_TRIANGLE_ARTIFACT_ROOT_CAUSE", {}).get(
                                                "status") ==
                                                "LIKELY_BUT_NOT_UNIQUELY_PROVEN",
        "act0r1_offline_public_assets": len(act0r1_offline_assets) == 6 and
                                        all(path.stat().st_size < 2_000_000
                                            for path in act0r1_offline_assets),
        "act0r1_offline_no_forbidden_geometry":
                                        act0r1_offline.get("constraints", {}).get(
                                            "legacy_plane_used") is False and
                                        act0r1_offline.get("constraints", {}).get(
                                            "legacy_bbox_used") is False and
                                        act0r1_offline.get("constraints", {}).get(
                                            "manual_boundary_used") is False and
                                        act0r1_offline.get("constraints", {}).get(
                                            "historical_results_modified") is False,
        "act0r2_required_compact_files": all(path.exists() for path in (
                                                act0r2_path, act0r2_manifest_path,
                                                act0r2_frames_path,
                                                Path("docs/ACT0R2_VISUAL_AUDIT.md"))) and
                                           act0r2.get("schema") == "act0r2.validation.v1" and
                                           act0r2_frame_count == 14,
        "act0r2_checkpoint_pose_source": act0r2_manifest.get("status") == "CAPTURE_COMPLETE" and
                                          act0r2_manifest.get("saved_frame_count") == 15 and
                                          len(act0r2_manifest.get("frames", [])) == 15 and
                                          act0r2_manifest.get("checkpoint_plan", {}).get(
                                              "locator_center_pose") ==
                                          act0r_search.get("plans", [{}])[0].get(
                                              "locator_center_pose") and
                                          act0r2_manifest.get("constraints", {}).get(
                                              "poses_old_new_used") is False,
        "act0r2_raw_hash_and_pairing": act0r2.get("raw", {}).get(
                                            "hash_audit", {}).get("status") == "PASS" and
                                        act0r2.get("raw", {}).get(
                                            "hash_audit", {}).get("checked_file_count") == 90 and
                                        act0r2_gates.get("SENSOR_PAIRING", {}).get(
                                            "status") == "PASS" and
                                        act0r2_gates.get("SENSOR_PAIRING", {}).get(
                                            "frame_count") == 15,
        "act0r2_center_alignment_and_same_start": act0r2_gates.get(
                                            "CHECKPOINT_POSE_ALIGNMENT", {}).get(
                                            "status") == "PASS" and
                                        act0r2_gates.get(
                                            "CHECKPOINT_POSE_ALIGNMENT", {}).get(
                                            "position_error_m", 1.0) <= 0.01 and
                                        act0r2_gates.get(
                                            "CHECKPOINT_POSE_ALIGNMENT", {}).get(
                                            "rotation_error_deg", 1.0) <= 0.05 and
                                        act0r2_gates.get("BILATERAL_SAME_START", {}).get(
                                            "status") == "PASS" and
                                        act0r2_gates.get("CENTER_LEFT_BOUNDARY_ABSENT", {}).get(
                                            "status") == "PASS" and
                                        act0r2_gates.get("CENTER_RIGHT_BOUNDARY_ABSENT", {}).get(
                                            "status") == "PASS",
        "act0r2_bilateral_event_ordering": all(
                                        act0r2.get("directions", {}).get(direction, {}).get(
                                            "computed_states", [])[:3] == [
                                                "NO_VALID_EXTERNAL_BOUNDARY", "APPROACH",
                                                "FIRST_PHYSICAL_TERMINATION"] and
                                        act0r2.get("directions", {}).get(direction, {}).get(
                                            "event_ordering", {}).get("status") == "PASS" and
                                        act0r2.get("directions", {}).get(direction, {}).get(
                                            "event_ordering", {}).get("uses_role_labels") is False
                                        for direction in ("LEFT", "RIGHT")),
        "act0r2_bilateral_physical_boundary": all(
                                        act0r2.get("directions", {}).get(direction, {}).get(
                                            "boundary_consensus", {}).get("boundary_type") ==
                                            "PHYSICAL_TERMINATION" and
                                        act0r2.get("directions", {}).get(direction, {}).get(
                                            "boundary_consensus", {}).get("consensus_count") == 4 and
                                        act0r2.get("directions", {}).get(direction, {}).get(
                                            "tier_v", {}).get("status") == "PASS"
                                        for direction in ("LEFT", "RIGHT")),
        "act0r2_same_pose_not_multiview": all(
                                        act0r2.get("directions", {}).get(direction, {}).get(
                                            "same_pose", {}).get("status") == "PASS" and
                                        act0r2.get("directions", {}).get(direction, {}).get(
                                            "same_pose_world_repeatability", {}).get("status") ==
                                            "PASS" and
                                        act0r2.get("directions", {}).get(direction, {}).get(
                                            "same_pose_world_repeatability", {}).get("spread_m") == 0.0
                                        for direction in ("LEFT", "RIGHT")) and
                                        act0r2_gates.get("MULTIVIEW_REPEATABILITY", {}).get(
                                            "status") == "NOT_EVALUATED",
        "act0r2_terminal_gates": act0r2_gates.get("EXTERNAL_VISUAL_REVIEW", {}).get(
                                        "status") == "PENDING" and
                                  act0r2_gates.get("READY_FOR_NEXT_SURFACE", {}).get(
                                        "status") == "CONDITIONAL_PASS" and
                                  act0r2_gates.get("READY_FOR_COUNTERFACTUAL_ROLLOUT", {}).get(
                                        "status") == "NOT_EVALUATED" and
                                  act0r2_gates.get("READY_FOR_JEPA", {}).get(
                                        "status") == "NOT_EVALUATED",
        "act0r2_public_assets": len(act0r2_assets) == 6 and
                                all(path.stat().st_size < 2_000_000 for path in act0r2_assets),
        "cf0_required_compact_files": all(path.exists() for path in (
                                        cf0_path, cf0_manifest_path, cf0_frame_path,
                                        cf0_branch_path, cf0_fold_path, cf0_action_path,
                                        Path("configs/experiments/cf0.yaml"),
                                        Path("docs/CF0_OBSERVABILITY_AUDIT.md"))) and
                                      cf0.get("schema") == "cf0.validation.v1" and
                                      cf0_manifest.get("schema") == "cf0.capture_manifest.v1",
        "cf0_capture_and_raw_audit": cf0.get("capture", {}).get("quartet_count") == 340 and
                                      cf0.get("capture", {}).get("shared_start_count") == 20 and
                                      cf0.get("capture", {}).get("branch_count") == 40 and
                                      cf0_manifest.get("saved_frame_count") == 340 and
                                      len(cf0_manifest.get("starts", [])) == 20 and
                                      len(cf0_manifest.get("frames", [])) == 340 and
                                      cf0.get("raw", {}).get("uploaded") is False and
                                      cf0.get("raw", {}).get("hash_audit", {}).get(
                                          "status") == "PASS" and
                                      cf0.get("raw", {}).get("hash_audit", {}).get(
                                          "checked_file_count") == 2040,
        "cf0_csv_consistency": cf0_csv_counts == {"frames": 360, "branches": 40,
                                                   "folds": 5, "actions": 80},
        "cf0_event_definition": cf0.get("event_definition") == {
                                      "minimum_span_over_target_bbox": 0.8,
                                      "minimum_boundary_penetration_px": 16.0,
                                      "minimum_target_side_fraction": 0.8,
                                      "minimum_external_side_fraction": 0.8},
        "cf0_pairing_start_and_coverage": all(cf0_gates.get(name, {}).get(
                                                "status") == "PASS" for name in (
                                                "COUNTERFACTUAL_PAIRING",
                                                "START_BOUNDARY_ABSENT",
                                                "ROBUST_EVENT_COVERAGE")) and
                                          cf0_gates.get("ROBUST_EVENT_COVERAGE", {}).get(
                                              "positive_branches") == 13 and
                                          cf0_gates.get("ROBUST_EVENT_COVERAGE", {}).get(
                                              "negative_branches") == 27,
        "cf0_group_split_and_train_only_processing": cf0_gates.get(
                                                "SPLIT_LEAKAGE_AUDIT", {}).get(
                                                "status") == "PASS" and
                                          cf0_gates.get("SPLIT_LEAKAGE_AUDIT", {}).get(
                                              "left_right_same_fold") is True and
                                          cf0_gates.get("SPLIT_LEAKAGE_AUDIT", {}).get(
                                              "forbidden_features_present") == [] and
                                          all(row.get("preprocessing", {}).get(
                                              "classification", row.get(
                                              "preprocessing", {})).get(
                                              "preprocessing_fit_scope") ==
                                              "training_fold_only"
                                              for row in cf0.get("evaluation", {}).get(
                                                  "baselines", {}).get("B3", {}).get(
                                                  "folds", [])),
        "cf0_preregistered_signal_failure": cf0_gates.get(
                                                "VISUAL_INCREMENTAL_VALUE", {}).get(
                                                "status") == "FAIL" and
                                           cf0_gates.get(
                                                "ACTION_SELECTION_SIGNAL", {}).get(
                                                "status") == "FAIL" and
                                           cf0_gates.get("SINGLE_SURFACE_SIGNAL", {}).get(
                                                "status") == "FAIL" and
                                           cf0_gates.get(
                                                "READY_FOR_MULTI_SURFACE_CAPTURE", {}).get(
                                                "status") == "FAIL",
        "cf0_terminal_scope": cf0_gates.get("CROSS_SURFACE_GENERALIZATION", {}).get(
                                                "status") == "NOT_EVALUATED" and
                               cf0_gates.get("READY_FOR_JEPA", {}).get(
                                                "status") == "NOT_EVALUATED" and
                               cf0.get("constraints", {}).get("rollout") is False and
                               cf0.get("constraints", {}).get("jepa_training") is False and
                               cf0.get("constraints", {}).get(
                                   "other_facades_captured") is False,
        "cf0_bootstrap_and_public_assets": all(
                               row.get("bootstrap_95_ci", {}).get(
                                   "balanced_accuracy", {}).get(
                                   "bootstrap_samples") == 1000
                               for row in cf0.get("evaluation", {}).get(
                                   "baselines", {}).values()) and
                               all(row.get("bootstrap_95_ci", {}).get(
                                   "accuracy", {}).get("bootstrap_samples") == 1000
                               for row in cf0.get("evaluation", {}).get(
                                   "action_selection", {}).values()) and
                               len(cf0_assets) == 4 and
                               all(path.stat().st_size < 2_000_000 for path in cf0_assets),
        "cf0_resource_limits": cf0.get("resources", {}).get(
                                    "configured_python_address_space_limit_bytes") == 4294967296 and
                               cf0.get("resources", {}).get(
                                    "actual_python_address_space_limit_bytes") == 4294967296 and
                               cf0.get("resources", {}).get(
                                    "configured_carla_address_space_limit_bytes") == 34359738368 and
                               cf0.get("resources", {}).get("numeric_threads") == 1 and
                               cf0.get("resources", {}).get("rss_watchdog_exceeded") is False,
        "probe0_required_compact_files": all(path.exists() for path in (
            probe0_path, probe0_source_path, probe0_predictions_path, probe0_config_path,
            Path("docs/PROBE0_ACTIVE_DISAMBIGUATION_AUDIT.md"))),
        "probe0_schema_and_frozen_protocol": probe0.get("schema") == "probe0.validation.v1" and
            probe0_source.get("schema") == "probe0.source_manifest.v1" and
            probe0.get("primary_probe_distance_m") == 1.0 and
            probe0.get("diagnostic_probe_distance_m") == 0.5 and
            probe0_config.get("probe", {}).get("forbidden_distance_m") == 2.0 and
            probe0.get("diagnostic_0_5m", {}).get("selection_role") ==
                "diagnostic_only_not_gated" and
            probe0.get("preregistered_thresholds") == probe0_thresholds,
        "probe0_frozen_p0_and_sample_counts": abs(float(probe0_p0 or -1) -
            0.5384615384615384) < 1e-12 and
            probe0_source.get("selected_start_count") == 13 and
            probe0_source.get("primary_sample_count") == 26 and
            probe0_pooled.get("sample_count") == 26 and
            probe0_left.get("sample_count") == 13 and probe0_right.get("sample_count") == 13,
        "probe0_group_split_and_prediction_integrity": probe0_fold_integrity and
            probe0_prediction_integrity,
        "probe0_preboundary_inputs_from_cf0_manifest": probe0_preboundary_inputs and
            probe0_gates.get("PREBOUNDARY_INPUT_AUDIT", {}).get(
                "boundary_or_postboundary_input_count") == 0,
        "probe0_raw_hash_and_evidence_audit": probe0_source.get("evidence_frame_count") == 65 and
            probe0_source.get("evidence_payload_size_bytes", 0) > 0 and
            probe0_source.get("hash_audit", {}).get("status") == "PASS" and
            probe0_source.get("hash_audit", {}).get("checked_file_count") == 390 and
            not probe0_source.get("hash_audit", {}).get("missing") and
            not probe0_source.get("hash_audit", {}).get("mismatches"),
        "probe0_forbidden_features_absent": probe0_source.get("model_feature_keys") == [
            "direction", "relative_distance_m", "relative_delta_m", "descriptor",
            "previous_descriptor", "history_valid"] and
            set(probe0_source.get("forbidden_feature_keys", [])) == {
                "offset", "absolute_coordinates", "world_boundary", "frame_id",
                "planned_role"},
        "probe0_preregistered_gates_recomputed": probe0_gates.get(
            "ACTIVE_DISAMBIGUATION_SIGNAL", {}).get("conditions") ==
                probe0_expected_conditions and all(probe0_expected_conditions.values()) and
            probe0_gates.get("ACTIVE_DISAMBIGUATION_SIGNAL", {}).get("status") == "PASS" and
            probe0_gates.get("READY_FOR_SECOND_SURFACE_REPLICATION", {}).get(
                "status") == "CONDITIONAL_PASS" and
            probe0_gates.get("READY_FOR_JEPA", {}).get("status") == "NOT_EVALUATED",
        "probe0_public_assets": len(probe0_assets) == 4 and
            {path.name for path in probe0_assets} == {
                "probe_1m_left_right.jpg", "near_away_rgb_change.jpg",
                "fold_predictions.jpg", "accuracy_ci.jpg"} and
            all(0 < path.stat().st_size < 2_000_000 for path in probe0_assets),
        "probe0_resource_limits": probe0.get("resources", {}).get(
            "address_space_limit_bytes") == 4294967296 and
            probe0.get("resources", {}).get("numeric_threads") == 1 and
            probe0.get("resources", {}).get("unique_input_rgb_frame_count") == 65 and
            probe0.get("resources", {}).get("maximum_feature_matrix_shape") == [26, 133] and
            probe0.get("resources", {}).get("model_artifacts_saved") is False,
        "probe0r1_required_compact_files": all(path.exists() for path in (
            probe0r1_path, probe0r1_bootstrap_path, probe0r1_ablation_path,
            probe0r1_config_path, Path("docs/PROBE0R1_CAUSAL_ATTRIBUTION_AUDIT.md"))),
        "probe0r1_schema_and_same_endpoint": probe0r1.get("schema") ==
            "probe0.validation_r1.v1" and
            probe0r1_bootstrap.get("schema") == "probe0r1.bootstrap.v1" and
            probe0r1.get("strict_same_endpoint") == {
                "status": "PASS", "distance_m": 1.0, "previous_distance_m": 0.5,
                "sample_count": 26, "start_count": 13},
        "probe0r1_historical_results_unchanged": probe0r1.get(
            "historical_probe0_files_modified") is False and
            probe0r1_historical_hashes_match,
        "probe0r1_source_and_raw_hashes": all(
            row.get("match") is True for row in probe0r1.get(
                "source_hash_checks", {}).values()) and
            probe0r1.get("raw_hash_audit", {}).get("status") == "PASS" and
            probe0r1.get("raw_hash_audit", {}).get("checked_file_count") == 390 and
            not probe0r1.get("raw_hash_audit", {}).get("missing") and
            not probe0r1.get("raw_hash_audit", {}).get("mismatches"),
        "probe0r1_ablation_matrix_and_csv": set(probe0r1_metrics) == {
            "E0", "E1", "E2", "E3_SWAP", "E4_ZERO"} and
            probe0r1.get("feature_matrix_shapes") == {
                "E0": [26, 4], "E1": [26, 66], "E2": [26, 133],
                "E3_SWAP": [26, 133], "E4_ZERO": [26, 133]} and
            len(probe0r1_ablation_rows) == 15 and
            {(row["ablation"], row["scope"]) for row in probe0r1_ablation_rows} == {
                (ablation, scope) for ablation in
                ("E0", "E1", "E2", "E3_SWAP", "E4_ZERO") for scope in
                ("pooled", "LEFT_probe", "RIGHT_probe")},
        "probe0r1_frozen_pipeline_and_fold_reuse": probe0r1_gates.get(
            "FROZEN_MODEL_PIPELINE", {}).get("status") == "PASS" and
            probe0r1_gates.get("FOLD_REUSE", {}).get("status") == "PASS" and
            probe0r1_gates.get("FOLD_REUSE", {}).get(
                "historical_probe0_fold_match") is True and
            probe0r1.get("fold_audit", {}).get("all_ablation_folds_match") is True and
            probe0r1.get("fold_audit", {}).get("historical_probe0_folds_match") is True,
        "probe0r1_heldout_intervention_protocol": probe0r1.get(
            "probability_diagnostics", {}).get(
                "E3_E4_fit_on_ordered_E2_train_features") is True and
            probe0r1.get("probability_diagnostics", {}).get(
                "intervention_scope") == "held-out test samples only" and
            set(probe0r1_config.get("experiment_design", {}).get(
                "intervention_protocol", {})) == {"E3_SWAP", "E4_ZERO"},
        "probe0r1_e2_exact_reproduction": probe0r1_gates.get(
            "E2_REPRODUCES_PROBE0", {}).get("status") == "PASS" and
            probe0r1_gates.get("E2_REPRODUCES_PROBE0", {}).get(
                "maximum_probability_error", 1.0) <= 1e-12 and
            probe0r1_metrics.get("E2", {}).get("pooled", {}).get("accuracy") ==
                probe0.get("primary_1m", {}).get("pooled", {}).get("accuracy"),
        "probe0r1_preregistered_gate_recomputed": probe0r1_gates.get(
            "TEMPORAL_HISTORY_INCREMENTAL_VALUE", {}).get("conditions") ==
                probe0r1_expected_conditions and
            not all(probe0r1_expected_conditions.values()) and
            probe0r1_gates.get("TEMPORAL_HISTORY_INCREMENTAL_VALUE", {}).get(
                "status") == "FAIL",
        "probe0r1_terminal_no_go": probe0r1_gates.get("ACTIVE_JEPA_ROUTE", {}).get(
            "status") == "NO_GO" and
            probe0r1_gates.get("READY_FOR_SECOND_SURFACE_REPLICATION", {}).get(
                "status") == "FAIL" and
            probe0r1_gates.get("READY_FOR_JEPA", {}).get("status") == "NOT_EVALUATED",
        "probe0r1_paired_bootstrap": probe0r1_bootstrap.get("samples") == 1000 and
            probe0r1_bootstrap.get("cluster_unit") == "start_id" and
            all(row.get("bootstrap_95_ci", {}).get("accuracy", {}).get(
                "bootstrap_samples") == 1000 for row in probe0r1_differences.values()),
        "probe0r1_public_assets": len(probe0r1_assets) == 4 and
            {path.name for path in probe0r1_assets} == {
                "e0_e4_overview.jpg", "e2_e1_paired_starts.jpg",
                "temporal_destruction.jpg", "bootstrap_accuracy_differences.jpg"} and
            all(0 < path.stat().st_size < 2_000_000 for path in probe0r1_assets),
        "probe0r1_resource_limits": probe0r1.get("resources", {}).get(
            "address_space_limit_bytes") == 4294967296 and
            probe0r1.get("resources", {}).get("numeric_threads") == 1 and
            probe0r1.get("resources", {}).get("largest_feature_matrix_shape") == [26, 133] and
            probe0r1.get("resources", {}).get("model_artifacts_saved") is False,
        "cap0_checkpoint_unchanged": hashlib.sha256(
                                      act0r_search_path.read_bytes()).hexdigest() ==
                                      "a56310883bb15513ea25c97c919d7faf14edb217b1a05fb0c4e12b060c664f73",
    }
    result = {"schema": "boundary_sweep.static_validation.v12", "checks": checks, "missing": missing, "private_path_files": forbidden,
              "geometry_reference_count": reference.get("geometry_reference_count"), "depth_metric": depth_result,
              "surface_stats_consistent": surface_checks, "gates": gates,
              "mask1r1_gates": mask1_r1.get("gates", {}), "obs0_gates": obs0.get("gates", {}),
              "obs0r1_gates": obs0r1_gates, "obs0r1_runtime_safety": runtime_safety,
              "act0s_gates": act0s_gates, "act0r_gates": act0r_gates,
              "cap0_gates": cap0_gates, "act0r1_gates": act0r1_gates,
              "act0r1_offline_gates": act0r1_offline_gates,
              "act0r2_gates": act0r2_gates, "cf0_gates": cf0_gates,
              "probe0_gates": probe0_gates, "probe0r1_gates": probe0r1_gates}
    print(json.dumps(result, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
