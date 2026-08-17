#!/usr/bin/env python3
"""Validate the compact public GEO-0.5R2 result set and its gate semantics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

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
    act0s_matrix_count = 0
    if act0s_matrix_path.exists():
        with act0s_matrix_path.open(newline="") as handle:
            act0s_matrix_count = sum(1 for _row in csv.DictReader(handle))
    act0s_assets = sorted(Path("docs/assets/act0_screening").glob("candidate_*_screening.jpg"))
    act0s_gates = act0s.get("gates", {})
    act0r_gates = act0r.get("gates", {})
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
    }
    result = {"schema": "boundary_sweep.static_validation.v6", "checks": checks, "missing": missing, "private_path_files": forbidden,
              "geometry_reference_count": reference.get("geometry_reference_count"), "depth_metric": depth_result,
              "surface_stats_consistent": surface_checks, "gates": gates,
              "mask1r1_gates": mask1_r1.get("gates", {}), "obs0_gates": obs0.get("gates", {}),
              "obs0r1_gates": obs0r1_gates, "obs0r1_runtime_safety": runtime_safety,
              "act0s_gates": act0s_gates, "act0r_gates": act0r_gates}
    print(json.dumps(result, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
