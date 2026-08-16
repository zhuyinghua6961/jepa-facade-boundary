#!/usr/bin/env python3
"""Validate the compact public GEO-0.5R2 result set and its gate semantics."""
from __future__ import annotations

import argparse
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
    r1 = json.loads(r1_path.read_text()) if r1_path.exists() else {}
    mask1 = json.loads(mask1_path.read_text()) if mask1_path.exists() else {}
    mask1_r1 = json.loads(mask1_r1_path.read_text()) if mask1_r1_path.exists() else {}
    event_r1 = json.loads(event_r1_path.read_text()) if event_r1_path.exists() else {}
    repeatability_r1 = json.loads(repeatability_r1_path.read_text()) if repeatability_r1_path.exists() else {}
    obs0 = json.loads(obs0_path.read_text()) if obs0_path.exists() else {}
    probes = json.loads(probe_path.read_text()) if probe_path.exists() else {}
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
    }
    result = {"schema": "boundary_sweep.static_validation.v3", "checks": checks, "missing": missing, "private_path_files": forbidden,
              "geometry_reference_count": reference.get("geometry_reference_count"), "depth_metric": depth_result,
              "surface_stats_consistent": surface_checks, "gates": gates,
              "mask1r1_gates": mask1_r1.get("gates", {}), "obs0_gates": obs0.get("gates", {})}
    print(json.dumps(result, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
