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
    }
    result = {"schema": "boundary_sweep.static_validation.v2", "checks": checks, "missing": missing, "private_path_files": forbidden,
              "geometry_reference_count": reference.get("geometry_reference_count"), "depth_metric": depth_result,
              "surface_stats_consistent": surface_checks, "gates": gates}
    print(json.dumps(result, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
