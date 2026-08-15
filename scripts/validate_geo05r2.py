#!/usr/bin/env python3
"""Validate GEO-0.5R2 gates from raw data or compact published results."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def flatten_numbers(value):
    """Flatten only raw numeric leaves, preserving the caller's selection."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)]
    if isinstance(value, dict):
        return [n for child in value.values() for n in flatten_numbers(child)]
    if isinstance(value, (list, tuple)):
        return [n for child in value for n in flatten_numbers(child)]
    return []


def reprojection_stats(reprojection_errors_px):
    """Compute statistics from per-view observations, never aggregate fields."""
    per_view = reprojection_errors_px.get("per_view", reprojection_errors_px)
    values = flatten_numbers(per_view)
    if not values:
        return {"median_px": None, "max_px": None, "sample_count": 0}
    return {"median_px": statistics.median(values), "max_px": max(values), "sample_count": len(values)}


def depth_gate(depth, evidence):
    """Evaluate recorded depth evidence; missing evidence fails closed."""
    required = ("sample_count", "z_depth_pass", "z_depth_median_abs_error_m")
    if any(key not in depth for key in required):
        return {"status": "FAIL", "metric": "z-depth", "evidence": evidence,
                "reason": "required depth evidence is missing"}
    sample_count = depth["sample_count"]
    abs_error = depth["z_depth_median_abs_error_m"]
    rel_error = depth.get("z_depth_median_relative_error")
    threshold_ok = (isinstance(abs_error, (int, float)) and abs_error < 0.1) or (isinstance(rel_error, (int, float)) and rel_error < 0.01)
    range_score = depth.get("ray_range_median_abs_error_m", depth.get("range_median_abs_error_m"))
    better = isinstance(range_score, (int, float)) and abs_error < range_score
    passed = isinstance(sample_count, int) and sample_count >= 15 and bool(depth["z_depth_pass"]) and threshold_ok and better
    result = {"status": "PASS" if passed else "FAIL", "metric": "z-depth", "evidence": evidence,
              "sample_count": sample_count, "z_depth_pass": bool(depth["z_depth_pass"]),
              "z_depth_median_abs_error_m": abs_error, "z_depth_median_relative_error": rel_error,
              "ray_range_median_abs_error_m": range_score}
    if not passed:
        result["reason"] = "sample count, z-depth threshold/pass flag, or z-depth-vs-ray-range comparison failed"
    return result


def compressed(states):
    out = []
    for state in states:
        if not out or out[-1] != state:
            out.append(state)
    return out


def monotone(states):
    filtered = [state for state in states if state != "UNKNOWN"]
    rank = {"IN": 0, "STRADDLE": 1, "OUT": 2}
    return all(rank[a] <= rank[b] for a, b in zip(filtered, filtered[1:])) and not any(a == "OUT" and b == "UNKNOWN" and c == "OUT" for a, b, c in zip(states, states[1:], states[2:]))


def _load_trajectories(root, compact_source):
    paths = sorted(root.glob("**/trajectory.json")) if root.exists() else []
    if paths:
        rows, frame_count, mono, complete, bad = [], 0, True, 0, []
        for path in paths:
            data = json.loads(path.read_text())
            frames = data.get("frames", [])
            states = [frame.get("labels", {}).get("label", "UNKNOWN") for frame in frames]
            frame_count += len(frames)
            ok = monotone(states)
            mono &= ok
            is_complete = all(state in states for state in ("IN", "STRADDLE", "OUT"))
            complete += int(is_complete)
            if not ok:
                bad.append({"trajectory": str(path), "sequence": compressed(states)})
            rows.append({"trajectory": str(path), "frames": len(frames), "state_counts": dict(Counter(states)), "compressed_state_sequence": compressed(states), "monotonic": ok, "complete_in_straddle_out": is_complete})
        return rows, frame_count, True, mono, complete, bad
    source = json.loads(compact_source.read_text())
    trajectories = source.get("trajectories", [])
    old_repro = source.get("gates", {}).get("REPRODUCIBILITY", {})
    old_order = source.get("gates", {}).get("TRAJECTORY_MONOTONICITY", {})
    rows = []
    for row in trajectories:
        states = row.get("compressed_state_sequence", [])
        rows.append({**row, "monotonic": monotone(states), "complete_in_straddle_out": all(s in states for s in ("IN", "STRADDLE", "OUT"))})
    return rows, int(old_repro.get("frame_count", sum(r.get("frames", 0) for r in rows))), bool(old_repro.get("rgb_depth_pairs", False)), bool(old_order.get("monotonic_all", all(r["monotonic"] for r in rows))), int(old_order.get("complete_transition_trajectories", sum(r["complete_in_straddle_out"] for r in rows))), old_order.get("bad_trajectories", [])


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/sweeps_geo05r2")
    parser.add_argument("--surfaces", default="results/geo05r2/surfaces_v3")
    parser.add_argument("--geometry-reference", default="results/geo05r2/geometry_reference_audit.json")
    parser.add_argument("--depth-metric", default="results/geo05r2/depth_metric_v2.json")
    parser.add_argument("--output", default="results/geo05r2/geo05r2_validation.json")
    parser.add_argument("--compact-source", default=None)
    args = parser.parse_args(argv)
    output = Path(args.output)
    compact_source = Path(args.compact_source) if args.compact_source else output
    rows, frame_count, pairs, mono, complete, bad = _load_trajectories(Path(args.root), compact_source)
    surface_rows, surface_errors = [], []
    for path in sorted(Path(args.surfaces).glob("*.json")):
        data = json.loads(path.read_text())
        stats = reprojection_stats(data.get("reprojection_errors_px", {}))
        surface_rows.append({"surface_id": data.get("surface_id", path.stem), "bbox_id": data.get("bbox_id"), **stats, "manual_confirmation_status": data.get("manual_confirmation_status") or data.get("source", {}).get("manual_confirmation_status")})
        surface_errors.extend(flatten_numbers(data.get("reprojection_errors_px", {}).get("per_view", {})))
    reference_path = Path(args.geometry_reference)
    reference = json.loads(reference_path.read_text()) if reference_path.exists() else {}
    source = json.loads(compact_source.read_text()) if compact_source.exists() else {}
    old_semantics = source.get("gates", {}).get("BOUNDARY_SEMANTICS", {})
    agreement = old_semantics.get("geometry_reference_agreement", old_semantics.get("accuracy", 0.0))
    depth_path = Path(args.depth_metric)
    depth = json.loads(depth_path.read_text()) if depth_path.exists() else {}
    depth_result = depth_gate(depth, str(depth_path))
    physical_pass = bool(surface_rows) and all(row["median_px"] <= 5 and row["max_px"] <= 10 and row["manual_confirmation_status"] for row in surface_rows)
    gates = {
        "REPRODUCIBILITY": {"status": "PASS" if pairs and frame_count == 960 else "FAIL", "frame_count": frame_count, "rgb_depth_pairs": pairs},
        "CAPTURE_PIPELINE": source.get("gates", {}).get("CAPTURE_PIPELINE", {"status": "PASS"}),
        "DEPTH_METRIC": depth_result,
        "PHYSICAL_BOUNDARY_GROUND_TRUTH": {"status": "PASS" if physical_pass else "FAIL", "surfaces": surface_rows},
        "GEOMETRY_REFERENCE_CONSISTENCY": {"status": "FAIL", "geometry_reference_count": reference.get("geometry_reference_count", len(reference.get("frames", []))), "geometry_reference_agreement": agreement, "operator_visual_review_completed": reference.get("operator_visual_review_completed", False)},
        "BOUNDARY_SEMANTICS": {"status": "FAIL", "operator_ground_truth_available": False, "geometry_reference_agreement": agreement, "complete_in_straddle_out_trajectories": complete},
        "TRAJECTORY_ORDERING": {"status": "PASS" if mono else "FAIL", "monotonic_all": mono, "bad_trajectories": bad},
        "EVENT_COVERAGE": {"status": "PASS" if complete >= 4 else "FAIL", "complete_in_straddle_out_trajectories": complete, "required_complete_trajectories": 4},
    }
    gates["READY_FOR_JEPA"] = {"status": "PASS" if all(gate.get("status") == "PASS" for gate in gates.values()) else "FAIL"}
    result = {"schema": "geo05r2.validation.v2", "gates": gates, "trajectories": rows, "surface_errors_px": surface_errors, "geometry_reference_audit": str(reference_path), "geometry_reference_agreement": agreement}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    transition_path = output.parent / "trajectory_transition_audit_r2.json"
    transition_path.write_text(json.dumps({"schema": "geo05r2.transition_audit.v2", "trajectories": [{"trajectory": row.get("trajectory"), "compressed_state_sequence": row.get("compressed_state_sequence", []), "monotonic": row.get("monotonic", False), "transition_count": max(0, len(row.get("compressed_state_sequence", [])) - 1)} for row in rows], "all_monotonic": mono}, indent=2) + "\n")
    print(json.dumps(gates, indent=2))
    return 0 if gates["READY_FOR_JEPA"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
