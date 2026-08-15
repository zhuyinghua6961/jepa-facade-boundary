import copy

from scripts.validate_geo05r2 import depth_gate, reprojection_stats


def valid_depth():
    return {
        "sample_count": 15,
        "z_depth_pass": True,
        "z_depth_median_abs_error_m": 0.0017,
        "z_depth_median_relative_error": 0.00017,
        "ray_range_median_abs_error_m": 0.45,
    }


def test_reprojection_ignores_aggregate_fields():
    errors = {"per_view": {"view_0": {"TL": 1.0, "TR": 3.0}}, "median": 99.0, "max": 100.0}
    result = reprojection_stats(errors)
    assert result["sample_count"] == 2
    assert result["median_px"] == 2.0
    assert result["max_px"] == 3.0


def test_depth_gate_passes_recorded_z_depth():
    result = depth_gate(valid_depth(), "depth.json")
    assert result["status"] == "PASS"


def test_depth_gate_fails_when_file_evidence_is_missing():
    result = depth_gate({}, "missing.json")
    assert result["status"] == "FAIL"


def test_depth_gate_fails_over_threshold():
    depth = valid_depth()
    depth["z_depth_median_abs_error_m"] = 0.2
    depth["z_depth_median_relative_error"] = 0.02
    assert depth_gate(depth, "depth.json")["status"] == "FAIL"


def test_depth_gate_fails_explicit_z_depth_failure():
    depth = valid_depth()
    depth["z_depth_pass"] = False
    assert depth_gate(depth, "depth.json")["status"] == "FAIL"


def test_depth_gate_fails_when_ray_range_is_better():
    depth = valid_depth()
    depth["ray_range_median_abs_error_m"] = 0.0001
    assert depth_gate(depth, "depth.json")["status"] == "FAIL"
