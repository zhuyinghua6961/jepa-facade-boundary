import numpy as np

from boundary_sweep.labels import adaptive_instance_boundary_evidence, classify_boundary


def _dense(coverage, projected, occ=1.0):
    return {"target_pixel_coverage": coverage, "projected_target_pixels": projected,
            "occlusion_visibility_ratio": occ}


def _probes(target=True, external_observed=True, external_not_target=False):
    return {"target": {"target_side_observed": target},
            "external": {"external_side_observed": external_observed,
                         "external_side_is_not_target": external_not_target}}


def test_continuous_wall_artificial_rectangle_is_not_straddle_or_out():
    boundary = np.asarray([[320.0, 100.0], [320.0, 380.0]])
    result = classify_boundary("ignored", "LEFT", _dense(0.50, 100), boundary, _probes(True, True, False), 640, 480, True)
    assert result["label"] == "UNKNOWN"


def test_real_termination_boundary_is_straddle():
    boundary = np.asarray([[320.0, 100.0], [320.0, 380.0]])
    result = classify_boundary("ignored", "LEFT", _dense(0.50, 100), boundary, _probes(True, True, True), 640, 480, True)
    assert result["label"] == "STRADDLE"


def test_target_fully_exited_is_out_only_with_exit_evidence():
    boundary = np.asarray([[-100.0, 100.0], [-100.0, 380.0]])
    result = classify_boundary("ignored", "LEFT", _dense(0.0, 0), boundary, _probes(False, False, False), 640, 480, False)
    assert result["label"] == "OUT"


def test_foreground_occlusion_is_unknown_not_out():
    boundary = np.asarray([[320.0, 100.0], [320.0, 380.0]])
    result = classify_boundary("ignored", "LEFT", _dense(0.0, 100, 0.0), boundary, _probes(False, True, True), 640, 480, True)
    assert result["label"] == "UNKNOWN"


def test_left_right_active_boundary_mapping():
    boundary = np.asarray([[320.0, 100.0], [320.0, 380.0]])
    for active in ("LEFT", "RIGHT"):
        result = classify_boundary("ignored", active, _dense(1.0, 100), boundary, _probes(True, True, True), 640, 480, True)
        assert result["label"] == "STRADDLE"


def test_sparse_zero_over_zero_is_not_implicit_out_when_not_confirmed():
    boundary = np.asarray([[320.0, 100.0], [320.0, 380.0]])
    result = classify_boundary("ignored", "LEFT", _dense(0.0, 0), boundary, _probes(False, False, False), 640, 480, False)
    # The line is still in the image, so no exit evidence exists.
    assert result["label"] == "UNKNOWN"


def test_adaptive_contour_requires_directional_external_side():
    left = np.zeros((40, 60), dtype=bool); left[:, 15:] = True
    right = np.zeros((40, 60), dtype=bool); right[:, :45] = True
    assert adaptive_instance_boundary_evidence(left, "LEFT")["label"] == "STRADDLE"
    assert adaptive_instance_boundary_evidence(right, "RIGHT")["label"] == "STRADDLE"


def test_adaptive_continuous_wall_has_no_straddle_contour():
    full = np.ones((40, 60), dtype=bool)
    assert adaptive_instance_boundary_evidence(full, "LEFT")["label"] == "IN"
