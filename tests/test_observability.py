import json
from pathlib import Path

import numpy as np

from boundary_sweep.geometry import intrinsics_from_fov
from boundary_sweep.observability import (action_axis_from_poses, backproject_contour,
                                          first_threshold_step, local_contour_evidence)


def test_global_threshold_reports_first_and_not_reached():
    assert first_threshold_step([0.0, 0.02, 0.031], 0.03) == 2
    assert first_threshold_step([0.0, 0.02], 0.03) is None


def test_local_contour_event_uses_real_external_termination():
    mask = np.zeros((80, 100), dtype=bool)
    mask[:, 20:80] = True
    evidence = local_contour_evidence(mask, "LEFT")
    assert evidence["local_straddle"] is True
    assert evidence["contour_present"] is True


def test_action_axis_follows_observed_translation():
    axis, metadata = action_axis_from_poses([[0, 0, 0], [0, -0.5, 0], [0, -1.0, 0]])
    assert np.allclose(axis, [0.0, -1.0, 0.0])
    assert metadata["fallback"] is False


def test_no_plane_backprojection_uses_z_depth_only():
    K = intrinsics_from_fov(3, 3, 90.0)
    contour = np.zeros((3, 3), dtype=bool)
    contour[1, 1] = True
    depth = np.full((3, 3), 5.0, dtype=np.float32)
    result = backproject_contour(contour, depth, K, np.eye(4), [0, 1, 0], depth_reference=depth[contour])
    assert result["valid_boundary_points"] == 1
    assert result["rejected_depth_points"] == 0


def test_r1_event_gate_reports_distinct_pose_and_overshoot_if_available():
    path = Path("results/mask1/validation_r1.json")
    if not path.exists():
        return
    data = json.loads(path.read_text())
    assert data["gates"]["SAME_POSE_CONFIRMATION"]["status"] == "NOT_EVALUATED"
    assert data["gates"]["STOP_OVERSHOOT"]["status"] == "FAIL"
    assert all(row["first_local_straddle_step"] == 10 for row in data["trajectories"])


def test_obs0_uses_complete_direction_holdout_if_available():
    path = Path("results/obs0/probe_results.json")
    if not path.exists():
        return
    data = json.loads(path.read_text())
    assert data["split"] == "complete direction holdout"
    assert set(data["probes"]) == {"P0", "P1", "P2", "P3", "P4", "P5", "P6"}
    for probe in data["probes"].values():
        assert set(probe["splits"]) == {"train_LEFT_test_RIGHT", "train_RIGHT_test_LEFT"}


def test_obs0_public_manifest_and_assets_if_available():
    validation = Path("results/obs0/validation.json")
    assets = Path("docs/assets/obs0")
    if not validation.exists():
        return
    data = json.loads(validation.read_text())
    assert data["gates"]["ACTION_SELECTION_OBSERVABILITY"]["status"] == "NOT_EVALUATED"
    assert data["gates"]["READY_FOR_JEPA"]["status"] == "NOT_EVALUATED"
    assert (assets / "pairwise_ssim_heatmap.jpg").exists()
    assert (assets / "alias_pair_05.jpg").exists()
