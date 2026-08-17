import numpy as np
import json
from pathlib import Path

from boundary_sweep.act0 import (backproject_mask_points, directional_external_coverage,
                                 directional_instance_contour,
                                 fit_target_plane, fit_terminal_boundaries, fixed_schedule_audit,
                                 paired_surface_folds, preferred_action,
                                 scout_sensor_pairing, start_match_metrics,
                                 tier_metric_repeatability, tier_physical_plane,
                                 tier_visual_event, trajectory_outcome)


def test_backproject_mask_points_uses_z_depth_and_stride():
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 0] = True
    mask[0, 2] = True
    mask[2, 0] = True
    mask[2, 2] = True
    depth = np.full((5, 5), 10.0, dtype=np.float32)
    K = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    points = backproject_mask_points(mask, depth, K, np.eye(4), pixel_step=2)
    assert points.shape == (4, 3)
    # CV camera +Z maps to CARLA UE world +X for an identity transform.
    assert np.allclose(points[:, 0], 10.0)


def test_terminal_boundaries_fit_independent_view_lines():
    left = [np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            np.array([[0.02, 0.0, 0.0], [0.02, 0.0, 2.0]])]
    right = [np.array([[8.0, 0.0, 0.0], [8.0, 0.0, 2.0]]),
             np.array([[8.01, 0.0, 0.0], [8.01, 0.0, 2.0]])]
    fitted = fit_terminal_boundaries({"LEFT": left, "RIGHT": right},
                                     [0, 0, 0], [1, 0, 0], [0, 0, 1])
    assert fitted["LEFT"]["view_count"] == 2
    assert fitted["RIGHT"]["view_count"] == 2
    assert abs(fitted["LEFT"]["horizontal_coordinate_m"] - 0.01) < 1e-9
    assert abs(fitted["RIGHT"]["horizontal_coordinate_m"] - 8.005) < 1e-9


def test_target_plane_rejects_side_wall_and_far_depth_outliers():
    yy, zz = np.meshgrid(np.linspace(-3, 3, 12), np.linspace(0, 6, 12))
    facade = np.column_stack([np.full(yy.size, 10.0), yy.ravel(), zz.ravel()])
    outliers = np.column_stack([np.linspace(12, 25, 50),
                                np.linspace(-2, 2, 50), np.linspace(0, 5, 50)])
    fitted = fit_target_plane(np.row_stack([facade, outliers]), [1, 0, 0])
    assert fitted["normal_error_deg"] < 1.0
    assert fitted["p95_residual_m"] < 0.01
    assert fitted["inlier_ratio"] > 0.70


def test_directional_external_coverage_tracks_command_side():
    mask = np.zeros((4, 10), dtype=bool)
    mask[:, 2:8] = True
    assert abs(directional_external_coverage(mask, "LEFT") - 0.2) < 1e-12
    assert abs(directional_external_coverage(mask, "RIGHT") - 0.2) < 1e-12


def test_closed_instance_outline_produces_distinct_left_and_right_lines():
    mask = np.zeros((20, 30), dtype=bool)
    mask[3:17, 7:24] = True
    left = directional_instance_contour(mask, "LEFT")
    right = directional_instance_contour(mask, "RIGHT")
    left_x = np.where(left["contour"])[1]
    right_x = np.where(right["contour"])[1]
    assert left["contour_present"] and right["contour_present"]
    assert np.median(left_x) == 7
    assert np.median(right_x) == 23


def test_counterfactual_start_match_is_pose_and_rgb_based():
    image = np.full((12, 16, 3), 80, dtype=np.uint8)
    metrics = start_match_metrics(image, image.copy(), np.eye(4), np.eye(4))
    assert metrics["position_error_m"] == 0.0
    assert metrics["rotation_error_deg"] == 0.0
    assert metrics["initial_rgb_ssim"] == 1.0
    assert metrics["initial_rgb_mean_absolute_pixel_error"] == 0.0


def test_outcome_and_preferred_action_preserve_censoring():
    left_frames = [
        {"state": "IN", "capture_role": "moving", "step_index": 0, "actual_displacement_m": 0.0},
        {"state": "STRADDLE", "capture_role": "moving", "step_index": 3, "actual_displacement_m": 0.7},
        {"state": "STRADDLE", "capture_role": "frozen_confirmation", "step_index": 3, "actual_displacement_m": 0.7},
    ]
    right_frames = [{"state": "IN", "capture_role": "moving", "step_index": 0,
                     "actual_displacement_m": 0.0}]
    left = trajectory_outcome(left_frames, 5.5)
    right = trajectory_outcome(right_frames, 5.5)
    assert left["distance_to_event_m"] == 0.7
    assert right["censored"] is True
    assert preferred_action(left, right) == "LEFT"


def test_fixed_schedule_requires_multiple_steps_distances_and_step_sizes():
    rows = [
        {"event_within_horizon": True, "time_to_event_steps": 3, "distance_to_event_m": 0.6, "step_m": 0.2},
        {"event_within_horizon": True, "time_to_event_steps": 7, "distance_to_event_m": 2.45, "step_m": 0.35},
        {"event_within_horizon": True, "time_to_event_steps": 9, "distance_to_event_m": 4.5, "step_m": 0.5},
    ]
    assert fixed_schedule_audit(rows)["status"] == "PASS"
    assert fixed_schedule_audit([rows[0], rows[0]])["status"] == "FAIL"


def test_leave_one_surface_out_keeps_each_surface_whole():
    rows = [{"surface_id": "a"}, {"surface_id": "a"},
            {"surface_id": "b"}, {"surface_id": "b"}]
    folds = paired_surface_folds(rows)
    assert len(folds) == 2
    for fold in folds:
        train_surfaces = {rows[i]["surface_id"] for i in fold["train_indices"]}
        test_surfaces = {rows[i]["surface_id"] for i in fold["test_indices"]}
        assert train_surfaces.isdisjoint(test_surfaces)


def test_act0_stops_honestly_when_facade_screening_fails_if_available():
    path = Path("results/act0/validation.json")
    if not path.exists():
        return
    data = json.loads(path.read_text())
    assert data["status"] == "STOPPED_AT_FACADE_SCREENING"
    assert data["gates"]["VALID_EXTERNAL_BOUNDARY"]["status"] == "FAIL"
    assert data["gates"]["COUNTERFACTUAL_EVENT_COVERAGE"]["status"] == "FAIL"
    assert data["gates"]["READY_FOR_JEPA"]["status"] == "NOT_EVALUATED"
    assert data["raw_data"]["rollout_raw_bytes"] == 0


def _screening_candidate():
    audit = {"contour_present": True, "contour_span_fraction": 0.4,
             "target_side_fraction": 1.0, "external_side_fraction": 1.0}
    view = {"view_index": 0, "frame_id": 10,
            "sensor_frames": {name: 10 for name in ("rgb", "depth", "semantic", "instance")},
            "sensor_timestamps": {name: 1.0 for name in ("rgb", "depth", "semantic", "instance")}}
    return {
        "target_instance_id_16bit": 42, "target_instance_id_stable": True,
        "center_target_coverage": 0.3, "rejection_reasons": [],
        "views": [{**view, "view_index": index, "frame_id": 10 + index,
                   "sensor_frames": {name: 10 + index for name in ("rgb", "depth", "semantic", "instance")}}
                  for index in range(3)],
        "boundary_audits": {"LEFT": [audit] * 3, "RIGHT": [audit] * 3},
        "boundaries": {"LEFT": {"view_count": 3, "horizontal_std_m": 0.01,
                                  "horizontal_coordinate_m": -5.0},
                       "RIGHT": {"view_count": 3, "horizontal_std_m": 0.02,
                                   "horizontal_coordinate_m": 5.0}},
        "plane": {"p95_residual_m": 0.02, "normal_error_deg": 1.0,
                  "inlier_ratio": 0.8},
        "raycast_audit": {"agreement": 1.0},
    }


def test_not_observed_is_not_counted_as_scene_failure():
    candidate = _screening_candidate()
    candidate.update({"target_instance_id_16bit": None, "target_instance_id_stable": False,
                      "center_target_coverage": 0.0})
    thresholds = {"min_contour_views": 2, "min_target_side_fraction": 0.5,
                  "min_external_side_fraction": 0.5, "max_contour_span_range": 0.15}
    result = tier_visual_event(candidate, thresholds,
                               {"tier_v_status": "NOT_OBSERVED",
                                "classification": "SCOUT_POSE_INSUFFICIENT"})
    assert result["status"] == "NOT_OBSERVED"
    assert result["status"] != "FAIL"


def test_tier_p_failure_does_not_veto_tier_v():
    candidate = _screening_candidate()
    visual_thresholds = {"min_contour_views": 2, "min_target_side_fraction": 0.5,
                         "min_external_side_fraction": 0.5, "max_contour_span_range": 0.15}
    physical_thresholds = {"max_plane_p95_residual_m": 0.15, "max_normal_error_deg": 20.0,
                           "min_plane_inlier_ratio": 0.5, "min_raycast_agreement": 0.8,
                           "min_width_m": 6.0}
    candidate["raycast_audit"]["agreement"] = 0.0
    assert tier_visual_event(candidate, visual_thresholds)["status"] == "PASS"
    assert tier_physical_plane(candidate, physical_thresholds)["status"] == "FAIL"


def test_legacy_plane_basis_never_satisfies_tier_m():
    candidate = _screening_candidate()
    result = tier_metric_repeatability(candidate, [0.05, 0.10, 0.25, 1.0])
    assert result["status"] == "NOT_EVALUATED"
    assert result["legacy_proxy_sensitivity"]["0.05m"] is True
    assert result["official"] == {}
    assert result["uses_legacy_plane_or_bbox"] is False


def test_act0s_manifest_and_twelve_public_sheets_exist_if_available():
    path = Path("results/act0/screening_audit_v2.json")
    if not path.exists():
        return
    data = json.loads(path.read_text())
    assert data["historical_full_physical_gate"]["selected"] == 0
    assert len(data["candidates"]) == 12
    assert data["gates"]["OPERATOR_VISUAL_REVIEW"]["status"] == "PENDING"
    assert data["gates"]["READY_FOR_JEPA"]["status"] == "NOT_EVALUATED"
    assert len(list(Path("docs/assets/act0_screening").glob("candidate_*_screening.jpg"))) == 12
    assert Path("results/act0/candidate_gate_matrix_v2.csv").exists()
    assert Path("results/act0/scout_coverage_audit.json").exists()
