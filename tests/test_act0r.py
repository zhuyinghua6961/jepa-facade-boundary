import json
from pathlib import Path

import numpy as np
import yaml

from boundary_sweep.act0r import (action_axis_from_transforms,
                                  boundary_type_consensus,
                                  checkpoint_pose_alignment,
                                  classify_boundary_pixels,
                                  config_outcome_override_audit,
                                  contour_span_metrics,
                                  event_ordering_from_geometry,
                                  official_tier_m,
                                  physical_termination_pixel_gate,
                                  pose_repeatability, sha256_file,
                                  select_repeated_pose_group,
                                  tier_v_from_pixel_frames,
                                  verify_manifest_hashes)
from boundary_sweep.carla_utils import transform_dict_from_matrix


def _classification_thresholds():
    return {"side_probe_offset_px": 3, "building_semantic_tag": 3,
            "min_bilateral_samples": 10, "min_depth_pairs": 10,
            "depth_margin_m": 0.3, "same_depth_tolerance_m": 0.2,
            "occlusion_closer_fraction": 0.6, "termination_max_closer_fraction": 0.2,
            "semantic_majority_fraction": 0.7, "same_depth_majority_fraction": 0.7}


def _synthetic_boundary(external_depth, external_semantic, external_instance):
    target = np.zeros((80, 100), dtype=bool)
    target[10:70, 20:60] = True
    span = contour_span_metrics(target, "RIGHT")
    depth = np.full(target.shape, 10.0, dtype=np.float32)
    semantic = np.zeros(target.shape, dtype=np.uint8)
    semantic[target] = 3
    instance = np.zeros(target.shape, dtype=np.uint32)
    instance[target] = 100
    depth[:, 60:] = external_depth
    semantic[:, 60:] = external_semantic
    instance[:, 60:] = external_instance
    result = classify_boundary_pixels(target, span["contour"], depth, semantic, instance,
                                      100, "RIGHT", _classification_thresholds())
    return span, result


def test_config_cannot_override_computed_outcomes():
    assert config_outcome_override_audit({"search": {"coarse_step_m": 1.0},
                                          "boundary_classification": {"depth_margin_m": 0.3}})["status"] == "PASS"
    failed = config_outcome_override_audit({"screening": {"tier_v_status": "PASS"}})
    assert failed["status"] == "FAIL"
    assert failed["forbidden_paths"] == ["screening.tier_v_status"]


def test_checked_in_act0r_config_has_no_outcome_override():
    path = Path("configs/experiments/act0r.yaml")
    if not path.exists():
        return
    assert config_outcome_override_audit(yaml.safe_load(path.read_text()))["status"] == "PASS"


def test_contour_span_is_normalized_by_target_bbox_height():
    span, _result = _synthetic_boundary(20.0, 0, 0)
    assert span["target_bbox_height_px"] == 60
    assert span["span_over_target_bbox_height"] > 0.90
    assert span["span_over_target_bbox_height"] > span["span_over_image_height"]


def test_sparse_contour_endpoints_do_not_count_as_full_height():
    target = np.zeros((80, 100), dtype=bool)
    target[10:70, :] = True
    span = contour_span_metrics(target, "LEFT")
    assert span["target_bbox_height_px"] == 60
    assert span["contour_pixel_count"] == 2
    assert span["contour_span_px"] == 2
    assert span["span_over_target_bbox_height"] < 0.05


def test_pixel_boundary_types_cover_termination_occlusion_and_internal_seam():
    _span, termination = _synthetic_boundary(20.0, 0, 0)
    _span, occlusion = _synthetic_boundary(5.0, 0, 0)
    _span, seam = _synthetic_boundary(10.05, 3, 200)
    assert termination["boundary_type"] == "PHYSICAL_TERMINATION"
    assert occlusion["boundary_type"] == "FOREGROUND_OCCLUSION"
    assert seam["boundary_type"] == "INTERNAL_INSTANCE_SEAM"


def test_official_tier_m_is_plane_and_bbox_free():
    frames = [{"valid_contour_point_count": 50, "action_axis_median_m": value,
               "action_axis_mad_m": 0.01} for value in (1.00, 1.04, 0.98, 1.02)]
    result = official_tier_m(frames, [0.05, 0.10, 0.25, 1.0], 0.25)
    assert result["status"] == "PASS"
    assert abs(result["spread_m"] - 0.06) < 1e-12
    assert result["uses_plane"] is False
    assert result["uses_bbox"] is False
    assert result["uses_legacy_boundary"] is False


def test_three_frozen_repeats_have_identical_pose():
    transform = np.eye(4)
    result = pose_repeatability([transform.copy() for _ in range(4)], 0.01, 0.05)
    assert result["status"] == "PASS"
    moved = transform.copy()
    moved[0, 3] = 0.02
    assert pose_repeatability([transform, transform, transform, moved], 0.01, 0.05)["status"] == "FAIL"


def test_raw_manifest_hashes_are_verified(tmp_path):
    path = tmp_path / "frame_rgb.png"
    path.write_bytes(b"pixel evidence")
    entry = {"frame_id": 7, "files": {"rgb": {"path": path.name, "sha256": sha256_file(path)}}}
    assert verify_manifest_hashes([entry], tmp_path)["status"] == "PASS"
    path.write_bytes(b"changed")
    assert verify_manifest_hashes([entry], tmp_path)["status"] == "FAIL"


def test_boundary_type_requires_three_of_four_independent_frames():
    rows = [{"boundary_type": value} for value in
            ("PHYSICAL_TERMINATION", "PHYSICAL_TERMINATION",
             "PHYSICAL_TERMINATION", "FOREGROUND_OCCLUSION")]
    result = boundary_type_consensus(rows, 3)
    assert result["status"] == "PASS"
    assert result["boundary_type"] == "PHYSICAL_TERMINATION"
    tied = boundary_type_consensus(rows[:2] + [
        {"boundary_type": "FOREGROUND_OCCLUSION"},
        {"boundary_type": "FOREGROUND_OCCLUSION"}], 3)
    assert tied["status"] == "FAIL"
    assert tied["boundary_type"] == "UNRESOLVED"


def test_repeated_pose_group_is_selected_without_role_labels():
    transforms = []
    for x in (0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0):
        matrix = np.eye(4)
        matrix[0, 3] = x
        transforms.append(matrix)
    result = select_repeated_pose_group(transforms, 4, 0.01, 0.05)
    assert result["status"] == "PASS"
    assert result["indices"] == [2, 3, 4, 5]
    assert result["uses_role_labels"] is False


def test_actual_action_axis_comes_from_camera_transforms():
    transforms = []
    for x in (0.0, -1.0, -2.0, -2.0, -3.0):
        matrix = np.eye(4)
        matrix[:3, 3] = [x, 4.0, 7.0]
        transforms.append(matrix)
    result = action_axis_from_transforms(transforms)
    assert result["status"] == "PASS"
    assert np.allclose(result["axis"], [-1.0, 0.0, 0.0])
    assert result["max_orthogonal_residual_m"] < 1e-12
    assert result["uses_commanded_action"] is False


def test_tier_v_uses_repeated_pixel_metrics_only():
    rows = [{
        "contour_present": True,
        "span_over_target_bbox_height": 0.85,
        "target_side_fraction": 1.0,
        "external_side_fraction": 1.0,
    } for _ in range(4)]
    thresholds = {
        "tier_v_min_span_over_target_bbox": 0.8,
        "min_target_side_fraction": 0.8,
        "min_external_side_fraction": 0.8,
    }
    result = tier_v_from_pixel_frames(rows, thresholds, 3)
    assert result["status"] == "PASS"
    assert result["pass_count"] == 4
    assert result["uses_role_labels"] is False
    rows[0]["span_over_target_bbox_height"] = 0.7
    rows[1]["span_over_target_bbox_height"] = 0.7
    assert tier_v_from_pixel_frames(rows, thresholds, 3)["status"] == "FAIL"


def test_published_incomplete_capture_fails_closed_and_has_audit_assets():
    validation_path = Path("results/act0r/validation.json")
    search_path = Path("results/act0r/search_plan_checkpoint.json")
    assert validation_path.exists()
    assert search_path.exists()
    validation = json.loads(validation_path.read_text())
    search = json.loads(search_path.read_text())
    gates = validation["gates"]
    assert validation["run_status"] == "INCOMPLETE_CAPTURE"
    assert validation["saved_frame_count"] == 1
    assert gates["RAW_PIXEL_EVIDENCE_AVAILABLE"]["status"] == "FAIL"
    assert gates["TIER_V_RECOMPUTED_FROM_PIXELS"]["recomputed_sides"] == 0
    assert gates["OFFICIAL_TIER_M"]["evaluated_sides"] == 0
    assert gates["READY_FOR_COUNTERFACTUAL_ROLLOUT"]["status"] == "FAIL"
    assert gates["READY_FOR_JEPA"]["status"] == "NOT_EVALUATED"
    assert search["complete"] is True
    assert [row["candidate_index"] for row in search["plans"]] == [1, 7, 10, 19]
    assert len(list(Path("docs/assets/act0r").glob("*.jpg"))) == 7


def test_act0r1_offline_audit_is_pixel_derived_if_available():
    path = Path("results/act0r1/offline_boundary_audit.json")
    if not path.exists():
        return
    result = json.loads(path.read_text())
    gates = result["gates"]
    assert result["schema"] == "act0r1.offline_boundary_audit.v1"
    assert len(result["frame_metrics"]) == 8
    assert result["role_label_policy"]["capture_roles_used_as_ground_truth"] is False
    assert result["constraints"]["legacy_plane_used"] is False
    assert result["constraints"]["legacy_bbox_used"] is False
    assert gates["RAW_HASH_AUDIT"]["status"] == "PASS"
    assert gates["SENSOR_PAIRING"]["status"] == "PASS"
    assert gates["EXTERNAL_VISUAL_REVIEW"]["status"] == "PENDING"
    assert gates["READY_FOR_COUNTERFACTUAL_ROLLOUT"]["status"] == "NOT_EVALUATED"
    assert gates["READY_FOR_JEPA"]["status"] == "NOT_EVALUATED"
    assert len(list(Path("docs/assets/act0r1").glob("offline_*.jpg"))) == 6


def test_checkpoint_matrix_conversion_and_pose_alignment():
    matrix = np.array([
        [0.0, 1.0, 0.0, -92.99478912353516],
        [-1.0, 0.0, 0.0, 288.3297424316406],
        [0.0, 0.0, 1.0, 12.002955436706543],
        [0.0, 0.0, 0.0, 1.0],
    ])
    pose = transform_dict_from_matrix(matrix)
    assert pose["location"]["y"] == 288.3297424316406
    assert abs(pose["rotation"]["yaw"] + 90.0) < 1e-9
    assert checkpoint_pose_alignment(matrix, matrix, 0.01, 0.05)["status"] == "PASS"
    moved = matrix.copy()
    moved[0, 3] += 0.011
    assert checkpoint_pose_alignment(moved, matrix, 0.01, 0.05)["status"] == "FAIL"


def _event_metric(termination=False):
    return {
        "contour_present": termination,
        "span_over_target_bbox_height": 1.0 if termination else 0.0,
        "target_side_fraction": 1.0 if termination else 0.0,
        "external_side_fraction": 1.0 if termination else 0.0,
        "boundary_classification": {
            "boundary_type": "PHYSICAL_TERMINATION" if termination else "UNRESOLVED"},
    }


def test_event_ordering_uses_pixel_boundary_and_world_reprojection_not_roles():
    transforms = []
    for camera_y in (1.0, -0.6, -2.0):
        transform = np.eye(4)
        transform[1, 3] = camera_y
        transforms.append(transform)
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    world_line = [[10.0, -6.0, z] for z in (-1.0, 0.0, 1.0)]
    thresholds = {"tier_v_min_span_over_target_bbox": 0.8,
                  "min_target_side_fraction": 0.8,
                  "min_external_side_fraction": 0.8,
                  "approach_outside_margin_px": 8.0}
    metrics = [_event_metric(False), _event_metric(False), _event_metric(True)]
    metrics[0].update({"contour_present": True,
                       "span_over_target_bbox_height": 0.01})
    metrics[1].update({"contour_present": True,
                       "span_over_target_bbox_height": 0.43,
                       "boundary_classification": {
                           "boundary_type": "PHYSICAL_TERMINATION"}})
    result = event_ordering_from_geometry(
        metrics, transforms, K, world_line, "LEFT", 100, thresholds)
    assert result["status"] == "PASS"
    assert [row["state"] for row in result["observations"]] == [
        "NO_VALID_EXTERNAL_BOUNDARY", "APPROACH", "FIRST_PHYSICAL_TERMINATION"]
    assert result["uses_role_labels"] is False


def test_event_ordering_fails_without_pixel_derived_approach():
    transforms = [np.eye(4) for _ in range(2)]
    transforms[0][1, 3] = 1.0
    transforms[1][1, 3] = -2.0
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    thresholds = {"tier_v_min_span_over_target_bbox": 0.8,
                  "min_target_side_fraction": 0.8,
                  "min_external_side_fraction": 0.8,
                  "approach_outside_margin_px": 8.0}
    result = event_ordering_from_geometry(
        [_event_metric(False), _event_metric(True)], transforms, K,
        [[10.0, -6.0, 0.0]], "LEFT", 100, thresholds)
    assert result["status"] == "FAIL"


def test_physical_termination_pixel_gate_rejects_weak_span():
    thresholds = {"tier_v_min_span_over_target_bbox": 0.8,
                  "min_target_side_fraction": 0.8,
                  "min_external_side_fraction": 0.8}
    metric = _event_metric(True)
    assert physical_termination_pixel_gate(metric, thresholds)
    metric["span_over_target_bbox_height"] = 0.79
    assert not physical_termination_pixel_gate(metric, thresholds)


def test_act0r2_capture_reads_checkpoint_pose_instead_of_named_poses():
    source = Path("scripts/check_sensor_health.py").read_text()
    section = source.split("def act0r2(", 1)[1].split("def postprocess(", 1)[0]
    assert 'plan["locator_center_pose"]' in section
    assert 'plan["camera_motion_axis"]' in section
    assert 'config["poses"]' not in section
    config = yaml.safe_load(Path("configs/experiments/cap0.yaml").read_text())
    assert config["act0r2"]["maximum_saved_frames"] == 15


def test_published_act0r2_bilateral_result_if_available():
    path = Path("results/act0r2/validation.json")
    if not path.exists():
        return
    result = json.loads(path.read_text())
    gates = result["gates"]
    assert result["schema"] == "act0r2.validation.v1"
    assert result["frame_count"] == 15
    assert result["constraints"]["roles_used_as_labels"] is False
    for direction in ("LEFT", "RIGHT"):
        side = result["directions"][direction]
        assert side["computed_states"][:3] == [
            "NO_VALID_EXTERNAL_BOUNDARY", "APPROACH", "FIRST_PHYSICAL_TERMINATION"]
        assert side["boundary_consensus"]["boundary_type"] == "PHYSICAL_TERMINATION"
        assert side["boundary_consensus"]["consensus_count"] >= 3
        assert side["tier_v"]["status"] == "PASS"
        assert side["same_pose"]["status"] == "PASS"
        assert side["same_pose_world_repeatability"]["status"] == "PASS"
    assert gates["READY_FOR_NEXT_SURFACE"]["status"] == "CONDITIONAL_PASS"
    assert gates["MULTIVIEW_REPEATABILITY"]["status"] == "NOT_EVALUATED"
    assert gates["EXTERNAL_VISUAL_REVIEW"]["status"] == "PENDING"
    assert gates["READY_FOR_COUNTERFACTUAL_ROLLOUT"]["status"] == "NOT_EVALUATED"
    assert gates["READY_FOR_JEPA"]["status"] == "NOT_EVALUATED"
    assert len(list(Path("docs/assets/act0r2").glob("*.jpg"))) == 6
