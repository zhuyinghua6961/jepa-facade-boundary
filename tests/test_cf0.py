import numpy as np
import yaml
from pathlib import Path

from boundary_sweep.observability import (binary_metrics, grouped_kfold,
                                          model_visible_termination,
                                          train_only_pca_ridge)


def test_grouped_folds_keep_counterfactual_branches_together():
    groups = [f"start_{i:02d}" for i in range(20) for _ in range(2)]
    folds = grouped_kfold(groups, n_splits=5, seed=20260817)
    assert len(folds) == 5
    seen = set()
    for fold in folds:
        assert set(fold["train_groups"]).isdisjoint(fold["test_groups"])
        assert len(fold["test_groups"]) == 4
        seen.update(fold["test_groups"])
    assert seen == set(groups)


def test_binary_metrics_known_ranking():
    result = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
    assert result["balanced_accuracy"] == 1.0
    assert result["AUROC"] == 1.0
    assert result["Brier"] < 0.1


def test_train_only_pca_ridge_is_bounded_by_train_rows():
    rng = np.random.default_rng(3)
    train_x = rng.normal(size=(8, 100))
    test_x = rng.normal(size=(2, 100))
    prediction, audit = train_only_pca_ridge(train_x, np.arange(8.0), test_x,
                                             max_components=16)
    assert prediction.shape == (2,)
    assert audit["pca_components"] == 7
    assert audit["preprocessing_fit_scope"] == "training_fold_only"


def test_model_visible_event_requires_all_preregistered_conditions():
    thresholds = {"minimum_span_over_target_bbox": 0.8,
                  "minimum_boundary_penetration_px": 16,
                  "minimum_target_side_fraction": 0.8,
                  "minimum_external_side_fraction": 0.8}
    frame = {"direction": "LEFT", "contour_median_x_px": 20,
             "span_over_target_bbox_height": 0.9,
             "target_side_fraction": 0.9, "external_side_fraction": 0.9,
             "boundary_classification": {"boundary_type": "PHYSICAL_TERMINATION"}}
    assert model_visible_termination(frame, 640, thresholds)["MODEL_VISIBLE_TERMINATION"]
    frame["contour_median_x_px"] = 15.9
    result = model_visible_termination(frame, 640, thresholds)
    assert result["FIRST_PHYSICAL_TERMINATION"]
    assert not result["MODEL_VISIBLE_TERMINATION"]


def test_right_penetration_is_measured_from_right_edge():
    thresholds = {"minimum_span_over_target_bbox": 0.8,
                  "minimum_boundary_penetration_px": 16,
                  "minimum_target_side_fraction": 0.8,
                  "minimum_external_side_fraction": 0.8}
    frame = {"direction": "RIGHT", "contour_median_x_px": 620,
             "span_over_target_bbox_height": 1.0,
             "target_side_fraction": 1.0, "external_side_fraction": 1.0,
             "boundary_classification": {"boundary_type": "PHYSICAL_TERMINATION"}}
    result = model_visible_termination(frame, 640, thresholds)
    assert result["boundary_penetration_px"] == 19.0
    assert result["MODEL_VISIBLE_TERMINATION"]


def test_cf0_config_has_bounded_capture_and_no_forbidden_model_features():
    config = yaml.safe_load(Path("configs/experiments/cf0.yaml").read_text())
    assert sum(row["count"] for row in config["start_sampling"]["bins"]) == 20
    assert config["actions"]["maximum_saved_quartets"] == 340
    assert config["resources"]["python_address_space_limit_bytes"] == 4294967296
    assert config["resources"]["numeric_threads"] == 1
    forbidden = ["absolute_coordinates_are_model_features",
                 "start_offsets_are_model_features", "frame_ids_are_model_features",
                 "planned_roles_are_model_features",
                 "world_boundary_coordinates_are_model_features"]
    assert all(config["constraints"][name] is False for name in forbidden)
