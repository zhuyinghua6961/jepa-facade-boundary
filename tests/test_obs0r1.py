import json
from pathlib import Path

import cv2
import numpy as np

from boundary_sweep.observability import (fixed_length_descriptor,
                                           grouped_history_descriptors,
                                           ridge_fit_predict,
                                           similarity_metrics)


def test_history_window_is_group_local_and_step_zero_invalid():
    records = [
        {"trajectory_id": "LEFT", "step_index": 0, "descriptor": [1.0, 2.0]},
        {"trajectory_id": "LEFT", "step_index": 1, "descriptor": [3.0, 4.0]},
        {"trajectory_id": "RIGHT", "step_index": 0, "descriptor": [9.0, 8.0]},
    ]
    previous, valid = grouped_history_descriptors(records)
    assert np.allclose(previous[0], [1.0, 2.0])
    assert np.allclose(previous[1], [1.0, 2.0])
    assert np.allclose(previous[2], [9.0, 8.0])
    assert valid.tolist() == [False, True, False]


def test_phase_alignment_improves_known_translation():
    base = np.zeros((120, 160, 3), dtype=np.uint8)
    cv2.rectangle(base, (15, 20), (130, 90), (220, 220, 220), 2)
    cv2.circle(base, (70, 55), 16, (255, 100, 30), -1)
    shifted = cv2.warpAffine(base, np.float32([[1, 0, -6], [0, 1, 4]]), (160, 120), borderMode=cv2.BORDER_REFLECT)
    metrics = similarity_metrics(base, shifted)
    assert metrics["phase_aligned_ssim"] > metrics["raw_ssim"]


def test_wide_ridge_uses_sample_space_system(monkeypatch):
    """Wide probes must never allocate a feature-by-feature solve matrix."""
    rng = np.random.default_rng(7)
    train_x = rng.normal(size=(6, 512))
    train_y = rng.normal(size=6)
    test_x = rng.normal(size=(2, 512))
    original_solve = np.linalg.solve
    solve_shapes = []

    def recording_solve(matrix, rhs):
        solve_shapes.append(matrix.shape)
        return original_solve(matrix, rhs)

    monkeypatch.setattr(np.linalg, "solve", recording_solve)
    prediction = ridge_fit_predict(train_x, train_y, test_x)

    assert prediction.shape == (2,)
    assert solve_shapes == [(6, 6)]


def test_fixed_descriptor_has_exact_memory_bounded_length():
    source = np.arange(1447, dtype=np.float32)
    compact = fixed_length_descriptor(source, length=128)
    assert compact.shape == (128,)
    assert compact.dtype == np.float32
    assert compact[0] == source[0]
    assert compact[-1] == source[-1]


def test_obs0r1_result_declares_train_only_preprocessing_if_available():
    path = Path("results/obs0/probe_results_r1.json")
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["train_only_preprocessing"] is True
    assert set(data["probes"]) == {"B0", "B1", "B2", "V0", "V1", "F1", "A0", "F2"}
    assert data["runtime_safety"]["descriptor_dim"] == 128
    assert data["runtime_safety"]["max_linear_system_dimension"] <= 20
    assert data["runtime_safety"]["address_space_limit_bytes"] == 2 * 1024 ** 3


def test_obs0r1_phase_a_gates_if_available():
    path = Path("results/obs0/validation_r1.json")
    assert path.exists()
    data = json.loads(path.read_text())
    for name in ("HISTORY_BOUNDARY_LEAKAGE_FIXED", "TRAIN_ONLY_PREPROCESSING", "SYNTHETIC_ALIGNMENT_TEST", "OBS0R1_REPRODUCIBILITY"):
        assert data["gates"][name]["status"] == "PASS"
