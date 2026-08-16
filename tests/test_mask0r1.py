import json
from pathlib import Path

import numpy as np

from boundary_sweep.segmentation import outer_transition_contour, semantic_instance_consistency


def test_outer_transition_excludes_enclosed_hole():
    mask = np.ones((30, 40), dtype=bool)
    mask[10:20, 12:22] = False
    contour = outer_transition_contour(mask)
    assert not contour[15, 15]


def test_outer_transition_detects_real_termination():
    mask = np.zeros((30, 40), dtype=bool)
    mask[:, 10:30] = True
    contour = outer_transition_contour(mask)
    assert contour[:, 10].sum() > 20
    assert contour[:, 29].sum() > 20


def test_decoder_audit_and_r1_gate_if_results_available():
    path = Path("results/mask0/decoder_audit_r1.json")
    if not path.exists():
        return
    data = json.loads(path.read_text())
    assert data["agreement"] >= 0.9999
    assert data["error_pixels"] == 0


def test_r1_validated_edges_are_sigma_horizontal_pair():
    path = Path("results/mask0/validation_r1.json")
    if not path.exists():
        return
    data = json.loads(path.read_text())
    assert data["gates"]["READY_FOR_ADAPTIVE_PILOT"]["status"] == "PASS"
    assert data["gates"]["VALIDATED_EDGE_SET"]["edges"] == ["surface_sigma:LEFT", "surface_sigma:RIGHT"]
    assert data["gates"]["LEGACY_EDGE_ALIGNMENT"]["status"] == "FAIL"


def test_mask1_pilot_sequence_if_available():
    path = Path("results/mask1/validation.json")
    if not path.exists():
        return
    data = json.loads(path.read_text())
    for row in data["trajectories"]:
        assert row["compressed_state_sequence"] == ["IN", "APPROACH", "STRADDLE"]
        assert row["unknown_ratio"] == 0.0
        assert row["confirmed_straddle_frames"] >= 3
