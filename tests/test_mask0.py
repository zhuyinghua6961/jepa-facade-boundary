import json
from pathlib import Path

import numpy as np

from boundary_sweep.labels import classify_mask_state, facade_outer_envelope
from boundary_sweep.segmentation import decode_instance_id, decode_semantic_tag


class FakeImage:
    height = 1
    width = 2
    raw_data = bytes([3, 2, 1, 255, 0, 0, 7, 255])


def test_mask0_bgra_semantic_and_instance_decode():
    assert decode_semantic_tag(FakeImage()).tolist() == [[1, 7]]
    assert decode_instance_id(FakeImage()).tolist() == [[197121, 7]]


def test_mask0_hole_fill_is_enclosed_and_does_not_shrink():
    mask = np.ones((12, 12), dtype=bool)
    mask[5, 5] = False
    envelope = facade_outer_envelope(mask, 3)
    assert envelope[5, 5]
    assert np.all(envelope >= mask)


def test_mask0_state_semantics_in_straddle_out_unknown():
    target = np.ones((20, 20), dtype=bool)
    envelope = target.copy()
    assert classify_mask_state(target, envelope, [[-20, 0], [-20, 19]], [10, 10], True)["label"] == "IN"
    straddle = np.zeros((20, 20), dtype=bool)
    straddle[:, 5:15] = True
    env = straddle.copy()
    assert classify_mask_state(straddle, env, [[10, -5], [10, 25]], [10, 10], True,
                                {"straddle_side_fraction": 0.01})["label"] == "STRADDLE"
    empty = np.zeros((20, 20), dtype=bool)
    assert classify_mask_state(empty, empty, [[-20, 0], [-20, 19]], [10, 10], True)["label"] == "OUT"
    assert classify_mask_state(target, envelope, [[-20, 0], [-20, 19]], [10, 10], False)["label"] == "UNKNOWN"


def test_mask0_capture_manifest_quartet_pairing_if_available():
    path = Path("results/mask0/capture_manifest.json")
    if not path.exists():
        return
    data = json.loads(path.read_text())
    assert len(data["frames"]) == 30
    for frame in data["frames"]:
        assert len(set(frame["sensor_frames"].values())) == 1
        assert max(frame["sensor_timestamps"].values()) - min(frame["sensor_timestamps"].values()) <= 1e-6
