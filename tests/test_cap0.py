import queue
import time
from types import SimpleNamespace

import numpy as np
import pytest

from boundary_sweep.cap0 import (HEALTH_GATES, SEARCH_PLAN_SHA256,
                                 classify_root_cause,
                                 enforce_saved_frame_limit,
                                 evaluate_motion_sequence_rgb,
                                 should_run_act0r1, verify_search_plan)
from boundary_sweep.sensors import (FrameSkippedError, OwnedSensorFrame,
                                    SensorFrameIncompleteError,
                                    SynchronousRGBDSeg)


def _owned(frame):
    return OwnedSensorFrame(frame=frame, timestamp=float(frame), transform_dict={},
                            T_world_camera=np.eye(4), width=1, height=1,
                            fov=90.0, raw_data=b"\x01\x02\x03\x04")


def test_future_frame_fails_immediately_instead_of_waiting():
    frames = queue.Queue()
    frames.put(_owned(11))
    started = time.monotonic()
    with pytest.raises(FrameSkippedError, match="FRAME_SKIPPED"):
        SynchronousRGBDSeg._get(frames, {}, 10, time.monotonic() + 1.0, "rgb")
    assert time.monotonic() - started < 0.1


def test_queue_wait_uses_one_absolute_deadline():
    frames = queue.Queue()
    frames.put(_owned(9))
    started = time.monotonic()
    with pytest.raises(SensorFrameIncompleteError, match="deadline"):
        SynchronousRGBDSeg._get(frames, {}, 10, started + 0.03, "depth")
    elapsed = time.monotonic() - started
    assert 0.02 <= elapsed < 0.15


def test_callback_copies_raw_bytes_before_carla_buffer_changes():
    source = bytearray(b"\x01\x02\x03\x04")
    transform = SimpleNamespace(
        location=SimpleNamespace(x=1.0, y=2.0, z=3.0),
        rotation=SimpleNamespace(pitch=0.0, yaw=0.0, roll=0.0))
    image = SimpleNamespace(raw_data=source, width=1, height=1, frame=7,
                            timestamp=0.35, transform=transform)
    rig = SynchronousRGBDSeg.__new__(SynchronousRGBDSeg)
    rig.fov = 90.0
    rig._queues = {"rgb": queue.Queue()}
    rig._trace_hook = None
    rig._copy_callback("rgb", image)
    source[:] = b"\xff\xff\xff\xff"
    snapshot = rig._queues["rgb"].get_nowait()
    assert isinstance(snapshot.raw_data, bytes)
    assert snapshot.raw_data == b"\x01\x02\x03\x04"


def test_warmup_frames_are_discarded_and_never_saved():
    roles = []
    rig = SynchronousRGBDSeg.__new__(SynchronousRGBDSeg)
    rig._trace_hook = None
    rig.capture = lambda action: roles.append(action["capture_role"]) or {"frame_id": len(roles)}
    result = rig.warmup(min_discard=5, consecutive_complete=3, max_ticks=8)
    assert result["discarded_frames"] == 5
    assert roles == ["WARMUP_DISCARD"] * 5


def test_teleport_requires_three_settle_ticks():
    roles = []
    rig = SynchronousRGBDSeg.__new__(SynchronousRGBDSeg)
    rig._trace_hook = None
    rig.capture = lambda action: roles.append(action["capture_role"]) or {"frame_id": len(roles)}
    result = rig.settle(ticks=3)
    assert len(result["discarded_frames"]) == 3
    assert roles == ["TELEPORT_SETTLE_DISCARD"] * 3


def test_visual_integrity_failure_blocks_act0r1():
    gates = {name: {"status": "PASS"} for name in HEALTH_GATES}
    gates["RENDER_INTEGRITY"] = {"status": "FAIL"}
    assert should_run_act0r1(gates) is False
    gates["RENDER_INTEGRITY"] = {"status": "PASS"}
    assert should_run_act0r1(gates) is True


def test_checked_in_search_checkpoint_sha_is_immutable():
    result = verify_search_plan("results/act0r/search_plan_checkpoint.json")
    assert result["expected_sha256"] == SEARCH_PLAN_SHA256
    assert result["status"] == "PASS"


def test_saved_frame_count_has_a_hard_limit():
    enforce_saved_frame_limit(15, 15)
    with pytest.raises(RuntimeError, match="limit exceeded"):
        enforce_saved_frame_limit(16, 15)


def test_motion_rgb_gate_uses_ssim_only_for_frozen_pose_frames():
    thresholds = {"min_entropy_bits": 0.0, "min_unique_colors": 1,
                  "min_consecutive_ssim": 0.9, "min_historical_ssim": 0.0}
    images = [
        np.zeros((16, 24, 3), dtype=np.uint8),
        np.full((16, 24, 3), 200, dtype=np.uint8),
        np.full((16, 24, 3), 200, dtype=np.uint8),
    ]
    result = evaluate_motion_sequence_rgb(images, [[1, 2]], thresholds)
    assert result["status"] == "PASS"
    assert result["all_frame_consecutive_ssim_diagnostic"][0] < 0.9
    assert result["same_pose_consecutive_ssim"] == [1.0]


def test_motion_rgb_gate_rejects_changed_frozen_pose_frames():
    thresholds = {"min_entropy_bits": 0.0, "min_unique_colors": 1,
                  "min_consecutive_ssim": 0.9, "min_historical_ssim": 0.0}
    images = [
        np.zeros((16, 24, 3), dtype=np.uint8),
        np.full((16, 24, 3), 255, dtype=np.uint8),
    ]
    result = evaluate_motion_sequence_rgb(images, [[0, 1]], thresholds)
    assert result["status"] == "FAIL"


def test_failed_two_gib_probe_classifies_address_space_failure():
    healthy = {
        name: {"capture_status": "PASS", "rgb_integrity": {"status": "PASS"}}
        for name in ("H1", "H2", "H3", "H4", "H5")
    }
    result = classify_root_cause(healthy, {"status": "FAIL"})
    assert result["status"] == "PASS"
    assert result["classification"] == "PYTHON_ADDRESS_SPACE_LIMIT_FAILURE"
    assert result["confidence"] == "CONFIRMED"
