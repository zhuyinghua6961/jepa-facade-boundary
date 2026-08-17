#!/usr/bin/env python3
"""CAP-0 fail-fast sensor diagnosis and gated ACT-0R1 LEFT recovery."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from boundary_sweep.act0r import (boundary_type_consensus,
                                  checkpoint_pose_alignment,
                                  classify_boundary_pixels,
                                  contour_action_axis_coordinate,
                                  contour_span_metrics,
                                  event_ordering_from_geometry,
                                  official_tier_m,
                                  physical_termination_pixel_gate,
                                  pose_repeatability,
                                  select_repeated_pose_group,
                                  tier_v_from_pixel_frames,
                                  verify_manifest_hashes)
from boundary_sweep.cap0 import (HEALTH_GATES, classify_root_cause,
                                 enforce_saved_frame_limit,
                                 evaluate_motion_sequence_rgb,
                                 evaluate_rgb_sequence, sha256_file,
                                 should_run_act0r1, verify_search_plan)
from boundary_sweep.carla_utils import (discover_carla_root, import_carla,
                                        transform_from_matrix)
from boundary_sweep.segmentation import (BUILDING_TAG, decode_instance_channels)
from boundary_sweep.sensors import (ConsecutiveIncompleteFramesError,
                                    SensorFrameError, SynchronousRGBDSeg)


def _proc_memory(pid: int) -> dict:
    values = {"VmRSS": 0, "VmSize": 0}
    try:
        for line in Path(f"/proc/{int(pid)}/status").read_text().splitlines():
            key = line.split(":", 1)[0]
            if key in values:
                values[key] = int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    return {"rss_bytes": values["VmRSS"], "vms_bytes": values["VmSize"]}


def _gpu_processes() -> list[dict]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], capture_output=True, text=True,
            timeout=5, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in completed.stdout.splitlines():
        try:
            pid, mib = (int(value.strip()) for value in line.split(",", 1))
        except (ValueError, TypeError):
            continue
        rows.append({"pid": pid, "used_memory_mib": mib})
    return rows


class TraceWriter:
    def __init__(self, path: Path, rss_limit: int, carla_pid: int | None):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")
        self.rss_limit = int(rss_limit)
        self.carla_pid = int(carla_pid) if carla_pid else None
        self.lock = threading.Lock()
        self.peak_rss = self.peak_vms = self.peak_carla_rss = 0
        self.rss_limit_exceeded = False

    def __call__(self, record: dict):
        process = _proc_memory(os.getpid())
        carla = _proc_memory(self.carla_pid) if self.carla_pid else {"rss_bytes": 0, "vms_bytes": 0}
        self.peak_rss = max(self.peak_rss, process["rss_bytes"])
        self.peak_vms = max(self.peak_vms, process["vms_bytes"])
        self.peak_carla_rss = max(self.peak_carla_rss, carla["rss_bytes"])
        self.rss_limit_exceeded |= process["rss_bytes"] > self.rss_limit
        payload = {**record, "process_rss_bytes": process["rss_bytes"],
                   "process_vms_bytes": process["vms_bytes"],
                   "carla_rss_bytes": carla["rss_bytes"]}
        with self.lock:
            with self.path.open("a") as handle:
                handle.write(json.dumps(payload, separators=(",", ":")) + "\n")

    def assert_rss(self):
        if self.rss_limit_exceeded:
            raise MemoryError(f"RSS watchdog exceeded {self.rss_limit} bytes")

    def summary(self):
        return {"python_peak_rss_bytes": self.peak_rss,
                "python_peak_vms_bytes": self.peak_vms,
                "carla_peak_rss_bytes": self.peak_carla_rss,
                "rss_watchdog_limit_bytes": self.rss_limit,
                "rss_watchdog_exceeded": self.rss_limit_exceeded,
                "gpu_processes_at_end": _gpu_processes()}


def _transform(carla, pose: dict):
    location, rotation = pose["location"], pose["rotation"]
    return carla.Transform(
        carla.Location(x=float(location["x"]), y=float(location["y"]), z=float(location["z"])),
        carla.Rotation(pitch=float(rotation["pitch"]), yaw=float(rotation["yaw"]),
                       roll=float(rotation["roll"])))


def _shift_pose(pose: dict, delta: dict) -> dict:
    return {"location": {axis: float(pose["location"][axis]) + float(delta[axis])
                         for axis in ("x", "y", "z")},
            "rotation": dict(pose["rotation"])}


def _read_rgb(metadata: dict) -> tuple[np.ndarray, bool]:
    width = int(metadata["sensor_config"]["width"])
    height = int(metadata["sensor_config"]["height"])
    raw_path = PROJECT_ROOT / metadata["files"]["rgb_bgra"]["path"]
    raw = raw_path.read_bytes()
    bgra = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
    decoded = bgra[..., [2, 1, 0]].copy()
    png = np.asarray(Image.open(PROJECT_ROOT / metadata["files"]["rgb"]["path"]).convert("RGB"))
    return decoded, bool(np.array_equal(decoded, png))


def _contact_sheet(test_id: str, frames: list[dict], output: Path, subtitle: str):
    canvas = Image.new("RGB", (960, 300), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for index, metadata in enumerate(frames[:3]):
        image = Image.open(PROJECT_ROOT / metadata["files"]["rgb"]["path"]).convert("RGB")
        image.thumbnail((320, 240))
        canvas.paste(image, (index * 320, 42))
        draw.text((index * 320 + 6, 284), f"frame={metadata['frame_id']}", fill=(240, 240, 240))
    draw.text((8, 7), f"{test_id}: {subtitle}", fill=(255, 230, 70))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=84, optimize=True, progressive=True)


def _role_contact_sheet(frames: list[dict], output: Path):
    canvas = Image.new("RGB", (1280, 600), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for index, metadata in enumerate(frames[:8]):
        row, column = divmod(index, 4)
        image = Image.open(PROJECT_ROOT / metadata["files"]["rgb"]["path"]).convert("RGB")
        image.thumbnail((320, 240))
        x, y = column * 320, row * 300 + 32
        canvas.paste(image, (x, y))
        draw.text((x + 6, row * 300 + 7),
                  f"{metadata['capture_role']} frame={metadata['frame_id']}",
                  fill=(255, 230, 70))
        draw.text((x + 6, y + 245),
                  f"displacement={metadata['commanded_displacement_m']:.4f} m",
                  fill=(240, 240, 240))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=84, optimize=True, progressive=True)


def _comparison(left: Image.Image, right: Image.Image, left_label: str,
                right_label: str, output: Path):
    canvas = Image.new("RGB", (1280, 530), (18, 18, 18))
    left = left.convert("RGB").resize((640, 480))
    right = right.convert("RGB").resize((640, 480))
    canvas.paste(left, (0, 40)); canvas.paste(right, (640, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), left_label, fill=(255, 230, 70))
    draw.text((648, 10), right_label, fill=(255, 230, 70))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=84, optimize=True, progressive=True)


def _sensor_panel(metadata: dict, output: Path):
    files = metadata["files"]
    available = [name for name in ("rgb", "semantic", "instance") if name in files]
    canvas = Image.new("RGB", (640 * len(available), 520), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for index, name in enumerate(available):
        image = Image.open(PROJECT_ROOT / files[name]["path"]).convert("RGB").resize((640, 480))
        canvas.paste(image, (index * 640, 40))
        draw.text((index * 640 + 8, 10), name, fill=(255, 230, 70))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=82, optimize=True, progressive=True)


def _raw_png_panel(metadata: dict, output: Path):
    decoded, _matches = _read_rgb(metadata)
    direct = Image.open(PROJECT_ROOT / metadata["files"]["rgb"]["path"]).convert("RGB")
    _comparison(Image.fromarray(decoded), direct, "decoded from persisted BGRA",
                "persisted PNG", output)


def _hashes_valid(metadata: dict) -> bool:
    return all((PROJECT_ROOT / item["path"]).exists() and
               sha256_file(PROJECT_ROOT / item["path"]) == item["sha256"] and
               (PROJECT_ROOT / item["path"]).stat().st_size == int(item["size_bytes"])
               for item in metadata.get("files", {}).values())


def _pairing_valid(metadata: dict) -> bool:
    frames = metadata.get("sensor_frames", {})
    timestamps = metadata.get("sensor_timestamps", {})
    return bool(frames) and len(set(frames.values())) == 1 and len(timestamps) == len(frames) and \
        max(timestamps.values()) - min(timestamps.values()) <= 1e-6


def _attempt_capture(rig, action: dict, trace: TraceWriter):
    first_error = None
    for attempt in range(2):
        try:
            sample = rig.capture(action, timeout=5.0, tick_timeout=5.0)
            trace.assert_rss()
            return sample
        except SensorFrameError as exc:
            first_error = first_error or repr(exc)
            trace({"event": "formal_capture_retry", "attempt": attempt + 1,
                   "error": repr(exc)})
    raise ConsecutiveIncompleteFramesError(first_error or "two incomplete frames")


def _run_test(test_id: str, sensor_types: tuple[str, ...], pose_name: str,
              pose: dict, config: dict, carla, world, trace: TraceWriter,
              raw_root: Path, assets_root: Path, saved_counter: list[int],
              formal_frames: int = 3, teleport_target: dict | None = None) -> dict:
    started = time.monotonic()
    result = {"test_id": test_id, "sensor_types": list(sensor_types),
              "pose_name": pose_name, "capture_status": "FAIL", "frames": []}
    trace({"event": "test_start", "test_id": test_id, "pose": pose_name,
           "sensor_types": list(sensor_types)})
    rig = None
    try:
        rig = SynchronousRGBDSeg(
            world, carla, _transform(carla, pose), config["sensor"]["width"],
            config["sensor"]["height"], config["sensor"]["horizontal_fov_deg"],
            config["sensor"]["fixed_delta_seconds"], sensor_types=sensor_types,
            trace_hook=trace)
        warmup = rig.warmup(config["warmup"]["minimum_discarded_ticks"],
                            config["warmup"]["required_consecutive_complete"],
                            config["warmup"]["maximum_ticks"])
        result["warmup"] = {"status": "PASS", **warmup}
        if teleport_target is not None:
            rig.set_transform(_transform(carla, teleport_target))
            result["settle"] = {"status": "PASS", **rig.settle(config["teleport"]["settle_ticks"])}
        for index in range(formal_frames):
            enforce_saved_frame_limit(saved_counter[0] + 1,
                                      config["diagnostic"]["maximum_saved_frames"])
            sample = _attempt_capture(rig, {"test_id": test_id, "capture_role": "FORMAL",
                                            "formal_index": index}, trace)
            stem = f"{test_id.lower()}_{index:02d}_frame_{int(sample['frame_id'])}"
            metadata = rig.save(sample, raw_root / test_id, stem)
            metadata.update({"test_id": test_id, "pose_name": pose_name,
                             "formal_index": index, "metadata_path": str(raw_root / test_id / f"{stem}.json")})
            (raw_root / test_id / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")
            result["frames"].append(metadata)
            saved_counter[0] += 1
        images, decode_matches = [], []
        for metadata in result["frames"]:
            image, matches = _read_rgb(metadata)
            images.append(image); decode_matches.append(matches)
        historical = None
        if pose_name == "OLD":
            historical_path = PROJECT_ROOT / config["diagnostic"]["historical_candidate1_rgb"]
            if historical_path.exists():
                historical = np.asarray(Image.open(historical_path).convert("RGB").resize(
                    (config["sensor"]["width"], config["sensor"]["height"])))
        result["rgb_integrity"] = evaluate_rgb_sequence(images, config["visual"], historical)
        result["raw_png_exact"] = all(decode_matches)
        result["hashes_valid"] = all(_hashes_valid(row) for row in result["frames"])
        result["pairing_valid"] = all(_pairing_valid(row) for row in result["frames"])
        result["capture_status"] = "PASS"
        _contact_sheet(test_id, result["frames"], assets_root / f"{test_id.lower()}_contact.jpg",
                       f"{pose_name} sensors={'+'.join(sensor_types)}")
        _raw_png_panel(result["frames"][0], assets_root / f"{test_id.lower()}_raw_vs_png.jpg")
        _sensor_panel(result["frames"][0], assets_root / f"{test_id.lower()}_sensors.jpg")
    except Exception as exc:
        result["error"] = repr(exc)
        result.setdefault("warmup", {"status": "FAIL", "error": repr(exc)})
        trace({"event": "test_failure", "test_id": test_id, "error": repr(exc)})
    finally:
        if rig is not None:
            rig.close()
    result["duration_s"] = time.monotonic() - started
    trace({"event": "test_end", "test_id": test_id,
           "capture_status": result["capture_status"], "duration_s": result["duration_s"]})
    return result


def _write_matrix(tests: dict, path: Path):
    fields = ["test_id", "pose_name", "sensor_types", "capture_status", "saved_frames",
              "warmup_status", "pairing_valid", "raw_png_exact", "hashes_valid",
              "rgb_integrity", "duration_s", "error"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for test_id in ("H1", "H2", "H3", "H4", "H5"):
            row = tests.get(test_id, {})
            writer.writerow({"test_id": test_id, "pose_name": row.get("pose_name"),
                             "sensor_types": "+".join(row.get("sensor_types", [])),
                             "capture_status": row.get("capture_status", "NOT_RUN"),
                             "saved_frames": len(row.get("frames", [])),
                             "warmup_status": row.get("warmup", {}).get("status"),
                             "pairing_valid": row.get("pairing_valid"),
                             "raw_png_exact": row.get("raw_png_exact"),
                             "hashes_valid": row.get("hashes_valid"),
                             "rgb_integrity": row.get("rgb_integrity", {}).get("status"),
                             "duration_s": row.get("duration_s"), "error": row.get("error")})


def _health_gates(tests: dict, config: dict, root_cause: dict, trace: TraceWriter) -> dict:
    all_frames = [frame for row in tests.values() for frame in row.get("frames", [])]
    quartet = [tests.get(name, {}) for name in ("H3", "H4")]
    warmups = [row for row in tests.values() if row.get("capture_status") != "NOT_RUN"]
    visual_rows = [row for row in tests.values() if row.get("frames")]
    return {
        "TICK_FAIL_FAST": {"status": "PASS" if config["sensor"]["tick_timeout_s"] <= 5 else "FAIL",
                           "world_tick_timeout_s": config["sensor"]["tick_timeout_s"]},
        "QUEUE_DEADLINE": {"status": "PASS" if config["sensor"]["queue_deadline_s"] <= 5 else "FAIL",
                           "single_deadline_s": config["sensor"]["queue_deadline_s"]},
        "RAW_BUFFER_OWNERSHIP": {"status": "PASS" if all_frames and all(
            frame.get("sensor_config", {}).get("raw_buffer_ownership") == "bytes copied inside callback"
            for frame in all_frames) else "FAIL"},
        "RAW_LENGTH_AND_HASH": {"status": "PASS" if all_frames and all(
            row.get("hashes_valid") and row.get("raw_png_exact") for row in visual_rows) else "FAIL",
                                "saved_frames": len(all_frames)},
        "GPU_WARMUP_COMPLETE": {"status": "PASS" if warmups and all(
            row.get("warmup", {}).get("status") == "PASS" for row in warmups) else "FAIL"},
        "KNOWN_GOOD_POSE_RGB_INTEGRITY": {"status": "PASS" if tests.get("H1", {}).get(
            "rgb_integrity", {}).get("status") == "PASS" else "FAIL"},
        "QUARTET_PAIRING_HEALTH": {"status": "PASS" if all(
            row.get("capture_status") == "PASS" and row.get("pairing_valid") for row in quartet) else "FAIL"},
        "POST_TELEPORT_HEALTH": {"status": "PASS" if tests.get("H5", {}).get(
            "capture_status") == "PASS" and tests.get("H5", {}).get("pairing_valid") else "FAIL"},
        "RENDER_INTEGRITY": {"status": "PASS" if len(visual_rows) == 5 and all(
            row.get("rgb_integrity", {}).get("status") == "PASS" for row in visual_rows) else "FAIL"},
        "ROOT_CAUSE_CLASSIFIED": {"status": root_cause["status"],
                                  "classification": root_cause["classification"],
                                  "confidence": root_cause["confidence"]},
        "RSS_WATCHDOG": {"status": "FAIL" if trace.rss_limit_exceeded else "PASS"},
    }


def diagnose(args, config, carla, client):
    result_root, raw_root, assets_root = Path(args.result_root), Path(args.raw_root), Path(args.assets_root)
    result_root.mkdir(parents=True, exist_ok=True); assets_root.mkdir(parents=True, exist_ok=True)
    if raw_root.exists() and any(raw_root.rglob("*")):
        raise RuntimeError(f"diagnostic raw output is non-empty: {raw_root}")
    checkpoint = verify_search_plan(PROJECT_ROOT / config["checkpoint"]["path"],
                                    config["checkpoint"]["sha256"])
    if checkpoint["status"] != "PASS":
        raise RuntimeError("search checkpoint SHA mismatch")
    trace = TraceWriter(result_root / "timing_trace.jsonl",
                        config["resources"]["python_rss_watchdog_bytes"], args.carla_pid)
    world = client.get_world()
    tests, saved_counter = {}, [0]
    started = time.monotonic()
    matrix = [
        ("H1", ("rgb",), "OLD", config["poses"]["OLD"]),
        ("H2", ("rgb",), "NEW", config["poses"]["NEW"]),
        ("H3", ("rgb", "depth", "semantic", "instance"), "OLD", config["poses"]["OLD"]),
        ("H4", ("rgb", "depth", "semantic", "instance"), "NEW", config["poses"]["NEW"]),
    ]
    for test_id, sensors, pose_name, pose in matrix:
        if time.monotonic() - started >= config["diagnostic"]["internal_duration_limit_s"]:
            tests[test_id] = {"test_id": test_id, "pose_name": pose_name,
                              "sensor_types": list(sensors), "capture_status": "NOT_RUN",
                              "error": "diagnostic duration limit reached"}
            continue
        tests[test_id] = _run_test(test_id, sensors, pose_name, pose, config, carla, world,
                                   trace, raw_root, assets_root, saved_counter)
    passed_base = "OLD" if tests.get("H3", {}).get("capture_status") == "PASS" else \
        "NEW" if tests.get("H4", {}).get("capture_status") == "PASS" else None
    if passed_base and time.monotonic() - started < config["diagnostic"]["internal_duration_limit_s"]:
        pose = config["poses"][passed_base]
        target = _shift_pose(pose, config["poses"]["TELEPORT_DELTA"])
        tests["H5"] = _run_test("H5", ("rgb", "depth", "semantic", "instance"),
                                f"{passed_base}_PLUS_0.5M", pose, config, carla, world,
                                trace, raw_root, assets_root, saved_counter,
                                teleport_target=target)
    else:
        tests["H5"] = {"test_id": "H5", "pose_name": "NOT_AVAILABLE",
                       "sensor_types": ["rgb", "depth", "semantic", "instance"],
                       "capture_status": "NOT_RUN",
                       "error": "no passing quartet pose or diagnostic duration limit reached"}
    enforce_saved_frame_limit(saved_counter[0], config["diagnostic"]["maximum_saved_frames"])
    root_cause = classify_root_cause(tests)
    gates = _health_gates(tests, config, root_cause, trace)
    recovered = should_run_act0r1(gates)
    resource_summary = trace.summary()
    resource_summary.update({"configured_python_as_limit_bytes": config["resources"]["python_address_space_limit_bytes"],
                             "actual_outer_python_as_limit_bytes": args.actual_as_limit})
    validation = {"schema": "cap0.validation.v1", "phase": "CAP-0 sensor diagnosis",
                  "run_status": "COMPLETE" if len(tests) == 5 else "INCOMPLETE",
                  "checkpoint": checkpoint, "saved_frame_count": saved_counter[0],
                  "maximum_saved_frames": config["diagnostic"]["maximum_saved_frames"],
                  "tests": tests, "gates": gates, "root_cause": root_cause,
                  "resources": resource_summary,
                  "CAPTURE_STACK_RECOVERED": "PASS" if recovered else "FAIL",
                  "ACT0R1_AUTHORIZED": recovered,
                  "constraints": {"adaptive_search_rerun": False, "rollout_run": False,
                                  "dataset_expansion_run": False, "jepa_training_run": False,
                                  "models_downloaded": False}}
    (result_root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    (result_root / "root_cause.json").write_text(json.dumps(root_cause, indent=2) + "\n")
    (result_root / "diagnostic_manifest.json").write_text(json.dumps(
        {"schema": "cap0.diagnostic_manifest.v1", "tests": tests,
         "saved_frame_count": saved_counter[0]}, indent=2) + "\n")
    _write_matrix(tests, result_root / "health_matrix.csv")
    generate_cap0_assets(tests, config, assets_root)
    print(json.dumps({"saved_frames": saved_counter[0], "gates": gates,
                      "root_cause": root_cause, "ACT0R1_AUTHORIZED": recovered}, indent=2))
    return 0


def generate_cap0_assets(tests: dict, config: dict, assets_root: Path):
    historical_path = PROJECT_ROOT / config["diagnostic"]["historical_candidate1_rgb"]
    if historical_path.exists() and tests.get("H1", {}).get("frames"):
        current = Image.open(PROJECT_ROOT / tests["H1"]["frames"][0]["files"]["rgb"]["path"])
        _comparison(Image.open(historical_path), current, "historical ACT-0S candidate 1",
                    "CAP-0 H1 OLD RGB-only", assets_root / "old_vs_historical.jpg")
    comparisons = [("H1", "H3", "old_rgb_vs_quartet.jpg"),
                   ("H2", "H4", "new_rgb_vs_quartet.jpg")]
    for left_id, right_id, filename in comparisons:
        left, right = tests.get(left_id, {}), tests.get(right_id, {})
        if left.get("frames") and right.get("frames"):
            _comparison(Image.open(PROJECT_ROOT / left["frames"][0]["files"]["rgb"]["path"]),
                        Image.open(PROJECT_ROOT / right["frames"][0]["files"]["rgb"]["path"]),
                        left_id, right_id, assets_root / filename)
    h5 = tests.get("H5", {})
    base_id = "H3" if h5.get("pose_name", "").startswith("OLD") else "H4"
    base = tests.get(base_id, {})
    if base.get("frames") and h5.get("frames"):
        _comparison(Image.open(PROJECT_ROOT / base["frames"][0]["files"]["rgb"]["path"]),
                    Image.open(PROJECT_ROOT / h5["frames"][0]["files"]["rgb"]["path"]),
                    f"{base_id} before teleport", "H5 after 0.5m teleport",
                    assets_root / "teleport_before_after.jpg")


def as2_probe(args, config, carla, client):
    result_root = Path(args.result_root); result_root.mkdir(parents=True, exist_ok=True)
    validation = json.loads((result_root / "validation.json").read_text())
    if validation.get("tests", {}).get("H1", {}).get("capture_status") != "PASS":
        result = {"schema": "cap0.as2_probe.v1", "status": "NOT_RUN",
                  "reason": "4 GiB H1 did not pass"}
    else:
        trace = TraceWriter(result_root / "timing_trace_as2.jsonl",
                            config["resources"]["python_rss_watchdog_bytes"], args.carla_pid)
        temporary = result_root / "as2_probe_raw"
        if temporary.exists() and any(temporary.rglob("*")):
            raise RuntimeError("2 GiB probe temporary output is non-empty")
        row = _run_test("AS2", ("rgb",), "OLD", config["poses"]["OLD"], config,
                        carla, client.get_world(), trace, temporary, Path(args.assets_root), [0],
                        formal_frames=1)
        result = {"schema": "cap0.as2_probe.v1",
                  "status": "PASS" if row.get("capture_status") == "PASS" else "FAIL",
                  "test": row, "resources": trace.summary(), "temporary_raw": str(temporary)}
    (result_root / "as2_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


def act0r1(args, config, carla, client):
    cap0 = json.loads((Path(args.result_root) / "validation.json").read_text())
    if not cap0.get("ACT0R1_AUTHORIZED") or not should_run_act0r1(cap0.get("gates", {})):
        raise RuntimeError("CAP-0 health gates do not authorize ACT-0R1")
    checkpoint_path = PROJECT_ROOT / config["checkpoint"]["path"]
    if verify_search_plan(checkpoint_path, config["checkpoint"]["sha256"])["status"] != "PASS":
        raise RuntimeError("search checkpoint changed")
    checkpoint = json.loads(checkpoint_path.read_text())
    plan = next(row for row in checkpoint["plans"] if row["candidate_index"] == 1)
    side = plan["searches"]["LEFT"]
    roles = [{"role": "CENTER", "displacement_m": 0.0}, *side["saved_roles"]]
    if len(roles) != 8:
        raise RuntimeError(f"expected 8 ACT-0R1 roles, got {len(roles)}")
    result_root, raw_root, assets_root = Path(args.act0r1_result_root), Path(args.act0r1_raw_root), Path(args.act0r1_assets_root)
    result_root.mkdir(parents=True, exist_ok=True); assets_root.mkdir(parents=True, exist_ok=True)
    if raw_root.exists() and any(raw_root.rglob("*")):
        raise RuntimeError("ACT-0R1 raw output is non-empty")
    trace = TraceWriter(result_root / "timing_trace.jsonl",
                        config["resources"]["python_rss_watchdog_bytes"], args.carla_pid)
    world = client.get_world()
    base = config["poses"]["OLD"]
    rig = SynchronousRGBDSeg(world, carla, _transform(carla, base),
                            config["sensor"]["width"], config["sensor"]["height"],
                            config["sensor"]["horizontal_fov_deg"],
                            config["sensor"]["fixed_delta_seconds"], trace_hook=trace)
    frames = []
    try:
        warmup = rig.warmup(config["warmup"]["minimum_discarded_ticks"],
                            config["warmup"]["required_consecutive_complete"],
                            config["warmup"]["maximum_ticks"])
        for index, role in enumerate(roles):
            pose = _shift_pose(base, {"x": -float(role["displacement_m"]), "y": 0.0, "z": 0.0})
            rig.set_transform(_transform(carla, pose))
            settle = rig.settle(config["teleport"]["settle_ticks"])
            sample = _attempt_capture(rig, {"sequence_id": "act0r1_candidate_01_left",
                                            "capture_role": role["role"],
                                            "commanded_displacement_m": role["displacement_m"]}, trace)
            stem = f"left_{index:02d}_{role['role'].lower()}"
            metadata = rig.save(sample, raw_root, stem)
            metadata.update({"candidate_index": 1, "bbox_id": 48389, "direction": "LEFT",
                             "capture_role": role["role"], "target_instance_id": plan["target_instance_id"],
                             "commanded_displacement_m": role["displacement_m"],
                             "settle": settle, "metadata_path": str(raw_root / f"{stem}.json")})
            (raw_root / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")
            frames.append(metadata)
            enforce_saved_frame_limit(len(frames), config["act0r1"]["maximum_saved_frames"])
    finally:
        rig.close()
    images = [_read_rgb(frame)[0] for frame in frames]
    repeat_indices = [index for index, row in enumerate(frames)
                      if row["capture_role"].startswith("STRADDLE")]
    integrity = evaluate_motion_sequence_rgb(images, [repeat_indices], config["visual"])
    repeats = [row for row in frames if row["capture_role"].startswith("STRADDLE")]
    translations = np.asarray([row["T_world_camera"] for row in repeats], dtype=float)[:, :3, 3]
    position_error = float(np.max(np.linalg.norm(translations - translations[0], axis=1)))
    rotations = [row["camera_transform"]["rotation"] for row in repeats]
    rotation_error = max(abs(float(row[key]) - float(rotations[0][key]))
                         for row in rotations for key in ("pitch", "yaw", "roll"))
    instance_ids = []
    for frame in frames:
        channels = decode_instance_channels(np.asarray(Image.open(
            PROJECT_ROOT / frame["files"]["instance"]["path"]).convert("RGB")))
        center = channels["instance_id_16bit"][160:320, 213:427]
        values, counts = np.unique(center[center > 0], return_counts=True)
        instance_ids.append(int(values[np.argmax(counts)]) if len(values) else None)
    pairing = all(_pairing_valid(row) for row in frames)
    same_pose = position_error <= 0.01 and rotation_error <= 0.05
    instance_stable = bool(instance_ids) and len(set(instance_ids)) == 1
    complete = len(frames) == 8 and pairing and integrity["status"] == "PASS" and same_pose and instance_stable
    gates = {"CAPTURE_STACK_RECOVERED": {"status": "PASS"},
             "CANDIDATE1_LEFT_CAPTURE_COMPLETE": {"status": "PASS" if complete else "FAIL", "frames": len(frames)},
             "SENSOR_QUADRUPLET_PAIRING": {"status": "PASS" if pairing else "FAIL"},
             "RGB_VISUAL_INTEGRITY": integrity,
             "SAME_POSE_CONFIRMATION": {"status": "PASS" if same_pose else "FAIL",
                                        "position_error_m": position_error,
                                        "rotation_error_deg": rotation_error},
             "TARGET_INSTANCE_STABILITY": {"status": "PASS" if instance_stable else "FAIL",
                                           "instance_ids": instance_ids},
             "READY_TO_RESUME_ACT0R": {"status": "PASS" if complete else "FAIL"},
             "READY_FOR_COUNTERFACTUAL_ROLLOUT": {"status": "NOT_EVALUATED"},
             "READY_FOR_JEPA": {"status": "NOT_EVALUATED"}}
    validation = {"schema": "act0r1.validation.v1", "frames": frames, "gates": gates,
                  "warmup": warmup, "resources": trace.summary(),
                  "constraints": {"adaptive_search_rerun": False, "right_capture_run": False,
                                  "other_candidates_captured": False, "rollout_run": False,
                                  "jepa_training_run": False}}
    (result_root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    _contact_sheet("ACT0R1_LEFT", frames[:3], assets_root / "left_start_contact.jpg", "CENTER/INSIDE/PRE")
    _contact_sheet("ACT0R1_LEFT", frames[2:5], assets_root / "left_straddle_contact.jpg", "STRADDLE repeats")
    _role_contact_sheet(frames, assets_root / "left_all_roles.jpg")
    print(json.dumps({"frames": len(frames), "gates": gates}, indent=2))
    return 0


def _act0r2_pose(center_matrix, action_axis, displacement_m, direction):
    matrix = np.asarray(center_matrix, dtype=float).copy()
    axis = np.asarray(action_axis, dtype=float)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    sign = -1.0 if direction == "LEFT" else 1.0 if direction == "RIGHT" else 0.0
    matrix[:3, 3] += sign * float(displacement_m) * axis
    return matrix


def _act0r2_pose_sha(matrix) -> str:
    payload = json.dumps(np.asarray(matrix, dtype=float).tolist(), separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _act0r2_arrays(entry):
    rgb = np.asarray(Image.open(PROJECT_ROOT / entry["files"]["rgb"]["path"]).convert("RGB"))
    semantic_rgb = np.asarray(Image.open(
        PROJECT_ROOT / entry["files"]["semantic"]["path"]).convert("RGB"))
    instance_rgb = np.asarray(Image.open(
        PROJECT_ROOT / entry["files"]["instance"]["path"]).convert("RGB"))
    semantic = decode_instance_channels(semantic_rgb)["semantic_tag"]
    instance_channels = decode_instance_channels(instance_rgb)
    instance = instance_channels["instance_id_16bit"]
    depth = np.load(PROJECT_ROOT / entry["files"]["depth_m"]["path"], allow_pickle=False)
    return rgb, depth, semantic, instance, instance_channels["semantic_tag"]


def _mask_bbox(mask):
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return {"x_min": int(xs.min()), "y_min": int(ys.min()),
            "x_max": int(xs.max()), "y_max": int(ys.max()),
            "width_px": int(xs.max() - xs.min() + 1),
            "height_px": int(ys.max() - ys.min() + 1)}


def _act0r2_frame_metric(entry, direction, action_axis, act0r_config):
    _rgb, depth, semantic, instance, instance_semantic = _act0r2_arrays(entry)
    target_id = int(entry["target_instance_id"])
    mask = (semantic == BUILDING_TAG) & (instance == target_id)
    span = contour_span_metrics(mask, direction)
    contour = span.pop("contour")
    boundary = classify_boundary_pixels(
        mask, contour, depth, semantic, instance, target_id, direction,
        act0r_config["boundary_classification"])
    metric = contour_action_axis_coordinate(
        contour, depth, np.asarray(entry["K"], dtype=float),
        np.asarray(entry["T_world_camera"], dtype=float), action_axis)
    metric["world_points_sample"] = metric["world_points_sample"][:64]
    count = int(mask.sum())
    return {
        "frame_id": int(entry["frame_id"]),
        "capture_role": entry["capture_role"],
        "role_used_as_scientific_label": False,
        "target_pixel_count": count,
        "target_coverage": float(mask.mean()),
        "mask_bbox": _mask_bbox(mask),
        "target_instance_camera_building_fraction": (
            float(np.mean(instance_semantic[instance == target_id] == BUILDING_TAG))
            if np.any(instance == target_id) else 0.0),
        **span,
        "boundary_classification": boundary,
        "tier_m_frame": metric,
    }


def _act0r2_center_boundary_present(metric, act0r_config):
    thresholds = {
        **act0r_config["contour"],
        "tier_v_min_span_over_target_bbox": act0r_config["search"][
            "search_min_span_over_target_bbox"],
    }
    return physical_termination_pixel_gate(metric, thresholds)


def _save_act0r2_frame(rig, sample, raw_root, stem, plan, direction, role,
                       displacement_m, center_pose, center_frame_id, center_pose_sha,
                       settle):
    metadata = rig.save(sample, raw_root, stem)
    actual = np.asarray(metadata["T_world_camera"], dtype=float)
    metadata.update({
        "sequence_id": f"act0r2_candidate_01_{direction.lower()}",
        "candidate_index": 1,
        "bbox_id": int(plan["bbox_id"]),
        "direction": direction,
        "capture_role": role,
        "role_used_as_scientific_label": False,
        "target_instance_id": int(plan["target_instance_id"]),
        "commanded_displacement_m": float(displacement_m),
        "actual_displacement_from_center_m": float(np.linalg.norm(
            actual[:3, 3] - np.asarray(center_pose, dtype=float)[:3, 3])),
        "shared_center_frame_id": int(center_frame_id),
        "shared_center_pose_sha256": center_pose_sha,
        "settle": settle,
        "metadata_path": str(raw_root / f"{stem}.json"),
    })
    (raw_root / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def _act0r2_pairing(entry):
    frames = entry.get("sensor_frames", {})
    stamps = entry.get("sensor_timestamps", {})
    return (set(frames) == {"rgb", "depth", "semantic", "instance"} and
            len(set(frames.values())) == 1 and set(stamps) == set(frames) and
            max(stamps.values()) - min(stamps.values()) <= 1e-6)


def _act0r2_overlay(entry, direction, metric):
    rgb, _depth, semantic, instance, _instance_semantic = _act0r2_arrays(entry)
    mask = (semantic == BUILDING_TAG) & (instance == int(entry["target_instance_id"]))
    span = contour_span_metrics(mask, direction)
    image = rgb.copy()
    image[mask] = (0.62 * image[mask] + 0.38 * np.array([35, 220, 80])).astype(np.uint8)
    image[span["contour"]] = (255, 215, 0)
    return Image.fromarray(image, mode="RGB")


def _act0r2_role_sheet(entries, metrics, states, direction, output):
    canvas = Image.new("RGB", (1280, 630), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    for index, (entry, metric, state) in enumerate(zip(entries, metrics, states)):
        tile = _act0r2_overlay(entry, direction, metric).resize((320, 240))
        x, y = (index % 4) * 320, (index // 4) * 280 + 70
        canvas.paste(tile, (x, y))
        draw.text((x + 6, y - 24), f"{entry['capture_role']} frame={entry['frame_id']}",
                  fill=(255, 230, 70))
        draw.text((x + 6, y + 222),
                  f"pixel={state} d={entry['actual_displacement_from_center_m']:.3f}m",
                  fill=(245, 245, 245))
    draw.text((8, 7), f"ACT-0R2 candidate 1 {direction}: role is provenance only",
              fill=(255, 230, 70))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=84, optimize=True, progressive=True)


def _act0r2_evidence_sheet(entry, direction, metric, output):
    rgb, depth, semantic, instance, _instance_semantic = _act0r2_arrays(entry)
    target_id = int(entry["target_instance_id"])
    mask = (semantic == BUILDING_TAG) & (instance == target_id)
    contour = contour_span_metrics(mask, direction)["contour"]
    panels = [np.asarray(_act0r2_overlay(entry, direction, metric))]
    semantic_panel = rgb.copy()
    semantic_panel[semantic == BUILDING_TAG] = (50, 175, 235)
    semantic_panel[contour] = (255, 215, 0)
    panels.append(semantic_panel)
    instance_panel = np.zeros_like(rgb)
    instance_panel[..., 0] = (instance & 255).astype(np.uint8)
    instance_panel[..., 1] = ((instance >> 8) & 255).astype(np.uint8)
    instance_panel[..., 2] = semantic.astype(np.uint8) * 9
    instance_panel[contour] = (255, 255, 0)
    panels.append(instance_panel)
    target_depth = float(np.median(depth[mask])) if mask.any() else 0.0
    residual = np.clip((depth - target_depth) / 8.0, -1.0, 1.0)
    depth_panel = np.zeros_like(rgb)
    depth_panel[..., 0] = np.where(residual < 0, -residual * 255, 0).astype(np.uint8)
    depth_panel[..., 2] = np.where(residual > 0, residual * 255, 0).astype(np.uint8)
    depth_panel[..., 1] = (255 - np.abs(residual) * 180).astype(np.uint8)
    depth_panel[contour] = (255, 255, 0)
    panels.append(depth_panel)
    canvas = Image.new("RGB", (1280, 560), (20, 20, 20))
    labels = ["RGB + target mask + contour", "semantic", "instance evidence",
              "z-depth residual: red closer / blue farther"]
    draw = ImageDraw.Draw(canvas)
    for index, (panel, label) in enumerate(zip(panels, labels)):
        canvas.paste(Image.fromarray(panel).resize((320, 240)), (index * 320, 38))
        draw.text((index * 320 + 6, 282), label, fill=(245, 245, 245))
    classification = metric["boundary_classification"]
    draw.text((8, 7),
              f"candidate 1 {direction} first termination frame={entry['frame_id']} type={classification['boundary_type']}",
              fill=(255, 230, 70))
    detail = [
        f"coverage={metric['target_coverage']:.6f} span/bbox={metric['span_over_target_bbox_height']:.3f}",
        f"bilateral={classification.get('bilateral_sample_count')} valid_depth={classification.get('valid_depth_pair_count')}",
        f"external-target depth median={classification.get('external_minus_target_depth_median_m')} m",
        f"external Building={classification.get('external_building_fraction')} non-target ID={classification.get('external_non_target_instance_fraction')}",
        "EXTERNAL_VISUAL_REVIEW=PENDING",
    ]
    for index, line in enumerate(detail):
        draw.text((12, 330 + index * 38), line, fill=(238, 238, 238))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=84, optimize=True, progressive=True)


def _write_act0r2_docs(validation, output):
    gates = validation["gates"]
    lines = [
        "# ACT-0R2 Candidate 1 Bilateral Event Audit", "",
        "This run used the checkpoint `locator_center_pose` and action axis directly.",
        "Role names are provenance only. CARLA pixels, z-depth, K and per-frame",
        "`T_world_camera` determine every scientific state. No rollout or JEPA training ran.", "",
        "External visual review is **PENDING**.", "", "## Gates", "",
        "| Gate | Status |", "| --- | --- |",
    ]
    for name, value in gates.items():
        lines.append(f"| {name} | {value.get('status')} |")
    raw = validation["raw"]
    resources = validation["resources"]
    lines.extend(["", "## Runtime and raw integrity", "",
                  f"- Persisted frames: {validation['frame_count']} (one shared CENTER plus seven per side)",
                  f"- Raw files: {raw['file_count']}; bytes: {raw['size_bytes']}",
                  f"- Manifest payload hashes: {raw['hash_audit']['checked_file_count']} checked, {raw['hash_audit']['status']}",
                  f"- Python AS limit: {resources.get('actual_python_address_space_limit_bytes')} bytes",
                  f"- Python peak RSS/VMS: {resources.get('python_peak_rss_bytes')} / {resources.get('python_peak_vms_bytes')} bytes",
                  f"- CARLA peak RSS: {resources.get('carla_peak_rss_bytes')} bytes",
                  f"- CARLA initial/effective AS limits: {resources.get('initial_carla_address_space_limit_bytes')} / {resources.get('effective_carla_address_space_limit_bytes')} bytes",
                  "- The 16 GiB CARLA launch failed during engine initialization; the bounded capture used 32 GiB. The Python limit remained 4 GiB.",
                  "", "## Bilateral evidence", "",
                  "![Shared CENTER and first bilateral events](assets/act0r2/same_start_comparison.jpg)", "",
                  "![CENTER bilateral absence](assets/act0r2/center_bilateral_absence.jpg)", "",
                  "![LEFT roles](assets/act0r2/left_all_roles.jpg)", "",
                  "![RIGHT roles](assets/act0r2/right_all_roles.jpg)", "",
                  "![LEFT first termination](assets/act0r2/left_first_termination.jpg)", "",
                  "![RIGHT first termination](assets/act0r2/right_first_termination.jpg)", "",
                  "## Per-direction summary", "",
                  "| Direction | Event order | Boundary | Tier V | Same-pose world spread |",
                  "| --- | --- | --- | --- | ---: |"])
    for direction in ("LEFT", "RIGHT"):
        row = validation["directions"].get(direction, {})
        lines.append(f"| {direction} | {row.get('event_ordering', {}).get('status')} | "
                     f"{row.get('boundary_consensus', {}).get('boundary_type')} | "
                     f"{row.get('tier_v', {}).get('status')} | "
                     f"{row.get('same_pose_world_repeatability', {}).get('spread_m')} m |")
    lines.extend(["", "## Per-frame pixel states", "",
                  "| Direction | Frame | Role (provenance) | Pixel/geometric state | Coverage | Span/bbox | Boundary u |",
                  "| --- | ---: | --- | --- | ---: | ---: | ---: |"])
    for direction in ("LEFT", "RIGHT"):
        row = validation["directions"][direction]
        for metric, observation in zip(row["frames"], row["event_ordering"]["observations"]):
            u = observation.get("projected_boundary_median_u_px")
            lines.append(f"| {direction} | {metric['frame_id']} | {metric['capture_role']} | "
                         f"{observation['state']} | {metric['target_coverage']:.6f} | "
                         f"{metric['span_over_target_bbox_height']:.6f} | {u:.6f} px |")
    lines.extend(["", "Same-pose spread is not multiview repeatability.",
                  "`MULTIVIEW_REPEATABILITY` remains `NOT_EVALUATED`.", "",
                  "The first offline pass labeled weak partial contours UNKNOWN before applying the",
                  "unchanged Tier V threshold and world-line projection. The corrected precedence",
                  "maps a two-pixel far-outside contour to NO_VALID_EXTERNAL_BOUNDARY and the",
                  "partial near-outside contour to APPROACH. No threshold or raw frame changed.", ""])
    output.write_text("\n".join(lines))


def postprocess_act0r2(args, config, act0r_config):
    result_root = Path(args.act0r2_result_root)
    manifest = json.loads((result_root / "capture_manifest.json").read_text())
    manifest.setdefault("resources", {}).pop("gpu_processes_at_end", None)
    (result_root / "capture_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    frames = manifest["frames"]
    plan = manifest["checkpoint_plan"]
    axis = np.asarray(plan["camera_motion_axis"], dtype=float)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    center = next(row for row in frames if row["direction"] == "CENTER")
    alignment = checkpoint_pose_alignment(
        center["T_world_camera"], plan["locator_center_pose"],
        config["act0r2"]["center_max_position_error_m"],
        config["act0r2"]["center_max_rotation_error_deg"])
    center_metrics = {direction: _act0r2_frame_metric(
        center, direction, axis, act0r_config) for direction in ("LEFT", "RIGHT")}
    center_absent = {direction: not _act0r2_center_boundary_present(
        center_metrics[direction], act0r_config) for direction in ("LEFT", "RIGHT")}
    event_thresholds = {**act0r_config["contour"],
                        "approach_outside_margin_px": config["act0r2"][
                            "approach_outside_margin_px"]}
    directions, csv_rows = {}, []
    for direction in ("LEFT", "RIGHT"):
        entries = [row for row in frames if row["direction"] == direction]
        metrics = [_act0r2_frame_metric(row, direction, axis, act0r_config)
                   for row in entries]
        transforms = [np.asarray(row["T_world_camera"], dtype=float) for row in entries]
        frozen = select_repeated_pose_group(
            transforms, minimum_count=4,
            position_tolerance_m=config["act0r2"]["center_max_position_error_m"],
            rotation_tolerance_deg=config["act0r2"]["center_max_rotation_error_deg"])
        frozen_metrics = [metrics[index] for index in frozen.get("indices", [])]
        classifications = [row["boundary_classification"] for row in frozen_metrics]
        consensus = boundary_type_consensus(
            classifications, act0r_config["offline_audit"]["boundary_consensus_min_frames"])
        tier_v = tier_v_from_pixel_frames(
            frozen_metrics, act0r_config["contour"],
            act0r_config["offline_audit"]["tier_v_min_pass_frames"])
        same_pose = pose_repeatability(
            [transforms[index] for index in frozen.get("indices", [])],
            config["act0r2"]["center_max_position_error_m"],
            config["act0r2"]["center_max_rotation_error_deg"])
        tier_m = official_tier_m(
            [row["tier_m_frame"] for row in frozen_metrics],
            act0r_config["tier_m"]["sensitivity_thresholds_m"],
            act0r_config["tier_m"]["gate_spread_m"])
        first_valid = next((row for row in metrics
                            if physical_termination_pixel_gate(row, event_thresholds)), None)
        world_points = first_valid["tier_m_frame"]["world_points_sample"] if first_valid else []
        ordering = event_ordering_from_geometry(
            metrics, transforms, np.asarray(center["K"], dtype=float), world_points,
            direction, int(center["sensor_config"]["width"]), event_thresholds)
        state_by_index = [row["state"] for row in ordering["observations"]]
        for entry, metric, state in zip(entries, metrics, state_by_index):
            csv_rows.append({"direction": direction, "frame_id": entry["frame_id"],
                             "capture_role": entry["capture_role"], "computed_state": state,
                             "target_coverage": metric["target_coverage"],
                             "contour_present": metric["contour_present"],
                             "span_over_target_bbox_height": metric["span_over_target_bbox_height"],
                             "boundary_type": metric["boundary_classification"]["boundary_type"],
                             "action_axis_median_m": metric["tier_m_frame"]["action_axis_median_m"]})
        directions[direction] = {
            "frame_count": len(entries), "frames": metrics,
            "computed_states": state_by_index,
            "frozen_pose_group": frozen, "event_ordering": ordering,
            "boundary_consensus": consensus, "tier_v": tier_v,
            "same_pose": same_pose, "same_pose_world_repeatability": tier_m,
        }
    pairing = all(_act0r2_pairing(row) for row in frames)
    hashes = verify_manifest_hashes(frames, PROJECT_ROOT)
    center_sha = _act0r2_pose_sha(center["T_world_camera"])
    bilateral_same_start = (len([row for row in frames if row["direction"] == "CENTER"]) == 1 and
                            all(row.get("shared_center_frame_id") == center["frame_id"] and
                                row.get("shared_center_pose_sha256") == center_sha
                                for row in frames if row["direction"] != "CENTER"))
    side_pass = {}
    for direction in ("LEFT", "RIGHT"):
        row = directions[direction]
        side_pass[direction] = (
            row["event_ordering"]["status"] == "PASS" and
            row["boundary_consensus"].get("boundary_type") == "PHYSICAL_TERMINATION" and
            row["tier_v"]["status"] == "PASS" and
            row["same_pose"]["status"] == "PASS" and
            row["same_pose_world_repeatability"]["status"] == "PASS")
    same_pose_all = all(directions[d]["same_pose"]["status"] == "PASS" for d in directions)
    world_repeat_all = all(directions[d]["same_pose_world_repeatability"]["status"] == "PASS"
                           for d in directions)
    ready = (alignment["status"] == "PASS" and pairing and bilateral_same_start and
             all(center_absent.values()) and all(side_pass.values()))
    gates = {
        "CHECKPOINT_POSE_ALIGNMENT": alignment,
        "SENSOR_PAIRING": {"status": "PASS" if pairing else "FAIL", "frame_count": len(frames)},
        "BILATERAL_SAME_START": {"status": "PASS" if bilateral_same_start else "FAIL",
                                  "shared_center_frame_id": center["frame_id"]},
        "CENTER_LEFT_BOUNDARY_ABSENT": {"status": "PASS" if center_absent["LEFT"] else "FAIL"},
        "CENTER_RIGHT_BOUNDARY_ABSENT": {"status": "PASS" if center_absent["RIGHT"] else "FAIL"},
        "LEFT_EVENT_ORDERING": {"status": directions["LEFT"]["event_ordering"]["status"]},
        "RIGHT_EVENT_ORDERING": {"status": directions["RIGHT"]["event_ordering"]["status"]},
        "LEFT_PHYSICAL_TERMINATION": {"status": "PASS" if directions["LEFT"]["boundary_consensus"].get("boundary_type") == "PHYSICAL_TERMINATION" else "FAIL"},
        "RIGHT_PHYSICAL_TERMINATION": {"status": "PASS" if directions["RIGHT"]["boundary_consensus"].get("boundary_type") == "PHYSICAL_TERMINATION" else "FAIL"},
        "LEFT_TIER_V": {"status": directions["LEFT"]["tier_v"]["status"]},
        "RIGHT_TIER_V": {"status": directions["RIGHT"]["tier_v"]["status"]},
        "SAME_POSE_CONFIRMATION": {"status": "PASS" if same_pose_all else "FAIL"},
        "SAME_POSE_WORLD_BOUNDARY_REPEATABILITY": {"status": "PASS" if world_repeat_all else "FAIL"},
        "MULTIVIEW_REPEATABILITY": {"status": "NOT_EVALUATED"},
        "EXTERNAL_VISUAL_REVIEW": {"status": "PENDING"},
        "READY_FOR_NEXT_SURFACE": {"status": "CONDITIONAL_PASS" if ready else "FAIL"},
        "READY_FOR_COUNTERFACTUAL_ROLLOUT": {"status": "NOT_EVALUATED"},
        "READY_FOR_JEPA": {"status": "NOT_EVALUATED"},
    }
    raw_root = PROJECT_ROOT / args.act0r2_raw_root
    validation = {
        "schema": "act0r2.validation.v1", "candidate_index": 1,
        "checkpoint_plan": plan, "frame_count": len(frames),
        "raw": {"path": args.act0r2_raw_root,
                "file_count": sum(1 for path in raw_root.rglob("*") if path.is_file()),
                "size_bytes": sum(path.stat().st_size for path in raw_root.rglob("*") if path.is_file()),
                "hash_audit": hashes},
        "center_metrics": center_metrics, "directions": directions, "gates": gates,
        "thresholds": {"contour": act0r_config["contour"],
                       "boundary_classification": act0r_config["boundary_classification"],
                       "tier_m": act0r_config["tier_m"],
                       "approach_outside_margin_px": event_thresholds["approach_outside_margin_px"]},
        "resources": {**manifest.get("resources", {}),
                      "initial_carla_address_space_limit_bytes": config["act0r2"][
                          "initial_carla_address_space_limit_bytes"],
                      "effective_carla_address_space_limit_bytes": config["act0r2"][
                          "effective_carla_address_space_limit_bytes"],
                      "carla_cpu_set": config["act0r2"]["carla_cpu_set"],
                      "python_cpu_set": config["act0r2"]["python_cpu_set"]},
        "constraints": {"roles_used_as_labels": False, "rollout_run": False,
                        "other_candidates_captured": False, "jepa_training_run": False,
                        "models_downloaded": False, "multiview_repeatability_claimed": False},
    }
    (result_root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    with (result_root / "frame_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(csv_rows)
    assets = Path(args.act0r2_assets_root); assets.mkdir(parents=True, exist_ok=True)
    for direction in ("LEFT", "RIGHT"):
        entries = [row for row in frames if row["direction"] == direction]
        row = directions[direction]
        _act0r2_role_sheet(entries, row["frames"], row["computed_states"], direction,
                           assets / f"{direction.lower()}_all_roles.jpg")
        first = row["event_ordering"]["first_physical_termination_index"]
        if first is not None:
            _act0r2_evidence_sheet(entries[first], direction, row["frames"][first],
                                   assets / f"{direction.lower()}_first_termination.jpg")
    center_image = Image.open(PROJECT_ROOT / center["files"]["rgb"]["path"]).convert("RGB")
    center_canvas = Image.new("RGB", (640, 520), (20, 20, 20)); center_canvas.paste(center_image, (0, 40))
    ImageDraw.Draw(center_canvas).text((8, 10),
        f"shared CENTER frame={center['frame_id']} LEFT absent={center_absent['LEFT']} RIGHT absent={center_absent['RIGHT']}",
        fill=(255, 230, 70))
    center_canvas.save(assets / "center_bilateral_absence.jpg", quality=84, optimize=True, progressive=True)
    comparison = Image.new("RGB", (960, 290), (20, 20, 20)); draw = ImageDraw.Draw(comparison)
    comparison.paste(center_image.resize((320, 240)), (0, 40)); draw.text((8, 10), "one shared CENTER", fill=(255, 230, 70))
    for index, direction in enumerate(("LEFT", "RIGHT"), start=1):
        first = directions[direction]["event_ordering"]["first_physical_termination_index"]
        entries = [row for row in frames if row["direction"] == direction]
        if first is not None:
            comparison.paste(_act0r2_overlay(entries[first], direction,
                directions[direction]["frames"][first]).resize((320, 240)), (index * 320, 40))
        draw.text((index * 320 + 8, 10), f"{direction} first physical termination", fill=(255, 230, 70))
    comparison.save(assets / "same_start_comparison.jpg", quality=84, optimize=True, progressive=True)
    _write_act0r2_docs(validation, PROJECT_ROOT / "docs/ACT0R2_VISUAL_AUDIT.md")
    print(json.dumps({"frame_count": len(frames), "gates": gates,
                      "directions": {key: {"states": value["computed_states"],
                                            "boundary": value["boundary_consensus"],
                                            "tier_v": value["tier_v"]["status"],
                                            "same_pose_spread_m": value["same_pose_world_repeatability"].get("spread_m")}
                                     for key, value in directions.items()}}, indent=2))
    return 0 if ready else 2


def act0r2(args, config, act0r_config, carla, client):
    checkpoint_path = PROJECT_ROOT / config["checkpoint"]["path"]
    if verify_search_plan(checkpoint_path, config["checkpoint"]["sha256"])["status"] != "PASS":
        raise RuntimeError("search checkpoint changed")
    checkpoint = json.loads(checkpoint_path.read_text())
    plan = next(row for row in checkpoint["plans"] if row["candidate_index"] == 1)
    if int(plan["bbox_id"]) != int(config["act0r2"]["bbox_id"]):
        raise RuntimeError("candidate 1 checkpoint bbox changed")
    for direction in ("LEFT", "RIGHT"):
        if len(plan["searches"][direction]["saved_roles"]) != 7:
            raise RuntimeError(f"checkpoint {direction} must contain seven saved roles")
    result_root, raw_root = Path(args.act0r2_result_root), Path(args.act0r2_raw_root)
    result_root.mkdir(parents=True, exist_ok=True)
    if raw_root.exists() and any(raw_root.rglob("*")):
        raise RuntimeError("ACT-0R2 raw output is non-empty")
    raw_root.mkdir(parents=True, exist_ok=True)
    trace = TraceWriter(raw_root / "timing_trace.jsonl",
                        config["resources"]["python_rss_watchdog_bytes"], args.carla_pid)
    world = client.get_world()
    if act0r_config["map"] not in world.get_map().name:
        raise RuntimeError(f"ACT-0R2 requires {act0r_config['map']}, got {world.get_map().name}")
    center_matrix = np.asarray(plan["locator_center_pose"], dtype=float)
    center_transform = transform_from_matrix(carla, center_matrix)
    center_pose_sha = _act0r2_pose_sha(center_matrix)
    sensor = config["sensor"]
    frames = []
    rig = SynchronousRGBDSeg(world, carla, center_transform, sensor["width"], sensor["height"],
                            sensor["horizontal_fov_deg"], sensor["fixed_delta_seconds"],
                            trace_hook=trace)
    try:
        warmup = rig.warmup(config["warmup"]["minimum_discarded_ticks"],
                            config["warmup"]["required_consecutive_complete"],
                            config["warmup"]["maximum_ticks"])
        center_sample = _attempt_capture(rig, {"sequence_id": "act0r2_candidate_01_shared_center",
                                               "capture_role": "CENTER",
                                               "role_used_as_scientific_label": False}, trace)
        center_meta = _save_act0r2_frame(
            rig, center_sample, raw_root, "center_00_shared", plan, "CENTER", "CENTER", 0.0,
            center_sample["T_world_camera"], center_sample["frame_id"],
            _act0r2_pose_sha(center_sample["T_world_camera"]), {"discarded_frames": []})
        frames.append(center_meta)
        alignment = checkpoint_pose_alignment(
            center_meta["T_world_camera"], center_matrix,
            config["act0r2"]["center_max_position_error_m"],
            config["act0r2"]["center_max_rotation_error_deg"])
        center_metrics = {direction: _act0r2_frame_metric(
            center_meta, direction, plan["camera_motion_axis"], act0r_config)
                          for direction in ("LEFT", "RIGHT")}
        invalid_center = alignment["status"] != "PASS" or any(
            _act0r2_center_boundary_present(center_metrics[d], act0r_config)
            for d in ("LEFT", "RIGHT"))
        if invalid_center:
            manifest = {"schema": "act0r2.capture_manifest.v1", "status": "STOPPED_AT_CENTER",
                        "checkpoint_plan": plan, "frames": frames, "warmup": warmup,
                        "center_alignment": alignment, "center_metrics": center_metrics,
                        "resources": trace.summary()}
            (result_root / "capture_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            left_absent = not _act0r2_center_boundary_present(center_metrics["LEFT"], act0r_config)
            right_absent = not _act0r2_center_boundary_present(center_metrics["RIGHT"], act0r_config)
            gates = {
                "CHECKPOINT_POSE_ALIGNMENT": alignment,
                "SENSOR_PAIRING": {"status": "PASS" if _act0r2_pairing(center_meta) else "FAIL"},
                "BILATERAL_SAME_START": {"status": "FAIL", "reason": "stopped at CENTER"},
                "CENTER_LEFT_BOUNDARY_ABSENT": {"status": "PASS" if left_absent else "FAIL"},
                "CENTER_RIGHT_BOUNDARY_ABSENT": {"status": "PASS" if right_absent else "FAIL"},
                "LEFT_EVENT_ORDERING": {"status": "FAIL", "reason": "not captured"},
                "RIGHT_EVENT_ORDERING": {"status": "FAIL", "reason": "not captured"},
                "LEFT_PHYSICAL_TERMINATION": {"status": "FAIL", "reason": "not captured"},
                "RIGHT_PHYSICAL_TERMINATION": {"status": "FAIL", "reason": "not captured"},
                "LEFT_TIER_V": {"status": "FAIL", "reason": "not captured"},
                "RIGHT_TIER_V": {"status": "FAIL", "reason": "not captured"},
                "SAME_POSE_CONFIRMATION": {"status": "FAIL", "reason": "not captured"},
                "SAME_POSE_WORLD_BOUNDARY_REPEATABILITY": {"status": "FAIL", "reason": "not captured"},
                "MULTIVIEW_REPEATABILITY": {"status": "NOT_EVALUATED"},
                "EXTERNAL_VISUAL_REVIEW": {"status": "PENDING"},
                "READY_FOR_NEXT_SURFACE": {"status": "FAIL"},
                "READY_FOR_COUNTERFACTUAL_ROLLOUT": {"status": "NOT_EVALUATED"},
                "READY_FOR_JEPA": {"status": "NOT_EVALUATED"},
            }
            validation = {"schema": "act0r2.validation.v1", "status": "STOPPED_AT_CENTER",
                          "candidate_index": 1, "frame_count": 1,
                          "checkpoint_plan": plan, "center_metrics": center_metrics,
                          "directions": {}, "gates": gates,
                          "constraints": {"roles_used_as_labels": False, "rollout_run": False,
                                          "other_candidates_captured": False,
                                          "jepa_training_run": False}}
            (result_root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
            assets = Path(args.act0r2_assets_root); assets.mkdir(parents=True, exist_ok=True)
            image = Image.open(PROJECT_ROOT / center_meta["files"]["rgb"]["path"]).convert("RGB")
            canvas = Image.new("RGB", (640, 520), (20, 20, 20)); canvas.paste(image, (0, 40))
            ImageDraw.Draw(canvas).text((8, 10),
                f"ACT-0R2 STOPPED AT CENTER left_absent={left_absent} right_absent={right_absent}",
                fill=(255, 230, 70))
            canvas.save(assets / "center_bilateral_absence.jpg", quality=84, optimize=True,
                        progressive=True)
            (PROJECT_ROOT / "docs/ACT0R2_VISUAL_AUDIT.md").write_text(
                "# ACT-0R2 Visual Audit\n\nCapture stopped at CENTER because a mandatory "
                "alignment or bilateral boundary-absence gate failed. Role names were not "
                "used as labels.\n\n![CENTER evidence](assets/act0r2/center_bilateral_absence.jpg)\n")
            print(json.dumps({"status": "STOPPED_AT_CENTER", "gates": gates}, indent=2))
            return 2
        center_actual = np.asarray(center_meta["T_world_camera"], dtype=float)
        shared_sha = _act0r2_pose_sha(center_actual)
        for direction in ("LEFT", "RIGHT"):
            for index, role in enumerate(plan["searches"][direction]["saved_roles"]):
                enforce_saved_frame_limit(len(frames) + 1, config["act0r2"]["maximum_saved_frames"])
                commanded = _act0r2_pose(center_matrix, plan["camera_motion_axis"],
                                         role["displacement_m"], direction)
                rig.set_transform(transform_from_matrix(carla, commanded))
                settle = rig.settle(config["teleport"]["settle_ticks"])
                sample = _attempt_capture(rig, {
                    "sequence_id": f"act0r2_candidate_01_{direction.lower()}",
                    "capture_role": role["role"], "role_used_as_scientific_label": False,
                    "commanded_displacement_m": role["displacement_m"],
                    "checkpoint_pose_source": "locator_center_pose + camera_motion_axis",
                }, trace)
                stem = f"{direction.lower()}_{index:02d}_{role['role'].lower()}"
                frames.append(_save_act0r2_frame(
                    rig, sample, raw_root, stem, plan, direction, role["role"],
                    role["displacement_m"], center_actual, center_meta["frame_id"], shared_sha, settle))
                print(json.dumps({"phase": "ACT0R2_SAVE", "direction": direction,
                                  "role": role["role"], "frame_id": sample["frame_id"],
                                  "saved_count": len(frames)}), flush=True)
        trace.assert_rss()
    finally:
        rig.close()
    resource_summary = trace.summary()
    resource_summary.pop("gpu_processes_at_end", None)
    manifest = {
        "schema": "act0r2.capture_manifest.v1", "status": "CAPTURE_COMPLETE",
        "checkpoint_path": config["checkpoint"]["path"],
        "checkpoint_sha256": config["checkpoint"]["sha256"],
        "checkpoint_plan": plan, "warmup": warmup, "frames": frames,
        "saved_frame_count": len(frames), "maximum_saved_frames": config["act0r2"]["maximum_saved_frames"],
        "resources": {**resource_summary,
                      "configured_python_address_space_limit_bytes": config["resources"]["python_address_space_limit_bytes"],
                      "actual_python_address_space_limit_bytes": args.actual_as_limit},
        "constraints": {"poses_old_new_used": False, "rollout_run": False,
                        "other_candidates_captured": False, "jepa_training_run": False},
    }
    (result_root / "capture_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return postprocess_act0r2(args, config, act0r_config)


def postprocess(args, config):
    """Recompute compact gates and figures from persisted bytes without CARLA."""
    cap0_path = Path(args.result_root) / "validation.json"
    cap0 = json.loads(cap0_path.read_text())
    probe_path = Path(args.result_root) / "as2_probe.json"
    if probe_path.exists():
        probe = json.loads(probe_path.read_text())
        root_cause = classify_root_cause(cap0.get("tests", {}), probe)
        cap0["address_space_probe"] = probe
        cap0["root_cause"] = root_cause
        cap0["gates"]["ROOT_CAUSE_CLASSIFIED"] = root_cause
        cap0["ACT0R1_AUTHORIZED"] = should_run_act0r1(cap0["gates"])
        cap0_path.write_text(json.dumps(cap0, indent=2) + "\n")
        (Path(args.result_root) / "root_cause.json").write_text(
            json.dumps(root_cause, indent=2) + "\n")

    validation_path = Path(args.act0r1_result_root) / "validation.json"
    validation = json.loads(validation_path.read_text())
    frames = validation["frames"]
    images = [_read_rgb(frame)[0] for frame in frames]
    repeat_indices = [index for index, row in enumerate(frames)
                      if row["capture_role"].startswith("STRADDLE")]
    integrity = evaluate_motion_sequence_rgb(images, [repeat_indices], config["visual"])
    gates = validation["gates"]
    gates["RGB_VISUAL_INTEGRITY"] = integrity
    prerequisite_names = ("SENSOR_QUADRUPLET_PAIRING", "SAME_POSE_CONFIRMATION",
                          "TARGET_INSTANCE_STABILITY")
    complete = len(frames) == 8 and integrity["status"] == "PASS" and all(
        gates.get(name, {}).get("status") == "PASS" for name in prerequisite_names)
    gates["CANDIDATE1_LEFT_CAPTURE_COMPLETE"] = {
        "status": "PASS" if complete else "FAIL", "frames": len(frames)}
    gates["READY_TO_RESUME_ACT0R"] = {"status": "PASS" if complete else "FAIL"}
    validation["postprocess"] = {
        "source": "persisted raw BGRA and metadata",
        "carla_started": False,
        "cross_pose_ssim_is_diagnostic_only": True,
    }
    validation_path.write_text(json.dumps(validation, indent=2) + "\n")
    assets_root = Path(args.act0r1_assets_root)
    _role_contact_sheet(frames, assets_root / "left_all_roles.jpg")
    _raw_png_panel(frames[3], assets_root / "straddle_raw_vs_png.jpg")
    _sensor_panel(frames[3], assets_root / "straddle_sensors.jpg")
    print(json.dumps({
        "root_cause": cap0.get("root_cause"),
        "frames": len(frames),
        "gates": {name: row.get("status") for name, row in gates.items()},
    }, indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("diagnose", "as2-probe", "act0r1", "act0r2",
                                         "act0r2-postprocess", "postprocess"))
    parser.add_argument("--config", default="configs/experiments/cap0.yaml")
    parser.add_argument("--act0r-config", default="configs/experiments/act0r.yaml")
    parser.add_argument("--carla-root")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--carla-pid", type=int)
    parser.add_argument("--actual-as-limit", type=int, default=4294967296)
    parser.add_argument("--result-root", default="results/cap0")
    parser.add_argument("--raw-root", default="results/cap0/raw")
    parser.add_argument("--assets-root", default="docs/assets/cap0")
    parser.add_argument("--act0r1-result-root", default="results/act0r1")
    parser.add_argument("--act0r1-raw-root", default="results/act0r1/raw")
    parser.add_argument("--act0r1-assets-root", default="docs/assets/act0r1")
    parser.add_argument("--act0r2-result-root", default="results/act0r2")
    parser.add_argument("--act0r2-raw-root", default="results/act0r2/raw")
    parser.add_argument("--act0r2-assets-root", default="docs/assets/act0r2")
    args = parser.parse_args(argv)
    config = yaml.safe_load((PROJECT_ROOT / args.config).read_text())
    act0r_config = yaml.safe_load((PROJECT_ROOT / args.act0r_config).read_text())
    if args.mode == "postprocess":
        return postprocess(args, config)
    if args.mode == "act0r2-postprocess":
        return postprocess_act0r2(args, config, act0r_config)
    root = discover_carla_root(args.carla_root)
    carla = import_carla(str(root))
    client = carla.Client(args.host or config["server"]["host"],
                          args.port or int(config["server"]["port"]))
    client.set_timeout(min(float(config["server"]["client_timeout_s"]), 10.0))
    if args.mode == "diagnose":
        return diagnose(args, config, carla, client)
    if args.mode == "as2-probe":
        return as2_probe(args, config, carla, client)
    if args.mode == "act0r2":
        return act0r2(args, config, act0r_config, carla, client)
    return act0r1(args, config, carla, client)


if __name__ == "__main__":
    raise SystemExit(main())
