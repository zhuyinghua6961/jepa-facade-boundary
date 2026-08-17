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

from boundary_sweep.cap0 import (HEALTH_GATES, classify_root_cause,
                                 enforce_saved_frame_limit,
                                 evaluate_motion_sequence_rgb,
                                 evaluate_rgb_sequence, sha256_file,
                                 should_run_act0r1, verify_search_plan)
from boundary_sweep.carla_utils import discover_carla_root, import_carla
from boundary_sweep.segmentation import decode_instance_channels
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
    parser.add_argument("mode", choices=("diagnose", "as2-probe", "act0r1", "postprocess"))
    parser.add_argument("--config", default="configs/experiments/cap0.yaml")
    parser.add_argument("--carla-root", required=True)
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
    args = parser.parse_args(argv)
    config = yaml.safe_load((PROJECT_ROOT / args.config).read_text())
    if args.mode == "postprocess":
        return postprocess(args, config)
    root = discover_carla_root(args.carla_root)
    carla = import_carla(str(root))
    client = carla.Client(args.host or config["server"]["host"],
                          args.port or int(config["server"]["port"]))
    client.set_timeout(min(float(config["server"]["client_timeout_s"]), 10.0))
    if args.mode == "diagnose":
        return diagnose(args, config, carla, client)
    if args.mode == "as2-probe":
        return as2_probe(args, config, carla, client)
    return act0r1(args, config, carla, client)


if __name__ == "__main__":
    raise SystemExit(main())
