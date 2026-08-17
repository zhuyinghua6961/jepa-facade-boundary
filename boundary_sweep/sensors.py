"""Synchronous RGB-D capture with frame-id pairing and reproducible metadata."""

from __future__ import annotations

import hashlib
import json
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

from .carla_utils import transform_to_dict
from .geometry import intrinsics_from_fov, transform_matrix
from .segmentation import bgra_array, rgb_from_bgra


def decode_carla_depth(image) -> np.ndarray:
    """Decode CARLA's 24-bit RGB depth encoding to meters (0..1000m)."""
    data = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    # raw_data is BGRA; the formula is documented in RGB order.
    b, g, r = data[..., 0].astype(np.uint32), data[..., 1].astype(np.uint32), data[..., 2].astype(np.uint32)
    normalized = (r + g * 256 + b * 256 * 256) / float(256 ** 3 - 1)
    return (normalized * 1000.0).astype(np.float32)


def depth_bytes(image) -> bytes:
    return bytes(image.raw_data)


def _se3_delta(previous, current) -> np.ndarray:
    if previous is None:
        return np.eye(4, dtype=np.float64)
    return transform_matrix(current) @ np.linalg.inv(transform_matrix(previous))


class SynchronousRGBD:
    """Own two sensors and restore world settings on close."""

    def __init__(self, world, carla, transform, width=640, height=480, fov=90.0,
                 fixed_delta_seconds=0.05, sensor_tick=0.0, output_dir: Optional[str] = None):
        self.world, self.carla = world, carla
        self.width, self.height, self.fov = int(width), int(height), float(fov)
        self.K = intrinsics_from_fov(self.width, self.height, self.fov)
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self._old_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(fixed_delta_seconds)
        world.apply_settings(settings)
        self._rgb_queue, self._depth_queue = queue.Queue(), queue.Queue()
        self._pending_rgb, self._pending_depth = {}, {}
        self._previous_transform = None
        self._actors = []
        bp = world.get_blueprint_library()
        rgb_bp = bp.find("sensor.camera.rgb")
        depth_bp = bp.find("sensor.camera.depth")
        for sensor_bp in (rgb_bp, depth_bp):
            sensor_bp.set_attribute("image_size_x", str(self.width))
            sensor_bp.set_attribute("image_size_y", str(self.height))
            sensor_bp.set_attribute("fov", str(self.fov))
            sensor_bp.set_attribute("sensor_tick", str(sensor_tick))
        self.rgb = world.spawn_actor(rgb_bp, transform)
        self.depth = world.spawn_actor(depth_bp, transform)
        self._actors.extend([self.rgb, self.depth])
        self.rgb.listen(self._rgb_queue.put)
        self.depth.listen(self._depth_queue.put)

    @staticmethod
    def _get_frame(q, pending, frame_id, timeout):
        if frame_id in pending:
            return pending.pop(frame_id)
        while True:
            data = q.get(timeout=timeout)
            if int(data.frame) == int(frame_id):
                return data
            pending[int(data.frame)] = data

    def capture(self, commanded_action=None, timeout=10.0):
        frame_id = int(self.world.tick())
        rgb = self._get_frame(self._rgb_queue, self._pending_rgb, frame_id, timeout)
        depth = self._get_frame(self._depth_queue, self._pending_depth, frame_id, timeout)
        if int(rgb.frame) != int(depth.frame):
            raise RuntimeError("RGB/depth frame-id mismatch")
        current_transform = rgb.transform
        result = {
            "frame_id": int(rgb.frame),
            "timestamp": float(rgb.timestamp),
            "rgb": rgb,
            "depth": depth,
            "depth_m": decode_carla_depth(depth),
            "K": self.K.copy(),
            "T_world_camera": transform_matrix(current_transform),
            "camera_transform": current_transform,
            "commanded_action": commanded_action or {},
            "executed_delta_pose": _se3_delta(self._previous_transform, current_transform),
        }
        self._previous_transform = current_transform
        return result

    def set_transform(self, transform):
        """Move both sensors to the same commanded pose before the next tick."""
        self.rgb.set_transform(transform)
        self.depth.set_transform(transform)

    def save(self, sample: dict, stem: Optional[str] = None, output_dir: Optional[str | Path] = None) -> dict:
        directory = Path(output_dir) if output_dir is not None else self.output_dir
        if directory is None:
            raise ValueError("output_dir was not configured")
        directory.mkdir(parents=True, exist_ok=True)
        frame_id = int(sample["frame_id"])
        stem = stem or f"frame_{frame_id:06d}"
        rgb = sample["rgb"]
        rgba = np.frombuffer(rgb.raw_data, dtype=np.uint8).reshape((rgb.height, rgb.width, 4))
        Image.fromarray(rgba[..., [2, 1, 0]], mode="RGB").save(directory / f"{stem}_rgb.png")
        (directory / f"{stem}_depth.raw.bin").write_bytes(depth_bytes(sample["depth"]))
        np.save(directory / f"{stem}_depth_m.npy", sample["depth_m"].astype(np.float32))
        metadata = {
            "frame_id": frame_id,
            "timestamp": sample["timestamp"],
            "K": sample["K"].tolist(),
            "T_world_camera": sample["T_world_camera"].tolist(),
            "camera_transform": transform_to_dict(sample["camera_transform"]),
            "commanded_action": sample["commanded_action"],
            "executed_delta_pose": sample["executed_delta_pose"].tolist(),
            "depth_encoding": "CARLA_24bit_RGB_normalized_times_1000m",
        }
        (directory / f"{stem}.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return metadata

    def close(self):
        for actor in reversed(self._actors):
            try:
                actor.stop()
            except Exception:
                pass
            try:
                actor.destroy()
            except Exception:
                pass
        self._actors.clear()
        self.world.apply_settings(self._old_settings)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


@dataclass(frozen=True)
class OwnedSensorFrame:
    """Callback-owned copy of a CARLA sensor frame."""

    frame: int
    timestamp: float
    transform_dict: dict
    T_world_camera: np.ndarray
    width: int
    height: int
    fov: float
    raw_data: bytes


class SensorFrameError(RuntimeError):
    pass


class FrameSkippedError(SensorFrameError):
    pass


class SensorFrameIncompleteError(SensorFrameError):
    pass


class ConsecutiveIncompleteFramesError(SensorFrameError):
    pass


def _se3_delta_matrices(previous, current) -> np.ndarray:
    if previous is None:
        return np.eye(4, dtype=np.float64)
    return np.asarray(current, dtype=float) @ np.linalg.inv(np.asarray(previous, dtype=float))


class SynchronousRGBDSeg:
    """Fail-fast synchronous camera rig with callback-owned raw buffers."""

    DEFAULT_SENSOR_TYPES = ("rgb", "depth", "semantic", "instance")
    BLUEPRINTS = {"rgb": "sensor.camera.rgb", "depth": "sensor.camera.depth",
                  "semantic": "sensor.camera.semantic_segmentation",
                  "instance": "sensor.camera.instance_segmentation"}

    def __init__(self, world, carla, transform, width=640, height=480, fov=90.0,
                 fixed_delta_seconds=0.05, sensor_tick=0.0, sensor_types=None,
                 trace_hook: Optional[Callable[[dict], None]] = None,
                 postprocess_effects: bool = True):
        self.world, self.carla = world, carla
        self.width, self.height, self.fov = int(width), int(height), float(fov)
        self.SENSOR_TYPES = tuple(sensor_types or self.DEFAULT_SENSOR_TYPES)
        unknown = set(self.SENSOR_TYPES) - set(self.BLUEPRINTS)
        if unknown or not self.SENSOR_TYPES:
            raise ValueError(f"invalid sensor types: {sorted(unknown)}")
        self.K = intrinsics_from_fov(self.width, self.height, self.fov)
        self._trace_hook = trace_hook
        self._old_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(fixed_delta_seconds)
        world.apply_settings(settings)
        library = world.get_blueprint_library()
        self._queues = {name: queue.Queue() for name in self.SENSOR_TYPES}
        self._pending = {name: {} for name in self.SENSOR_TYPES}
        self._actors, self.sensors = [], {}
        for name in self.SENSOR_TYPES:
            bp = library.find(self.BLUEPRINTS[name])
            for key, value in (("image_size_x", self.width), ("image_size_y", self.height),
                               ("fov", self.fov), ("sensor_tick", sensor_tick)):
                bp.set_attribute(key, str(value))
            if name == "rgb" and bp.has_attribute("enable_postprocess_effects"):
                bp.set_attribute("enable_postprocess_effects", "true" if postprocess_effects else "false")
            actor = world.spawn_actor(bp, transform)
            self.sensors[name] = actor
            self._actors.append(actor)
            actor.listen(lambda image, sensor_name=name: self._copy_callback(sensor_name, image))
        self._previous_matrix = None
        self._consecutive_incomplete = 0

    def _trace(self, event: str, **fields):
        record = {"event": event, "wall_time": time.time(),
                  "monotonic_time": time.monotonic(), **fields}
        if self._trace_hook is not None:
            self._trace_hook(record)

    def _copy_callback(self, name, image):
        started = time.monotonic()
        try:
            raw = bytes(image.raw_data)
            expected = int(image.width) * int(image.height) * 4
            if len(raw) != expected:
                raise ValueError(f"raw length {len(raw)} != {expected}")
            snapshot = OwnedSensorFrame(
                frame=int(image.frame), timestamp=float(image.timestamp),
                transform_dict=transform_to_dict(image.transform),
                T_world_camera=np.asarray(transform_matrix(image.transform), dtype=np.float64).copy(),
                width=int(image.width), height=int(image.height), fov=self.fov, raw_data=raw)
            self._queues[name].put(snapshot)
            self._trace("queue_callback", sensor=name, frame=snapshot.frame,
                        raw_byte_length=len(raw), copy_duration_s=time.monotonic() - started)
        except Exception as exc:
            self._queues[name].put(exc)
            self._trace("queue_callback_error", sensor=name, error=repr(exc),
                        copy_duration_s=time.monotonic() - started)

    @staticmethod
    def _get(queue_obj, pending, frame_id, deadline, sensor_name="unknown", trace=None):
        if frame_id in pending:
            return pending.pop(frame_id)
        while True:
            remaining = float(deadline) - time.monotonic()
            if remaining <= 0:
                raise SensorFrameIncompleteError(f"{sensor_name} deadline waiting for frame {frame_id}")
            try:
                data = queue_obj.get(timeout=remaining)
            except queue.Empty as exc:
                raise SensorFrameIncompleteError(
                    f"{sensor_name} deadline waiting for frame {frame_id}") from exc
            if isinstance(data, Exception):
                raise SensorFrameIncompleteError(f"{sensor_name} callback failed: {data}")
            observed = int(data.frame)
            if trace is not None:
                trace("queue_receive", sensor=sensor_name, target_frame=int(frame_id),
                      observed_frame=observed)
            if observed == int(frame_id):
                return data
            if observed > int(frame_id):
                raise FrameSkippedError(
                    f"FRAME_SKIPPED {sensor_name}: target={frame_id} observed={observed}")
            if trace is not None:
                trace("queue_discard_old", sensor=sensor_name, target_frame=int(frame_id),
                      observed_frame=observed)

    def capture(self, commanded_action=None, timeout=5.0, tick_timeout=5.0):
        queue_timeout = min(max(float(timeout), 0.001), 5.0)
        tick_timeout = min(max(float(tick_timeout), 0.001), 5.0)
        self._trace("world_tick_start", tick_timeout_s=tick_timeout,
                    sensors=list(self.SENSOR_TYPES))
        try:
            frame_id = int(self.world.tick(tick_timeout))
            self._trace("world_tick_end", frame=frame_id)
            deadline = time.monotonic() + queue_timeout
            data = {name: self._get(self._queues[name], self._pending[name], frame_id,
                                    deadline, name, self._trace)
                    for name in self.SENSOR_TYPES}
            frames = {name: int(item.frame) for name, item in data.items()}
            timestamps = {name: float(item.timestamp) for name, item in data.items()}
            if len(set(frames.values())) != 1:
                raise SensorFrameIncompleteError(f"sensor frame-id mismatch: {frames}")
            if max(timestamps.values()) - min(timestamps.values()) > 1e-6:
                raise SensorFrameIncompleteError(f"sensor timestamp mismatch: {timestamps}")
            first = data["rgb"] if "rgb" in data else data[self.SENSOR_TYPES[0]]
            if any(not np.allclose(item.T_world_camera, first.T_world_camera, atol=1e-5)
                   for item in data.values()):
                raise SensorFrameIncompleteError("sensor transform mismatch")
            result = {"frame_id": frame_id, "timestamp": float(first.timestamp), "data": data,
                      "K": self.K.copy(), "T_world_camera": first.T_world_camera.copy(),
                      "camera_transform": dict(first.transform_dict),
                      "commanded_action": commanded_action or {},
                      "executed_delta_pose": _se3_delta_matrices(self._previous_matrix,
                                                                 first.T_world_camera),
                      "sensor_frames": frames, "sensor_timestamps": timestamps}
            if "depth" in data:
                result["depth_m"] = decode_carla_depth(data["depth"])
            self._previous_matrix = first.T_world_camera.copy()
            self._consecutive_incomplete = 0
            self._trace("capture_complete", frame=frame_id, sensors=list(self.SENSOR_TYPES))
            return result
        except Exception as exc:
            self._consecutive_incomplete += 1
            self._trace("capture_incomplete", consecutive=self._consecutive_incomplete,
                        error=repr(exc))
            if self._consecutive_incomplete >= 2:
                raise ConsecutiveIncompleteFramesError(
                    f"two consecutive incomplete frames; last error: {exc}") from exc
            raise

    def clear_buffers(self):
        discarded = {}
        for name, queue_obj in self._queues.items():
            count = 0
            while True:
                try:
                    queue_obj.get_nowait()
                    count += 1
                except queue.Empty:
                    break
            self._pending[name].clear()
            discarded[name] = count
        self._trace("buffers_cleared", discarded=discarded)
        return discarded

    def set_transform(self, transform):
        self._trace("set_transform_start", transform=transform_to_dict(transform))
        for actor in self.sensors.values():
            actor.set_transform(transform)
        self.clear_buffers()
        self._trace("set_transform_end", transform=transform_to_dict(transform))

    def warmup(self, min_discard=5, consecutive_complete=3, max_ticks=12):
        discarded = consecutive = 0
        self._trace("warmup_start", min_discard=int(min_discard),
                    consecutive_complete=int(consecutive_complete), max_ticks=int(max_ticks))
        while discarded < int(min_discard) or consecutive < int(consecutive_complete):
            if discarded >= int(max_ticks):
                raise SensorFrameIncompleteError("GPU warmup did not converge within max_ticks")
            self.capture({"capture_role": "WARMUP_DISCARD"})
            discarded += 1
            consecutive += 1
        self._trace("warmup_complete", discarded=discarded, consecutive_complete=consecutive)
        return {"discarded_frames": discarded, "consecutive_complete": consecutive}

    def settle(self, ticks=3):
        self._trace("settle_start", required_ticks=int(ticks))
        frames = []
        for _index in range(int(ticks)):
            sample = self.capture({"capture_role": "TELEPORT_SETTLE_DISCARD"})
            frames.append(int(sample["frame_id"]))
        self._trace("settle_complete", discarded_frames=frames)
        return {"discarded_frames": frames}

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def save(self, sample: dict, output_dir: str | Path, stem: str) -> dict:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self._trace("save_start", frame=int(sample["frame_id"]), stem=stem)
        data = sample["data"]
        paths = {}
        raw_lengths = {}
        for name, item in data.items():
            expected = int(item.width) * int(item.height) * 4
            if len(item.raw_data) != expected:
                raise ValueError(f"{name} raw length {len(item.raw_data)} != {expected}")
            raw_lengths[name] = len(item.raw_data)
        if "rgb" in data:
            paths["rgb_bgra"] = directory / f"{stem}_rgb.bgra.bin"
            paths["rgb_bgra"].write_bytes(data["rgb"].raw_data)
            paths["rgb"] = directory / f"{stem}_rgb.png"
            Image.fromarray(rgb_from_bgra(bgra_array(data["rgb"])), mode="RGB").save(paths["rgb"])
        if "depth" in data:
            paths["depth_raw"] = directory / f"{stem}_depth.raw.bin"
            paths["depth_raw"].write_bytes(data["depth"].raw_data)
            paths["depth_m"] = directory / f"{stem}_depth_m.npy"
            np.save(paths["depth_m"], sample["depth_m"].astype(np.float32))
        for name in ("semantic", "instance"):
            if name in data:
                paths[name] = directory / f"{stem}_{name}_raw.png"
                Image.fromarray(rgb_from_bgra(bgra_array(data[name])), mode="RGB").save(paths[name])
        metadata = {"frame_id": int(sample["frame_id"]), "timestamp": float(sample["timestamp"]),
                    "sensor_frames": sample["sensor_frames"],
                    "sensor_timestamps": sample["sensor_timestamps"],
                    "raw_byte_lengths": raw_lengths,
                    "K": sample["K"].tolist(),
                    "T_world_camera": sample["T_world_camera"].tolist(),
                    "camera_transform": sample["camera_transform"],
                    "commanded_action": sample["commanded_action"],
                    "executed_delta_pose": sample["executed_delta_pose"].tolist(),
                    "sensor_config": {"width": self.width, "height": self.height,
                                      "horizontal_fov_deg": self.fov,
                                      "fixed_delta_seconds": float(self.world.get_settings().fixed_delta_seconds),
                                      "blueprints": {name: self.sensors[name].type_id for name in self.SENSOR_TYPES},
                                      "raw_buffer_ownership": "bytes copied inside callback",
                                      "raw_channel_order": "BGRA from CARLA; RGB PNG decoded from saved bytes"},
                    "files": {key: {"path": str(path), "sha256": self._sha256(path),
                                    "size_bytes": path.stat().st_size}
                              for key, path in paths.items()}}
        metadata_path = directory / f"{stem}.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        self._trace("save_end", frame=int(sample["frame_id"]), stem=stem,
                    files={key: str(path) for key, path in paths.items()})
        return metadata

    def close(self):
        for actor in reversed(self._actors):
            try:
                actor.stop()
            except Exception:
                pass
            try:
                actor.destroy()
            except Exception:
                pass
        self._actors.clear()
        self.sensors.clear()
        self.world.apply_settings(self._old_settings)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class SynchronousInstanceSeg(SynchronousRGBDSeg):
    """Single callback-owned instance camera for bounded acquisition search."""

    def __init__(self, world, carla, transform, width=640, height=480, fov=90.0,
                 fixed_delta_seconds=0.05, sensor_tick=0.0, trace_hook=None):
        super().__init__(world, carla, transform, width, height, fov,
                         fixed_delta_seconds, sensor_tick, sensor_types=("instance",),
                         trace_hook=trace_hook)
