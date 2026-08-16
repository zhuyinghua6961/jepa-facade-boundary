"""Synchronous RGB-D capture with frame-id pairing and reproducible metadata."""

from __future__ import annotations

import hashlib
import json
import queue
from pathlib import Path
from typing import Optional

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


class SynchronousRGBDSeg:
    """Synchronous RGB/depth/semantic/instance camera quartet."""

    SENSOR_TYPES = ("rgb", "depth", "semantic", "instance")

    def __init__(self, world, carla, transform, width=640, height=480, fov=90.0,
                 fixed_delta_seconds=0.05, sensor_tick=0.0):
        self.world, self.carla = world, carla
        self.width, self.height, self.fov = int(width), int(height), float(fov)
        self.K = intrinsics_from_fov(self.width, self.height, self.fov)
        self._old_settings = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(fixed_delta_seconds)
        world.apply_settings(settings)
        names = {"rgb": "sensor.camera.rgb", "depth": "sensor.camera.depth",
                 "semantic": "sensor.camera.semantic_segmentation",
                 "instance": "sensor.camera.instance_segmentation"}
        library = world.get_blueprint_library()
        self._queues = {name: queue.Queue() for name in self.SENSOR_TYPES}
        self._pending = {name: {} for name in self.SENSOR_TYPES}
        self._actors, self.sensors = [], {}
        for name, blueprint_name in names.items():
            bp = library.find(blueprint_name)
            for key, value in (("image_size_x", self.width), ("image_size_y", self.height),
                               ("fov", self.fov), ("sensor_tick", sensor_tick)):
                bp.set_attribute(key, str(value))
            actor = world.spawn_actor(bp, transform)
            self.sensors[name] = actor
            self._actors.append(actor)
            actor.listen(self._queues[name].put)
        self._previous_transform = None

    @staticmethod
    def _get(queue_obj, pending, frame_id, timeout):
        if frame_id in pending:
            return pending.pop(frame_id)
        while True:
            data = queue_obj.get(timeout=timeout)
            if int(data.frame) == int(frame_id):
                return data
            pending[int(data.frame)] = data

    def capture(self, commanded_action=None, timeout=10.0):
        frame_id = int(self.world.tick())
        data = {name: self._get(self._queues[name], self._pending[name], frame_id, timeout)
                for name in self.SENSOR_TYPES}
        frames = {name: int(item.frame) for name, item in data.items()}
        if len(set(frames.values())) != 1:
            raise RuntimeError(f"sensor frame-id mismatch: {frames}")
        timestamps = {name: float(item.timestamp) for name, item in data.items()}
        if max(timestamps.values()) - min(timestamps.values()) > 1e-6:
            raise RuntimeError(f"sensor timestamp mismatch: {timestamps}")
        transform = data["rgb"].transform
        matrices = {name: transform_matrix(item.transform) for name, item in data.items()}
        if any(not np.allclose(value, matrices["rgb"], atol=1e-5) for value in matrices.values()):
            raise RuntimeError("sensor transform mismatch")
        result = {"frame_id": frame_id, "timestamp": timestamps["rgb"], "data": data,
                  "K": self.K.copy(), "T_world_camera": matrices["rgb"],
                  "camera_transform": transform, "commanded_action": commanded_action or {},
                  "executed_delta_pose": _se3_delta(self._previous_transform, transform),
                  "sensor_frames": frames, "sensor_timestamps": timestamps}
        result["depth_m"] = decode_carla_depth(data["depth"])
        self._previous_transform = transform
        return result

    def set_transform(self, transform):
        for actor in self.sensors.values():
            actor.set_transform(transform)

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
        frame_id = int(sample["frame_id"])
        data = sample["data"]
        rgb_arr, semantic_arr, instance_arr = (bgra_array(data[name]) for name in ("rgb", "semantic", "instance"))
        paths = {"rgb": directory / f"{stem}_rgb.png", "depth_raw": directory / f"{stem}_depth.raw.bin",
                 "depth_m": directory / f"{stem}_depth_m.npy", "semantic": directory / f"{stem}_semantic_raw.png",
                 "instance": directory / f"{stem}_instance_raw.png"}
        Image.fromarray(rgb_from_bgra(rgb_arr), mode="RGB").save(paths["rgb"])
        paths["depth_raw"].write_bytes(bytes(data["depth"].raw_data))
        np.save(paths["depth_m"], sample["depth_m"].astype(np.float32))
        Image.fromarray(rgb_from_bgra(semantic_arr), mode="RGB").save(paths["semantic"])
        Image.fromarray(rgb_from_bgra(instance_arr), mode="RGB").save(paths["instance"])
        metadata = {"frame_id": frame_id, "timestamp": float(sample["timestamp"]),
                    "sensor_frames": sample["sensor_frames"], "sensor_timestamps": sample["sensor_timestamps"],
                    "K": sample["K"].tolist(), "T_world_camera": sample["T_world_camera"].tolist(),
                    "camera_transform": transform_to_dict(sample["camera_transform"]),
                    "commanded_action": sample["commanded_action"],
                    "executed_delta_pose": sample["executed_delta_pose"].tolist(),
                    "sensor_config": {"width": self.width, "height": self.height, "horizontal_fov_deg": self.fov,
                                      "fixed_delta_seconds": float(self.world.get_settings().fixed_delta_seconds),
                                      "blueprints": {name: self.sensors[name].type_id for name in self.SENSOR_TYPES},
                                      "raw_channel_order": "BGRA from CARLA; encoded images saved as RGB"},
                    "files": {key: {"path": str(path), "sha256": self._sha256(path)} for key, path in paths.items()}}
        metadata_path = directory / f"{stem}.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
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
