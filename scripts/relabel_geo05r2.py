#!/usr/bin/env python3
"""Regenerate R2 labels/overlays from already captured R2 RGB-D frames."""
from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from boundary_sweep.labels import generate_frame_label
from boundary_sweep.surfaces import load_surface


def transform(d):
    return SimpleNamespace(location=SimpleNamespace(**d["location"]), rotation=SimpleNamespace(**d["rotation"]))


def main():
    counts = {}
    for surface_path in sorted(Path("data/surfaces_v3").glob("*.json")):
        surface = load_surface(surface_path)
        counts[surface["surface_id"]] = {}
        root = Path("data/sweeps_geo05r2") / surface["surface_id"] / "NORMAL_LOCK"
        for trajectory in sorted(root.glob("*/*")):
            direction = trajectory.name
            if direction not in ("LEFT", "RIGHT"):
                continue
            records = []
            for label_path in sorted(trajectory.glob("*_labels.json")):
                stem = label_path.name.replace("_labels.json", "")
                record = json.loads(label_path.read_text())
                metadata = json.loads((trajectory / f"{stem}.json").read_text())
                depth = np.load(trajectory / f"{stem}_depth_m.npy")
                overlay = Path("data/overlays_geo05r2") / surface["surface_id"] / "NORMAL_LOCK" / trajectory.parent.name / direction / f"{stem}_overlay.png"
                labels = generate_frame_label(surface, direction, transform(metadata["camera_transform"]),
                                               np.asarray(metadata["K"], dtype=float), depth, 640, 480, 4,
                                               overlay_path=overlay, rgb_path=trajectory / f"{stem}_rgb.png")
                record["labels"] = labels
                label_path.write_text(json.dumps(record, indent=2) + "\n")
                records.append(record)
            states = [r["labels"]["label"] for r in records]
            compressed = []
            for state in states:
                if not compressed or compressed[-1] != state:
                    compressed.append(state)
            summary = {"surface_id": surface["surface_id"], "facade_id": surface["surface_id"],
                       "mode": "NORMAL_LOCK", "distance_m": float(trajectory.parent.name[:-1]),
                       "direction": direction, "frames": len(records),
                       "state_counts": {state: states.count(state) for state in ("IN", "STRADDLE", "OUT", "UNKNOWN")},
                       "compressed_state_sequence": compressed, "orientation_locked": True, "step_m": 0.5}
            (trajectory / "trajectory.json").write_text(json.dumps({"summary": summary, "frames": records}, indent=2) + "\n")
            counts[surface["surface_id"]][f"{trajectory.parent.name}/{direction}"] = summary["state_counts"]
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
