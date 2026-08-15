#!/usr/bin/env python3
"""Create active-boundary-only R2 contact sheets and transition overlays."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

STATES = ("IN", "STRADDLE", "OUT", "UNKNOWN")

def rows(root: Path):
    for p in sorted(root.glob("**/trajectory.json")):
        d = json.loads(p.read_text())
        for frame in d.get("frames", []):
            yield p, frame

def sheet(items, output: Path, title: str, cols=4, tile=(320, 240)):
    if not items:
        return
    font = ImageFont.load_default()
    rows_n = math.ceil(len(items) / cols)
    canvas = Image.new("RGB", (cols * tile[0], rows_n * (tile[1] + 24)), "black")
    draw = ImageDraw.Draw(canvas)
    for i, (img_path, caption) in enumerate(items):
        x, y = (i % cols) * tile[0], (i // cols) * (tile[1] + 24)
        try:
            img = Image.open(img_path).convert("RGB").resize(tile)
            canvas.paste(img, (x, y))
        except Exception:
            continue
        draw.text((x + 3, y + tile[1] + 3), caption, fill="white", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", default="data/sweeps_geo05r2")
    ap.add_argument("--overlays", default="data/overlays_geo05r2")
    ap.add_argument("--output", default="data/contact_sheets_geo05r2")
    args = ap.parse_args()
    out = Path(args.output); overlays = Path(args.overlays)
    grouped = {(sid, dist, direction, state): [] for sid in ("surface_alpha", "surface_beta")
               for dist in ("5m", "10m", "15m") for direction in ("LEFT", "RIGHT") for state in STATES}
    by_seq = {}
    for traj, frame in rows(Path(args.sweeps)):
        labels = frame.get("labels", {})
        state, sid = labels.get("label", "UNKNOWN"), labels.get("surface_id", traj.parts[-4])
        direction, dist = traj.parent.name, traj.parent.parent.name
        stem = Path(frame.get("commanded_action", {}).get("direction", direction.lower()))
        step = int(frame.get("commanded_action", {}).get("step_index", 0))
        name = f"{direction.lower()}_{step:04d}_overlay.png"
        img = overlays / sid / "NORMAL_LOCK" / dist / direction / name
        grouped.setdefault((sid, dist, direction, state), []).append((img, f"{step:04d} {state}"))
        by_seq.setdefault((sid, dist, direction), []).append((step, state, img))
    for key, items in grouped.items():
        sid, dist, direction, state = key
        # Keep all frames for auditability, with deterministic cap for huge sheets.
        sheet(items[:80], out / sid / dist / direction / f"{state}.png", f"{sid} {dist} {direction} {state}")
    # Transition windows: three frames before and after each state transition.
    for (sid, dist, direction), seq in by_seq.items():
        seq.sort(); states = [x[1] for x in seq]
        for i in range(1, len(seq)):
            if states[i] == states[i-1]: continue
            lo, hi = max(0, i-3), min(len(seq), i+4)
            kind = f"{states[i-1]}_to_{states[i]}"
            items = [(seq[j][2], f"{seq[j][0]:04d} {seq[j][1]}") for j in range(lo, hi)]
            sheet(items, out / sid / dist / direction / "transitions" / f"{i:04d}_{kind}.png", f"{sid} {dist} {direction} {kind}", cols=7, tile=(180, 135))
    print(json.dumps({"output": str(out), "sequences": len(by_seq), "groups": len(grouped)}))

if __name__ == "__main__": main()
