#!/usr/bin/env python3
"""Build an independent, geometry-only reference set for R2 transitions.

This file never reads labels_v3 output to choose a state.  It uses only the
recorded physical boundary projection and full-image target polygon geometry;
the resulting rows are candidates for later operator visual review.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from boundary_sweep.geometry import world_to_pixel, intrinsics_from_fov
from boundary_sweep.surfaces import load_surface, physical_corners
from types import SimpleNamespace

def transform_obj(d):
    return SimpleNamespace(location=SimpleNamespace(**d["location"]), rotation=SimpleNamespace(**d["rotation"]))

def independent_state(surface, frame, width=640, height=480):
    K = intrinsics_from_fov(width, height, 90.0)
    tr = transform_obj(frame["camera_transform"])
    corners = world_to_pixel(physical_corners(surface), tr, K)
    line = frame["boundary_pixel_line"]
    finite = np.isfinite(corners).all() and np.isfinite(line).all()
    if not finite: return "UNKNOWN", line
    inside = np.any((corners[:,0] >= 0) & (corners[:,0] < width) & (corners[:,1] >= 0) & (corners[:,1] < height))
    boundary_inside = ((0 <= line[:,0]) & (line[:,0] < width) & (0 <= line[:,1]) & (line[:,1] < height)).any()
    if boundary_inside: return "STRADDLE", line.tolist()
    if inside: return "IN", line.tolist()
    return "OUT", line.tolist()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--sweeps",default="data/sweeps_geo05r2"); ap.add_argument("--surfaces",default="data/surfaces_v3"); ap.add_argument("--output",default="results/geo05r2/geometry_reference_audit.json"); ap.add_argument("--min-frames",type=int,default=60)
    a=ap.parse_args(); surfaces={p.stem:load_surface(p) for p in Path(a.surfaces).glob("*.json")}; candidates=[]
    for p in sorted(Path(a.sweeps).glob("**/trajectory.json")):
        traj=json.loads(p.read_text()); frames=traj.get("frames",[])
        for frame in frames:
            l=frame.get("labels",{}); b=l.get("boundary",{}); sid=l.get("surface_id",p.parts[-4])
            if sid not in surfaces: continue
            independent, line=independent_state(surfaces[sid], {"camera_transform":frame["camera_transform"],"boundary_pixel_line":np.asarray(b.get("boundary_pixel_line",[[np.nan,np.nan],[np.nan,np.nan]]),float)})
            candidates.append({"frame_id":frame.get("frame_id"),"sequence_id":frame.get("sequence_id"),"facade_id":sid,"direction":frame.get("commanded_action",{}).get("direction"),"distance_m":frame.get("commanded_action",{}).get("distance_m"),"step_index":frame.get("commanded_action",{}).get("step_index"),"rgb_path":str(p.parent / (f"{frame.get('commanded_action',{}).get('direction','left').lower()}_{int(frame.get('commanded_action',{}).get('step_index',0)):04d}_rgb.png")),"geometry_reference_state":independent,"geometry_reference_boundary_pixel":line,"reference_method":"independent_geometry_projection_pending_operator_visual_review"})
    # Select transition neighborhoods, then deterministically fill to >= min frames.
    chosen=[]
    by={}
    for row in candidates: by.setdefault(row["sequence_id"],[]).append(row)
    for seq, rs in by.items():
        rs.sort(key=lambda x:x.get("step_index",0)); states=[x["geometry_reference_state"] for x in rs]
        for i in range(1,len(rs)):
            if states[i]!=states[i-1]: chosen.extend(rs[max(0,i-3):min(len(rs),i+4)])
    seen={(x["sequence_id"],x["step_index"]) for x in chosen}
    for row in candidates:
        if len(chosen)>=a.min_frames: break
        key=(row["sequence_id"],row["step_index"])
        if key not in seen: chosen.append(row); seen.add(key)
    result={"schema":"geo05r2.geometry_reference_audit.v1","independent_of":"boundary_sweep.labels","geometry_reference_count":len(chosen),"operator_visual_review_completed":False,"operator_visual_review_required":True,"frames":chosen}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({"count":len(chosen),"output":str(out)}))
if __name__=="__main__": main()
