#!/usr/bin/env python3
"""Validate the compact R2 result set without RGB-D frames or CARLA."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--results",default="results/geo05r2"); a=ap.parse_args()
    root=Path(a.results)
    required=["geo05r2_validation.json","trajectory_transition_audit_r2.json","manual_boundary_audit.json","depth_metric_v2.json","r2_dataset_manifest.json"]
    missing=[x for x in required if not (root/x).exists()]
    validation=json.loads((root/"geo05r2_validation.json").read_text()) if not missing else {}
    depth=json.loads((root/"depth_metric_v2.json").read_text()) if not missing else {}
    manual=json.loads((root/"manual_boundary_audit.json").read_text()) if not missing else {}
    surfaces=list((root/"surfaces_v3").glob("*.json")) if (root/"surfaces_v3").exists() else []
    surface_rows=[json.loads(p.read_text()) for p in surfaces]
    forbidden=[]
    private_markers=("/" + "mnt/fast18/", "/" + "Users/")
    for p in Path(".").rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc" and p.name != "validate_static.py" and ".git" not in p.parts and p.stat().st_size < 2_000_000:
            try: text=p.read_text(errors="ignore")
            except Exception: continue
            if any(marker in text for marker in private_markers): forbidden.append(str(p))
    checks={
      "required_files": not missing,
      "manual_audit_count": int(manual.get("count",len(manual.get("frames",[])))) >= 60,
      "surface_count": len(surface_rows) >= 2,
      "z_depth_pass": bool(depth.get("z_depth_pass")),
      "validation_schema": validation.get("schema")=="geo05r2.validation.v1",
      "no_private_paths": not forbidden,
    }
    result={"schema":"boundary_sweep.static_validation.v1","checks":checks,"missing":missing,"private_path_files":forbidden,"surface_ids":[x.get("surface_id") for x in surface_rows],"manual_audit_count":manual.get("count",len(manual.get("frames",[]))),"z_depth_median_abs_error_m":depth.get("z_depth_median_abs_error_m"),"gates":validation.get("gates",{})}
    print(json.dumps(result,indent=2)); return 0 if all(checks.values()) else 1

if __name__=="__main__": raise SystemExit(main())
