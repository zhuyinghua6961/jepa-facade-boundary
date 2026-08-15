#!/usr/bin/env python3
"""Safe project-local cleanup with explicit allowlists."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

CACHE_NAMES = {"__pycache__", ".pytest_cache"}
GENERATED_DIRS = {"sweeps", "sweeps_d2", "sweeps_final", "sweeps_final2", "sweeps_final3", "sweeps_geo05", "overlays", "overlays_d2", "overlays_final", "overlays_final2", "overlays_final3", "overlays_geo05", "contact_sheets", "selection", "selection_final", "selection_top", "selection_top2"}
ARCHIVE_PREFIXES = ("GEO-0.5", "GEO05_")

def candidates(root: Path):
    found=[]
    for p in root.iterdir():
        if p.name in CACHE_NAMES or p.name in GENERATED_DIRS or p.name.startswith(ARCHIVE_PREFIXES):
            found.append(p)
    for p in (root / "data").glob("*"):
        if p.name in GENERATED_DIRS:
            found.append(p)
    for p in root.rglob("__pycache__"):
        if "_cleanup_quarantine" not in p.parts:
            found.append(p)
    return sorted(set(found), key=lambda p: str(p))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--dry-run",action="store_true"); ap.add_argument("--apply",action="store_true"); ap.add_argument("--root",default="."); ap.add_argument("--quarantine",default="../_cleanup_quarantine/boundary_sweep_20260816")
    a=ap.parse_args()
    if a.apply == a.dry_run: ap.error("choose exactly one of --dry-run or --apply")
    root=Path(a.root).resolve(); q=Path(a.quarantine).resolve(); items=candidates(root); rows=[]
    for p in items:
        rel=p.relative_to(root); rows.append({"path":str(rel),"action":"MOVE_TO_QUARANTINE","reason":"explicit regenerable cache/archive allowlist"})
        if a.apply:
            target=q/rel
            if target.exists():
                continue
            target.parent.mkdir(parents=True,exist_ok=True); shutil.move(str(p),str(target))
    report={"schema":"boundary_sweep.clean_artifacts.v1","mode":"apply" if a.apply else "dry-run","root":".","quarantine":"../_cleanup_quarantine/boundary_sweep_20260816","items":rows}
    (root/"results").mkdir(exist_ok=True); (root/"results/cleanup_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__ == "__main__": main()
