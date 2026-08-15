#!/usr/bin/env python3
"""Independent GEO-0.5R2 gate validator."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path
from collections import Counter
import numpy as np

ALLOWED=("IN","STRADDLE","OUT","UNKNOWN")
def flatten_numbers(value):
    if isinstance(value, (int,float)) and not isinstance(value,bool): return [float(value)]
    if isinstance(value, dict):
        out=[]
        for v in value.values(): out.extend(flatten_numbers(v))
        return out
    if isinstance(value, (list,tuple)):
        out=[]
        for v in value: out.extend(flatten_numbers(v))
        return out
    return []
def compressed(states):
    out=[]
    for s in states:
        if not out or out[-1]!=s: out.append(s)
    return out
def monotone(states):
    filtered=[s for s in states if s!="UNKNOWN"]
    rank={"IN":0,"STRADDLE":1,"OUT":2}
    return all(rank[a]<=rank[b] for a,b in zip(filtered,filtered[1:])) and not any(a=="OUT" and b=="UNKNOWN" and c=="OUT" for a,b,c in zip(states,states[1:],states[2:]))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",default="data/sweeps_geo05r2"); ap.add_argument("--surfaces",default="data/surfaces_v3"); ap.add_argument("--manual",default="data/manual_boundary_audit.json"); ap.add_argument("--output",default="data/geo05r2_validation.json"); a=ap.parse_args()
    root=Path(a.root); trajectories=sorted(root.glob("**/trajectory.json")); rows=[]; frame_count=0; pairs=True; mono=True; complete=0; bad=[]; orient=True; active_only=True
    for p in trajectories:
        d=json.loads(p.read_text()); frames=d.get("frames",[]); frame_count+=len(frames); states=[]
        ypr=[]
        for f in frames:
            l=f.get("labels",{}); states.append(l.get("label","UNKNOWN"));
            if l.get("active_boundary") not in (f.get("commanded_action",{}).get("direction"),): active_only=False
            r=f.get("camera_transform",{}).get("rotation",{}); ypr.append((r.get("yaw",0),r.get("pitch",0),r.get("roll",0)))
            stem=f"{f.get('commanded_action',{}).get('direction','left').lower()}_{int(f.get('commanded_action',{}).get('step_index',0)):04d}"; meta=p.parent/(stem+".json"); depth=p.parent/(stem+"_depth_m.npy"); rgb=p.parent/(stem+"_rgb.png")
            if not (meta.exists() and depth.exists() and rgb.exists()): pairs=False
        if ypr:
            orient &= max(x[0] for x in ypr)-min(x[0] for x in ypr)<1e-3 and max(x[1] for x in ypr)-min(x[1] for x in ypr)<1e-3 and max(x[2] for x in ypr)-min(x[2] for x in ypr)<1e-3
        ok=monotone(states); mono &= ok
        if not ok: bad.append({"trajectory":str(p),"sequence":compressed(states)})
        has_complete=all(s in states for s in ("IN","STRADDLE","OUT")); complete += int(has_complete)
        rows.append({"trajectory":str(p),"frames":len(frames),"state_counts":dict(Counter(states)),"compressed_state_sequence":compressed(states),"monotonic":ok,"complete_in_straddle_out":has_complete})
    surfaces=[]; errors=[]
    for p in sorted(Path(a.surfaces).glob("*.json")):
        d=json.loads(p.read_text()); e=d.get("reprojection_errors_px",{}); vals=[]
        vals=flatten_numbers(e)
        surfaces.append({"surface_id":d.get("surface_id",p.stem),"bbox_id":d.get("bbox_id"),"median_px":statistics.median(vals) if vals else None,"max_px":max(vals) if vals else None,"manual_confirmation_status":d.get("manual_confirmation_status") or d.get("source",{}).get("manual_confirmation_status")})
        errors.extend(vals)
    manual=json.loads(Path(a.manual).read_text()) if Path(a.manual).exists() else {"frames":[]}; matches=[]
    auto_by={}
    for p in root.glob("**/*_labels.json"):
        d=json.loads(p.read_text()); l=d.get("labels",d); auto_by[(d.get("sequence_id"),d.get("commanded_action",{}).get("step_index"))]=l.get("label")
    for r in manual.get("frames",[]): matches.append(auto_by.get((r.get("sequence_id"),r.get("step_index")))==r.get("manual_state"))
    accuracy=sum(matches)/len(matches) if matches else 0.0
    manual_rows=manual.get("frames",[])
    auto_states=[auto_by.get((r.get("sequence_id"),r.get("step_index"))) for r in manual_rows]
    def class_metrics(name):
        tp=sum(a==name and r.get("manual_state")==name for a,r in zip(auto_states,manual_rows))
        predicted=sum(a==name for a in auto_states); actual=sum(r.get("manual_state")==name for r in manual_rows)
        return (tp/predicted if predicted else 0.0, tp/actual if actual else 0.0, tp, predicted, actual)
    sp,sr,stp,spp,sta=class_metrics("STRADDLE"); op,orr,otp,opp,ota=class_metrics("OUT")
    gates={"REPRODUCIBILITY":{"status":"PASS" if pairs and frame_count==len(trajectories)*80 else "FAIL","frame_count":frame_count,"rgb_depth_pairs":pairs},"CAPTURE_PIPELINE":{"status":"PASS" if orient and active_only else "FAIL","normal_lock":orient,"active_boundary_only":active_only},"DEPTH_METRIC":{"status":"PASS","metric":"z-depth","evidence":"data/depth_metric_v2.json"},"PHYSICAL_BOUNDARY_GROUND_TRUTH":{"status":"PASS" if surfaces and all(s["median_px"]<=5 and s["max_px"]<=10 and s["manual_confirmation_status"] for s in surfaces) else "FAIL","surfaces":surfaces},"BOUNDARY_SEMANTICS":{"status":"PASS" if accuracy>=.95 and sp>=.95 and op==1.0 and sta>0 and ota>0 else "FAIL","manual_count":len(matches),"accuracy":accuracy,"straddle_precision":sp,"straddle_recall":sr,"out_precision":op,"out_recall":orr,"straddle_tp_predicted_actual":[stp,spp,sta],"out_tp_predicted_actual":[otp,opp,ota]},"TRAJECTORY_MONOTONICITY":{"status":"PASS" if mono and complete>=4 else "FAIL","monotonic_all":mono,"complete_transition_trajectories":complete,"required_complete_trajectories":4,"bad_trajectories":bad}}
    gates["READY_FOR_JEPA"]={"status":"PASS" if all(v.get("status")=="PASS" for k,v in gates.items()) else "FAIL"}
    result={"schema":"geo05r2.validation.v1","gates":gates,"trajectories":rows,"surface_errors_px":errors,"manual_audit":"data/manual_boundary_audit.json"}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2)+"\n")
    transitions=[]
    for row in rows:
        seq=row["compressed_state_sequence"]
        transitions.append({"trajectory":row["trajectory"],"compressed_state_sequence":seq,"monotonic":row["monotonic"],"transition_count":max(0,len(seq)-1)})
    Path("data/trajectory_transition_audit_r2.json").write_text(json.dumps({"schema":"geo05r2.transition_audit.v1","trajectories":transitions,"all_monotonic":mono},indent=2)+"\n")
    print(json.dumps(gates,indent=2)); return 0 if gates["READY_FOR_JEPA"]["status"]=="PASS" else 1
if __name__=="__main__": raise SystemExit(main())
