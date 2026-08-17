# Cleanup Manifest

This manifest was generated before cleanup. Paths are relative to the project root.

Pre-cleanup disk usage: 18430413740 bytes

| Path | Size (bytes) | Class | Reason |
|---|---:|---|---|
| `.carla_pkg` | 12061853 | REVIEW_REQUIRED | unclassified; do not delete without dependency check |
| `.pytest_cache` | 13629 | REVIEW_REQUIRED | unclassified; do not delete without dependency check |
| `.venv` | 24680 | REVIEW_REQUIRED | unclassified; do not delete without dependency check |
| `GEO-0.5R2_audit_20260815` | 4261987008 | QUARANTINE | reproducible audit archive or extracted duplicate |
| `GEO-0.5R2_audit_20260815.tar.gz` | 1228818615 | QUARANTINE | reproducible audit archive or extracted duplicate |
| `GEO-0.5R2_minimal_20260815` | 21349180 | QUARANTINE | reproducible audit archive or extracted duplicate |
| `GEO-0.5R2_minimal_20260815.tar.gz` | 5437844 | QUARANTINE | reproducible audit archive or extracted duplicate |
| `GEO-0.5_audit_20260815.tar.gz` | 36595591 | QUARANTINE | reproducible audit archive or extracted duplicate |
| `GEO05_AUDIT_NOTES.md` | 1907 | KEEP | project documentation |
| `README.md` | 6934 | KEEP | project documentation |
| `boundary_sweep` | 132158 | KEEP | active public source/config/test directory |
| `configs` | 4599 | KEEP | active public source/config/test directory |
| `data` | 12863775129 | REVIEW_REQUIRED | unclassified; do not delete without dependency check |
| `debug_projection.py` | 958 | QUARANTINE | one-off diagnostic script |
| `scripts` | 173083 | KEEP | active public source/config/test directory |
| `tests` | 26476 | KEEP | active public source/config/test directory |

No deletion is authorized by this manifest alone. Items marked REVIEW_REQUIRED remain until dependency checks and public-subset validation finish.

## MASK-0/MASK-1 follow-up disposition

The existing MASK-0 canonical raw audit remains at `results/mask0/raw` (87,780,770 bytes) for external review and R1 recomputation. MASK-1 retained one canonical pilot raw quartet at `results/mask1/raw` (76,161,702 bytes), outside Git tracking. Three superseded MASK-1 raw attempts were moved to the desktop trash with `gio trash` after the final pilot was validated. Their byte-level pre-quarantine sizes were not captured, so those size fields remain unverified in `results/cleanup_report.json`.

## ACT-0S follow-up disposition

ACT-0S analyzed the retained `results/act0/scout_manifest.json` and compressed
center-view evidence. The original scout did not persist raw RGB/depth/semantic/
instance arrays: the raw sensor-data path is absent and its retained size is
`0 bytes`. The manifest itself is `96,717 bytes`; it retains 36 synchronized
quartet frame/timestamp/pose records.

The ACT-0S run retained `57,194 bytes` of new JSON/CSV and `2,360,066 bytes` of
compressed public JPG evidence. After validation, regenerated caches and two
unreferenced candidate-15 images were moved to the desktop trash with
`gio trash` (total measured pre-trash size `488,506 bytes`). No raw data,
historical result, or Git-tracked file was removed. The measured project size
after this cleanup is `209,918,831 bytes`, excluding `.git` metadata.
