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
