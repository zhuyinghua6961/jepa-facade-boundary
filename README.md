# JEPA Facade Boundary

Reusable CARLA geometry and RGB-D tooling for active facade-boundary perception from an aerial-camera viewpoint. This repository validates the image-boundary-pixel -> 3D-world-coordinate -> facade-coordinate chain; it does not train JEPA.

## Status

The repository contains the GEO-0.5R2 implementation and its reproducible audit. CARLA 0.10.0 UE5 / Town10HD_Opt was used with 640x480 RGB-D sensors, horizontal FOV 90 degrees, synchronous mode, fixed delta 0.05 s, and NORMAL_LOCK camera motion.

```text
pytest: 61 passed
RGB-D: 960 pairs
surface_alpha: bbox 48391
surface_beta: bbox 48393
alpha reprojection median/max: 1.475 / 4.148 px
beta reprojection median/max: 1.379 / 4.118 px
z-depth median absolute error: approximately 0.0017 m
geometry-reference audit candidates: 60 frames
operator visual review: pending
geometry-reference agreement: 23.33%
```

Gate results are intentionally not overstated:

```text
REPRODUCIBILITY: PASS
CAPTURE_PIPELINE: PASS
DEPTH_METRIC: PASS
PHYSICAL_BOUNDARY_GROUND_TRUTH: PASS
GEOMETRY_REFERENCE_CONSISTENCY: FAIL
BOUNDARY_SEMANTICS: FAIL
TRAJECTORY_ORDERING: PASS
EVENT_COVERAGE: FAIL
READY_FOR_JEPA: FAIL
```

The selected facades contain balconies and railings that cause severe occlusion. The run produced `UNKNOWN=490` and `OUT=470`, with no reliable IN/STRADDLE coverage. The 60-frame set is a geometry-reference candidate set, not completed operator visual review; `23.33%` is geometry-reference agreement, not human accuracy. The trajectory order is monotonic, but there are no complete IN -> STRADDLE -> OUT events. JEPA training has not started.

## Layout

```text
boundary_sweep/       reusable geometry, sensors, surfaces and labels modules
scripts/              thin experiment and validation entry points
configs/              configuration files; experiment changes are not source edits
tests/                geometry and label unit tests
docs/                 cleanup and experiment policy
results/geo05r2/      small, reproducible metrics, manifests and figures
```

The physical boundary truth is stored separately from bbox candidates. `surfaces.py` loads plane support points and multi-view fitted `physical_boundary` lines. `labels.py` uses dense depth-aware sampling and separates `occlusion_visibility_ratio` from `target_pixel_coverage`.

## Setup

Use an existing CARLA 0.10.0 Python wheel or environment. No network installation is required.

```bash
cd /path/to/boundary_sweep
export CARLA_ROOT=/path/to/Carla-0.10.0-Linux-Shipping
PYTHONPATH=. python3 scripts/check_carla.py --carla-root "$CARLA_ROOT"
```

The CARLA server must be running on `localhost:2000`. The client restores world settings and destroys its sensors on exit.

## Tests and validation

```bash
cd /path/to/boundary_sweep
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python3 -m pytest -q tests
PYTHONPATH=. python3 scripts/validate_static.py
```

The validator reads the compact result files under `results/geo05r2/`; it does not require the 960-frame RGB-D directory. Depth evidence is read from `results/geo05r2/depth_metric_v2.json` and fails closed when required fields are absent.

For a new capture, use configuration and a unique result directory. The capture implementation is a free-camera rig and does not require a flight plugin:

```bash
PYTHONPATH=. python3 scripts/capture.py \
  --carla-root "$CARLA_ROOT" \
  --surfaces results/geo05r2/surfaces_v3 \
  --distances 5 10 15 --frames-per-direction 80 --step-m 0.5 \
  --output-root results/<run_id>/raw \
  --overlay-root results/<run_id>/figures
```

## Published data policy

The public repository contains source code, tests, configuration, key validation JSON, fitted surface truth, manifests, logs and a small set of transition figures. It intentionally does not contain CARLA installations, wheels, model weights, datasets, full 960-frame RGB-D, RAW/NPY/BIN files, full overlays, cookies, tokens, SSH material, or server-local paths.

The complete GEO-0.5R2 audit archive is not part of the repository. Its SHA-256 is recorded in the audit notes and release manifest:

`12d478035955b54ec39510517131458e08a19db53e27ba10ab0fe29626c7398b`

## GEO-0.6 data-feasibility audit

GEO-0.6 evaluates two new Town10HD_Opt facade candidates with collision-raycast terminal lines and eight NORMAL_LOCK outward sweeps (`LEFT`, `RIGHT`, `UP`, `DOWN`). It is not a JEPA experiment and no training was run.

The audit is published in [`docs/GEO06_VISUAL_AUDIT.md`](docs/GEO06_VISUAL_AUDIT.md), with compressed real RGB audit images under [`docs/assets/geo06/`](docs/assets/geo06/). The result is intentionally a failure audit: sensor pairing and trajectory ordering pass, but only one of the two candidates remains depth-consistent. The second candidate has a `0.731 m` collision-raycast versus z-depth offset, and the selected facade has too much depth-visible occlusion for the `UNKNOWN <= 10%` requirement. Event coverage is `0/8` complete `IN -> STRADDLE -> OUT` tracks.

```text
SCENE_SUITABILITY: FAIL
SENSOR_PAIRING: PASS
PHYSICAL_BOUNDARY_GT: FAIL
BOUNDARY_VISIBILITY: FAIL
TRAJECTORY_ORDERING: PASS
EVENT_COVERAGE: FAIL
OPERATOR_VISUAL_REVIEW: PENDING
READY_FOR_DATASET_EXPANSION: FAIL
READY_FOR_JEPA: NOT_EVALUATED
```

GEO-0.6 raw RGB-D was never tracked in GitHub and, after the failed event-coverage gate was recorded, the `results/geo06/raw` directory was moved to the user trash with `gio trash`. Only compact manifests, frame metadata, failure images and thresholds remain. No GEO-0.5R2 result was rewritten.

## MASK-0 instance-mask feasibility audit

MASK-0 tested 30 key-pose RGB/depth/semantic/instance groups on `surface_omega` and `surface_sigma` in Town10HD_Opt. CARLA `CityObjectLabel.Buildings` decodes to semantic tag `3`; instance IDs were decoded from raw BGRA and grouped only when stable across views. The audit is published in [`docs/MASK0_VISUAL_AUDIT.md`](docs/MASK0_VISUAL_AUDIT.md) with compact RGB/semantic/instance/envelope composites.

The four-sensor pairing, decoder, stable-ID grouping and enclosed-hole envelope checks pass. CENTER is IN on both surfaces and sigma LEFT/RIGHT are STRADDLE, but omega boundary views and sigma TOP remain UNKNOWN; downward AGL raycasts did not establish the required 2 m safety margin. Therefore `READY_FOR_SEQUENCE_RECAPTURE` is FAIL, while `READY_FOR_DATASET_EXPANSION` and `READY_FOR_JEPA` remain NOT_EVALUATED. No JEPA training or full trajectory capture was performed.

## MASK-0R1 corrected boundary audit

MASK-0R1 reuses the existing MASK-0 raw quartet without overwriting the historical result. CARLA fields are decoded separately: semantic tag `R`, 16-bit instance ID `G | (B << 8)`, and packed index key `R | (G << 8) | (B << 16)`. Independent semantic/instance R agreement is `1.000000000` over `9,216,000` pixels.

The old projected-line visibility check is no longer called boundary alignment. An alignment result requires an exterior target/non-target instance contour, directional side evidence, and distance/support thresholds. `surface_sigma` has two validated opposite-direction contour candidates (LEFT and RIGHT), while its legacy lines are 28 px and 51 px from the actual contours. `surface_omega` and sigma TOP/BOTTOM have no validated exterior transition contour. `INTERNAL_HOLE_REJECTION` is `NOT_APPLICABLE` because the raw masks contain no enclosed holes.

The full R1 evidence is in [`docs/MASK0R1_VISUAL_AUDIT.md`](docs/MASK0R1_VISUAL_AUDIT.md) and [`results/mask0/validation_r1.json`](results/mask0/validation_r1.json). R1 authorizes only a small horizontal pilot:

```text
READY_FOR_ADAPTIVE_PILOT: PASS
READY_FOR_DATASET_EXPANSION: NOT_EVALUATED
READY_FOR_JEPA: NOT_EVALUATED
```

## MASK-1 adaptive pilot

MASK-1 ran only on `surface_sigma` at 10 m with NORMAL_LOCK, 0.5 m horizontal steps, and the instance-mask exterior contour as a privileged simulator oracle. LEFT and RIGHT each captured 13 frames. Both trajectories were:

```text
IN x3 -> APPROACH -> STRADDLE x3 -> STOP
```

The pilot stopped after three consecutive directional STRADDLE frames; it did not attempt OUT, TOP, BOTTOM, or underground motion. RGB/depth/semantic/instance frame IDs and timestamps are paired, camera orientation stayed locked, and the recovered boundary points use z-depth back-projection. Surface-horizontal boundary spread is `0.0013 m` for LEFT and `0.0405 m` for RIGHT under a `1.0 m` threshold; vertical spread is reported separately because visible line height changes with the view. Operator visual review is pending; `READY_FOR_JEPA` remains `NOT_EVALUATED`.

See [`docs/MASK1_VISUAL_AUDIT.md`](docs/MASK1_VISUAL_AUDIT.md) and [`results/mask1/validation.json`](results/mask1/validation.json). MASK-1 is a feasibility pilot, not a dataset expansion or JEPA experiment.

## OBS-0R1 leakage-corrected audit

OBS-0R1 reanalyzes the 20 pre-STRADDLE MASK-1 RGB frames without changing the
historical OBS-0 files. Previous-frame descriptors are grouped by trajectory,
every trajectory step 0 has an invalid-history mask, and all preprocessing is
fitted inside the training direction. The fixed 128-value descriptor and
sample-space ridge solve bound the largest linear system to 10 x 10. Under a
2 GiB process limit and single-threaded numeric libraries, the full run used
182,620 KiB peak RSS and completed in 7.51 seconds.

```text
HISTORY_BOUNDARY_LEAKAGE_FIXED: PASS
TRAIN_ONLY_PREPROCESSING: PASS
SYNTHETIC_ALIGNMENT_TEST: PASS
OBS0R1_REPRODUCIBILITY: PASS
RGB_INCREMENTAL_VALUE_OVER_ODOMETRY: FAIL
BOUNDARY_DISTANCE_OBSERVABILITY: INCONCLUSIVE
READY_FOR_JEPA: NOT_EVALUATED
```

The step-index shortcut reaches 0.114 m direction-holdout MAE and relative
odometry reaches 0.120 m. RGB history plus odometry is worse at 0.836 m, so
OBS-0R1 does not claim incremental visual value. The detailed evidence and
similarity candidates are in [`docs/OBS0R1_AUDIT.md`](docs/OBS0R1_AUDIT.md).
No JEPA training was performed.

## ACT-0S screening-definition audit

ACT-0S reuses the existing 12-candidate, 36-quartet ACT-0 scout without
starting CARLA or changing the historical full physical result (`0/12`). It
separates visual-event suitability (Tier V), plane-free metric repeatability
(Tier M), and strict physical-plane quality (Tier P). Tier P no longer vetoes
Tier V.

The compact scout retained frame/timestamp/pose metadata for all 36 quartets,
but persisted pixels only for each candidate's center RGB and center instance
overlay. Raw depth, mask/contour pixels, K records, and the 24 off-center RGB
views were not retained. Therefore the old plane-basis spread is published
only as a sensitivity proxy: official Tier M and complete three-view public
evidence fail closed instead of being reconstructed.

```text
Tier V geometry-reference pass: 6/12
Tier M official pass: 0 verified; NOT_EVALUATED
Tier P strict pass: 3/12
PUBLIC_ACT0_EVIDENCE: FAIL
SCOUT_SENSOR_PAIRING: PASS
SCOUT_POSE_COVERAGE: FAIL
SCREENING_DEFINITION_VALID: PASS
INSTANCE_GROUPING_RESOLVED: FAIL
READY_FOR_ADAPTIVE_RESCOUT: CONDITIONAL_PASS
READY_FOR_COUNTERFACTUAL_ROLLOUT: FAIL
READY_FOR_DATASET_EXPANSION: NOT_EVALUATED
OPERATOR_VISUAL_REVIEW: PENDING
READY_FOR_JEPA: NOT_EVALUATED
```

Six compact-evidence classifications pass Tier V, but that does not authorize
a rollout. The next permissible step is a small adaptive rescout that retains
the missing boundary-side pixels and plane-free Tier M inputs. See
[`docs/ACT0_SCREENING_AUDIT.md`](docs/ACT0_SCREENING_AUDIT.md).

## ACT-0R adaptive boundary rescout

ACT-0R was restricted to candidates 1, 7, 10 and 19. A bounded, instance-only
locator completed LEFT and RIGHT searches for all four candidates and saved a
pixel-derived search-plan checkpoint. The subsequent canonical
RGB/depth/semantic/instance capture did not complete its first side frame
within the 1200-second client limit. Only one candidate 1 CENTER quartet was
persisted. Its four sensor frame IDs and timestamps match and its five raw
files pass SHA-256 verification. The RGB itself visibly contains severe
repeated triangular geometry/tiling artifacts, so it is capture-failure
evidence rather than usable facade imagery. One corrupted CENTER frame cannot
establish target instance stability, a physical termination, Tier V, official
Tier M, or same-pose repeatability.

The locator events are acquisition positions only. They are not relabeled as
physical boundaries, and ACT-0S classifications are not reused as ACT-0R
outcomes. Historical three-view scout images are included in the public audit
only as candidate context and are visibly marked as non-ACT-0R evidence.

```text
SENSOR_QUADRUPLET_PAIRING: FAIL (1/60 available; the available quartet is paired)
RAW_PIXEL_EVIDENCE_AVAILABLE: FAIL (1/60)
CONFIG_OUTCOME_OVERRIDE_ABSENT: PASS
TARGET_INSTANCE_STABILITY: FAIL
BILATERAL_BOUNDARY_OBSERVED: FAIL
BOUNDARY_TYPE_RESOLVED: FAIL
PHYSICAL_TERMINATION_COUNT: FAIL (0 eligible sides)
TIER_V_RECOMPUTED_FROM_PIXELS: FAIL (0/8 sides evaluated)
OFFICIAL_TIER_M: FAIL (0/8 sides evaluated)
SAME_POSE_CONFIRMATION: FAIL
PUBLIC_VISUAL_EVIDENCE: FAIL
OPERATOR_VISUAL_REVIEW: PENDING
READY_FOR_COUNTERFACTUAL_ROLLOUT: FAIL (0/4 eligible surfaces)
READY_FOR_DATASET_EXPANSION: NOT_EVALUATED
READY_FOR_JEPA: NOT_EVALUATED
```

The detailed failure evidence is in
[`docs/ACT0R_VISUAL_AUDIT.md`](docs/ACT0R_VISUAL_AUDIT.md). Raw ACT-0R arrays
remain server-local and are not tracked by Git. No rollout, dataset expansion,
model download, or JEPA training was run.

## CAP-0 sensor diagnosis and ACT-0R1 recovery pilot

CAP-0 replaced the former long client wait with a bounded sensor path: client
timeout at most 10 s, tick and single-frame queue deadline at most 5 s, two
consecutive incomplete frames as the stop condition, callback-owned raw bytes,
five discarded warmup frames and three settle ticks after teleport. On a fresh
Town10HD_Opt server all five bounded checks passed:

```text
H1 OLD RGB-only: PASS
H2 NEW RGB-only: PASS
H3 OLD RGB/depth/semantic/instance: PASS
H4 NEW RGB/depth/semantic/instance: PASS
H5 quartet after 0.5 m teleport: PASS
CAPTURE_STACK_RECOVERED: PASS
```

The OLD RGB-only path passed with a 4 GiB Python address-space limit but the
single permitted 2 GiB probe stalled near that virtual-address ceiling and was
terminated by its 90 s outer timeout. The result is recorded as
`PYTHON_ADDRESS_SPACE_LIMIT_FAILURE: CONFIRMED`; the probe did not complete
normally and is not presented as an RSS failure.

ACT-0R1 then reused the immutable search checkpoint and captured only candidate
1 LEFT: CENTER, INSIDE, PRE_EDGE, STRADDLE, three frozen-pose STRADDLE repeats
and POST_EDGE. All eight RGB/depth/semantic/instance quartets are paired, raw
BGRA decodes exactly to the persisted PNG, target instance ID is stable at
`39220`, and frozen-pose position/rotation error is `0 m / 0 deg`. The
frozen-pose consecutive SSIM values are `0.991, 0.997, 0.996`; SSIM between
deliberately different camera positions is diagnostic only.

```text
CANDIDATE1_LEFT_CAPTURE_COMPLETE: PASS
SENSOR_QUADRUPLET_PAIRING: PASS
RGB_VISUAL_INTEGRITY: PASS
SAME_POSE_CONFIRMATION: PASS
TARGET_INSTANCE_STABILITY: PASS
READY_TO_RESUME_ACT0R: PASS
READY_FOR_COUNTERFACTUAL_ROLLOUT: NOT_EVALUATED
READY_FOR_JEPA: NOT_EVALUATED
```

See [`docs/CAP0_SENSOR_AUDIT.md`](docs/CAP0_SENSOR_AUDIT.md) and
[`docs/ACT0R1_PILOT_AUDIT.md`](docs/ACT0R1_PILOT_AUDIT.md). Raw BGRA, depth
arrays and segmentation frames remain server-local and ignored by Git. No
RIGHT side, other candidate, rollout, dataset expansion, model download or
JEPA training was run.

## ACT-0R1 offline physical-boundary audit

The offline follow-up reads only the eight retained ACT-0R1 quartets. It
revalidates 56 files, hashes, sensor pairing and the immutable search
checkpoint; decodes instance 39220 from persisted pixels; and computes the
LEFT contour, bilateral depth/semantic/instance evidence, pixel Tier V and
plane-free Tier M without starting CARLA.

The four automatically selected identical-pose frames independently classify
the LEFT edge as `PHYSICAL_TERMINATION` (4/4 agreement). Tier V passes and
the action-axis world-coordinate spread is `0.000 m` under the unchanged
`0.250 m` gate. Same-pose error is `0 m / 0 deg`.

The planned role strings are not ground truth: all eight frames already show
the LEFT contour and target coverage stays near 0.246. They do not establish
an IN/PRE/STRADDLE state sequence.

```text
RAW_HASH_AUDIT: PASS
SENSOR_PAIRING: PASS
TARGET_MASK_PIXEL_VALID: PASS
ROLE_LABEL_INDEPENDENCE: PASS
LEFT_BOUNDARY_TYPE_RESOLVED: PASS (PHYSICAL_TERMINATION)
TIER_V: PASS
OFFICIAL_TIER_M: PASS
SAME_POSE_CONFIRMATION: PASS
EXTERNAL_VISUAL_REVIEW: PENDING
READY_FOR_CANDIDATE1_RIGHT: CONDITIONAL_PASS
READY_FOR_COUNTERFACTUAL_ROLLOUT: NOT_EVALUATED
READY_FOR_JEPA: NOT_EVALUATED
```

The corrected attribution is `TWO_GIB_ADDRESS_SPACE_FAILURE=CONFIRMED`;
the historical triangle artifact cause is
`LIKELY_BUT_NOT_UNIQUELY_PROVEN`. See
[`docs/ACT0R1_OFFLINE_BOUNDARY_AUDIT.md`](docs/ACT0R1_OFFLINE_BOUNDARY_AUDIT.md).
