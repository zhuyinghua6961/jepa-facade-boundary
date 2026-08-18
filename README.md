# JEPA Facade Boundary

Reusable CARLA geometry and RGB-D tooling for active facade-boundary perception from an aerial-camera viewpoint. This repository validates the image-boundary-pixel -> 3D-world-coordinate -> facade-coordinate chain; it does not train JEPA.

## Status

The repository contains the GEO-0.5R2 implementation and its reproducible audit. CARLA 0.10.0 UE5 / Town10HD_Opt was used with 640x480 RGB-D sensors, horizontal FOV 90 degrees, synchronous mode, fixed delta 0.05 s, and NORMAL_LOCK camera motion.

```text
pytest: 112 passed
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

## ACT-0R2 checkpoint-aligned bilateral event capture

ACT-0R2 captures candidate 1 only. It reads `locator_center_pose`, the action
axis and both seven-role displacement lists directly from the immutable ACT-0R
search checkpoint; the named CAP-0 `OLD` and `NEW` poses are not used. One
actual CENTER quartet is shared by LEFT and RIGHT, for a hard total of 15
persisted quartets.

The CENTER pose differs from the checkpoint by 0.00000763 m and 0 degrees.
Neither physical termination is present at CENTER. Both directions then show
the same pixel/geometric sequence under unchanged thresholds:

```text
NO_VALID_EXTERNAL_BOUNDARY
APPROACH
FIRST_PHYSICAL_TERMINATION
PHYSICAL_TERMINATION x 4
```

The four identical-pose termination frames classify as
`PHYSICAL_TERMINATION` with 4/4 agreement on each side. Both Tier V gates pass,
and each same-pose action-axis world-coordinate spread is 0.000 m. This is a
same-pose sensor confirmation, not multiview repeatability.

```text
CHECKPOINT_POSE_ALIGNMENT: PASS
SENSOR_PAIRING: PASS
BILATERAL_SAME_START: PASS
CENTER_LEFT_BOUNDARY_ABSENT: PASS
CENTER_RIGHT_BOUNDARY_ABSENT: PASS
LEFT_EVENT_ORDERING: PASS
RIGHT_EVENT_ORDERING: PASS
LEFT_PHYSICAL_TERMINATION: PASS
RIGHT_PHYSICAL_TERMINATION: PASS
LEFT_TIER_V: PASS
RIGHT_TIER_V: PASS
SAME_POSE_CONFIRMATION: PASS
SAME_POSE_WORLD_BOUNDARY_REPEATABILITY: PASS
MULTIVIEW_REPEATABILITY: NOT_EVALUATED
EXTERNAL_VISUAL_REVIEW: PENDING
READY_FOR_NEXT_SURFACE: CONDITIONAL_PASS
READY_FOR_COUNTERFACTUAL_ROLLOUT: NOT_EVALUATED
READY_FOR_JEPA: NOT_EVALUATED
```

Raw RGB, z-depth and segmentation payloads remain server-local and ignored by
Git. See [`docs/ACT0R2_VISUAL_AUDIT.md`](docs/ACT0R2_VISUAL_AUDIT.md) for the
public compressed evidence. No other candidate, rollout, model download or
JEPA training was run.

## CF-0 single-facade counterfactual observability kill test

CF-0 reuses candidate 1 and the ACT-0R2 checkpoint. Seed `20260817` produced
20 accepted shared starts: eight LEFT-biased, four near-center and eight
RIGHT-biased. Each shared start has an independently captured LEFT and RIGHT
branch at 0.5 m increments through 4.0 m. All 340 RGB/depth/semantic/instance
quartets pair exactly and all 2,040 sensor payload hashes pass. Both valid
physical boundaries are absent in every shared-start image.

Pixel/geometric GT finds 13 branches with a preregistered
`MODEL_VISIBLE_TERMINATION` and 27 right-censored branches. Five-fold splits
use `start_id`, keeping both counterfactual branches together. PCA and linear
probes are fitted only on training folds. Absolute coordinates, planned start
offset, frame ID, planned role and world-boundary coordinates are excluded
from every model feature.

```text
                              B0       B1       B2       B3
rolling balanced accuracy   0.5000   0.5000   0.5557   0.5196
rolling TTE MAE (m)         0.7946   0.4525   0.2775   0.2906
same-start action accuracy  0.5000   0.3077   0.5385   0.3077
same-start regret (m)       0.7308   0.8846   0.6923   0.9615
```

B3 improves MAE over B1 by only `0.1619 m` (required `0.25 m`) and balanced
accuracy by `0.0196` (required `0.10`). Its non-tie same-start action accuracy
is `0.3077`, identical to B1 and below the required `0.65`. At a shared start,
B3 has no valid earlier branch frame; its history-valid mask is false. Rolling
B3 metrics use only frames strictly before the robust event, while the action
comparison uses shared-start predictions only.

```text
COUNTERFACTUAL_PAIRING: PASS
START_BOUNDARY_ABSENT: PASS
ROBUST_EVENT_COVERAGE: PASS
SPLIT_LEAKAGE_AUDIT: PASS
VISUAL_INCREMENTAL_VALUE: FAIL
ACTION_SELECTION_SIGNAL: FAIL
SINGLE_SURFACE_SIGNAL: FAIL
CROSS_SURFACE_GENERALIZATION: NOT_EVALUATED
READY_FOR_MULTI_SURFACE_CAPTURE: FAIL
READY_FOR_JEPA: NOT_EVALUATED
```

The preregistered kill test therefore stops at one facade: current evidence
does not justify multi-surface capture or a JEPA experiment. Raw CF-0 payloads
remain as one server-local, Git-ignored canonical copy. See
[`docs/CF0_OBSERVABILITY_AUDIT.md`](docs/CF0_OBSERVABILITY_AUDIT.md).

## PROBE-0 active-disambiguation kill test

PROBE-0 is a one-shot offline audit over the frozen CF-0 raw. It compares the
frozen shared-start B2 action-selection accuracy (`0.5385`) with the unchanged
CF-0 B3 feature/model pipeline after a fixed 1.0 m probe. The primary history
contains exactly the 0.5 m and 1.0 m frames; 0.5 m is reported only as a
diagnostic, and no 1.5 m boundary frame or later frame enters the model.

The 26 primary samples come from LEFT and RIGHT probes at the 13 frozen
non-tie starts. Both probes from a start remain in the same held-out fold.
Pooled accuracy is `0.8846` (balanced accuracy `0.8869`, AUROC `0.9762`, mean
regret `0.1731 m`), with a start-cluster bootstrap accuracy interval of
`[0.7308, 1.0000]`. LEFT-probe and RIGHT-probe accuracy are `0.8462` and
`0.9231`. The improvement over the frozen B2 reference is `0.3462`.

```text
SOURCE_RAW_HASH_AUDIT: PASS (390/390 payloads)
GROUP_SPLIT_LEAKAGE_AUDIT: PASS
PREBOUNDARY_INPUT_AUDIT: PASS
ACTIVE_DISAMBIGUATION_SIGNAL: PASS
ACTIVE_FACADE_JEPA_ROUTE: CONDITIONAL_GO
READY_FOR_NEW_CAPTURE: CONDITIONAL_PASS
READY_FOR_SECOND_SURFACE_REPLICATION: CONDITIONAL_PASS
EXTERNAL_VISUAL_REVIEW: PENDING
READY_FOR_JEPA: NOT_EVALUATED
```

This is evidence for replication on a second surface, not authorization to
train JEPA. It uses one facade and only 13 independent start groups, so
cross-surface generalization remains untested. CARLA was not started, no new
capture or download occurred, and no model artifact was saved. See
[`docs/PROBE0_ACTIVE_DISAMBIGUATION_AUDIT.md`](docs/PROBE0_ACTIVE_DISAMBIGUATION_AUDIT.md).

## PROBE-0R1 causal attribution audit

PROBE-0R1 reuses the same 13 starts, 26 samples, 1.0 m endpoints, folds,
descriptor, seed, PCA limit and ridge setting. It asks whether PROBE-0 gains
come from ordered RGB history or from the more informative static endpoint.
E3/E4 are held-out interventions: every fold is fit on normal ordered E2
features, then only held-out previous/current inputs are swapped or removed.

```text
               E0       E1       E2       E3_SWAP  E4_ZERO
accuracy       0.3077   0.8846   0.8846   0.9231   0.4615
balanced acc   0.3214   0.8869   0.8869   0.9226   0.5000
AUROC          0.2976   0.9464   0.9762   0.9821   0.5000
regret m       1.0000   0.1731   0.1731   0.1346   0.8077
```

E2 has zero accuracy improvement over current endpoint RGB alone (E1), and
E3_SWAP is 0.0385 better rather than at least 0.10 worse. E4_ZERO degradation
shows that the fitted E2 classifier uses the previous-RGB block, but it does
not establish useful temporal order. The paired E2-E1 accuracy difference is
`0.0000`, with start-cluster bootstrap 95% CI `[-0.1154, 0.1154]`.

```text
SOURCE_FILE_HASH_AUDIT: PASS
RAW_PAYLOAD_HASH_AUDIT: PASS (390/390)
STRICT_SAME_ENDPOINT: PASS
FOLD_REUSE: PASS
E2_REPRODUCES_PROBE0: PASS
TEMPORAL_HISTORY_INCREMENTAL_VALUE: FAIL
ACTIVE_JEPA_ROUTE: NO_GO
READY_FOR_SECOND_SURFACE_REPLICATION: FAIL
READY_FOR_JEPA: NOT_EVALUATED
```

Under the preregistered kill-test rule, the second-surface capture is stopped.
The evidence is more consistent with static endpoint appearance than with an
incremental ordered-history signal. See
[`docs/PROBE0R1_CAUSAL_ATTRIBUTION_AUDIT.md`](docs/PROBE0R1_CAUSAL_ATTRIBUTION_AUDIT.md).

## AVS-0 active viewpoint-selection headroom audit

AVS-0 asks a narrower cross-surface question: can choosing a LEFT or RIGHT
1.0 m endpoint per start outperform the best fixed endpoint? It does not train
an active policy or JEPA. Candidates were qualified in the preregistered order
`7, 8, 10, 19`; candidates 7 and 8 were the first two to pass bilateral
physical-termination, Tier V, plane-free Tier M, and endpoint-safety gates.
Together with frozen candidate 1, the surface leave-one-out evaluation contains
three surfaces with eight shared non-tie starts each.

```text
FIXED_LEFT accuracy:                    0.666667
FIXED_RIGHT accuracy:                   0.541667
RANDOM expected accuracy:               0.604167
ORACLE_PER_START accuracy:              0.666667
ORACLE - best FIXED:                    0.000000
start-cluster bootstrap 95% CI:         [0.000000, 0.000000]
unique LEFT / RIGHT optimum fractions:  0.125000 / 0.000000
```

The preregistered oracle accuracy (`>=0.70`), oracle headroom (`>=0.15`),
strictly positive CI lower bound, and bilateral unique-action support (`>=20%`
each) all fail. The data therefore provide no performance space for an adaptive
probe policy under this frozen E1 endpoint-RGB setup.

Candidate 1's earlier fixed-RIGHT value of `1.0` was a 13-start within-surface
grouped estimate. AVS-0 instead holds candidate 1 out, trains only on candidates
7 and 8, and evaluates eight frozen candidate-1 starts; its fixed-RIGHT accuracy
is `0.125`. The two numbers answer different questions, so the historical value
is retained as motivation and is not reused as a cross-surface score.

```text
PHYSICAL_BOUNDARY_AND_SAFETY: PASS
SENSOR_QUADRUPLET_PAIRING: PASS (70/70 new groups)
SURFACE_LEAVE_ONE_OUT: PASS
ACTIVE_VIEW_SELECTION_HEADROOM: FAIL
READY_FOR_POLICY_PILOT: FAIL
READY_FOR_JEPA: NOT_EVALUATED
```

Raw sensor quartets remain as one Git-ignored server copy. Git contains only
compact manifests, CSV results, documentation and compressed JPG evidence. See
[`docs/AVS0_ACTIVE_VIEW_SELECTION_AUDIT.md`](docs/AVS0_ACTIVE_VIEW_SELECTION_AUDIT.md).
