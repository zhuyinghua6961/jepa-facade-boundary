# JEPA Facade Boundary

Reusable CARLA geometry and RGB-D tooling for active facade-boundary perception from an aerial-camera viewpoint. This repository validates the image-boundary-pixel -> 3D-world-coordinate -> facade-coordinate chain; it does not train JEPA.

## Status

The repository contains the GEO-0.5R2 implementation and its reproducible audit. CARLA 0.10.0 UE5 / Town10HD_Opt was used with 640x480 RGB-D sensors, horizontal FOV 90 degrees, synchronous mode, fixed delta 0.05 s, and NORMAL_LOCK camera motion.

```text
pytest: 17 passed
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
