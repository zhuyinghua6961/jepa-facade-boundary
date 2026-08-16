# GEO-0.6 Visual Audit

This is a data-feasibility audit for later modeling. Operator visual review is **PENDING** and `READY_FOR_JEPA` is **NOT_EVALUATED**.

## Overview

![surface_omega](assets/geo06/overview_surface_omega.jpg)
![surface_sigma](assets/geo06/overview_surface_sigma.jpg)

## Gate Results

| Gate | Status |
|---|---|
| SCENE_SUITABILITY | FAIL |
| SENSOR_PAIRING | PASS |
| PHYSICAL_BOUNDARY_GT | FAIL |
| BOUNDARY_VISIBILITY | FAIL |
| TRAJECTORY_ORDERING | PASS |
| EVENT_COVERAGE | FAIL |
| OPERATOR_VISUAL_REVIEW | PENDING |
| READY_FOR_DATASET_EXPANSION | FAIL |
| READY_FOR_JEPA | NOT_EVALUATED |

## Trajectory Evidence

| Sequence | States | UNKNOWN | Spearman | Reverse jump | Events |
|---|---|---:|---:|---:|---|
| `surface_omega/NORMAL_LOCK/DOWN` | `['UNKNOWN', 'OUT']` | 0.833 | 0.963 | 0.020 | False |

![Contact surface_omega_NORMAL_LOCK_DOWN](assets/geo06/contact_surface_omega_NORMAL_LOCK_DOWN.jpg)
![Triptych surface_omega_NORMAL_LOCK_DOWN](assets/geo06/triptych_surface_omega_NORMAL_LOCK_DOWN.jpg)
![STRADDLE surface_omega_NORMAL_LOCK_DOWN](assets/geo06/straddle_surface_omega_NORMAL_LOCK_DOWN.jpg)
![Failure or UNKNOWN surface_omega_NORMAL_LOCK_DOWN](assets/geo06/failure_surface_omega_NORMAL_LOCK_DOWN.jpg)
| `surface_omega/NORMAL_LOCK/LEFT` | `['UNKNOWN', 'OUT']` | 0.893 | 0.923 | 0.036 | False |

![Contact surface_omega_NORMAL_LOCK_LEFT](assets/geo06/contact_surface_omega_NORMAL_LOCK_LEFT.jpg)
![Triptych surface_omega_NORMAL_LOCK_LEFT](assets/geo06/triptych_surface_omega_NORMAL_LOCK_LEFT.jpg)
![STRADDLE surface_omega_NORMAL_LOCK_LEFT](assets/geo06/straddle_surface_omega_NORMAL_LOCK_LEFT.jpg)
![Failure or UNKNOWN surface_omega_NORMAL_LOCK_LEFT](assets/geo06/failure_surface_omega_NORMAL_LOCK_LEFT.jpg)
| `surface_omega/NORMAL_LOCK/RIGHT` | `['UNKNOWN', 'OUT']` | 0.893 | 0.863 | 0.036 | False |

![Contact surface_omega_NORMAL_LOCK_RIGHT](assets/geo06/contact_surface_omega_NORMAL_LOCK_RIGHT.jpg)
![Triptych surface_omega_NORMAL_LOCK_RIGHT](assets/geo06/triptych_surface_omega_NORMAL_LOCK_RIGHT.jpg)
![STRADDLE surface_omega_NORMAL_LOCK_RIGHT](assets/geo06/straddle_surface_omega_NORMAL_LOCK_RIGHT.jpg)
![Failure or UNKNOWN surface_omega_NORMAL_LOCK_RIGHT](assets/geo06/failure_surface_omega_NORMAL_LOCK_RIGHT.jpg)
| `surface_omega/NORMAL_LOCK/UP` | `['UNKNOWN', 'STRADDLE', 'UNKNOWN', 'STRADDLE', 'UNKNOWN', 'OUT']` | 0.722 | 0.977 | 0.020 | False |

![Contact surface_omega_NORMAL_LOCK_UP](assets/geo06/contact_surface_omega_NORMAL_LOCK_UP.jpg)
![Triptych surface_omega_NORMAL_LOCK_UP](assets/geo06/triptych_surface_omega_NORMAL_LOCK_UP.jpg)
![STRADDLE surface_omega_NORMAL_LOCK_UP](assets/geo06/straddle_surface_omega_NORMAL_LOCK_UP.jpg)
![Failure or UNKNOWN surface_omega_NORMAL_LOCK_UP](assets/geo06/failure_surface_omega_NORMAL_LOCK_UP.jpg)
| `surface_sigma/NORMAL_LOCK/DOWN` | `['UNKNOWN', 'OUT']` | 0.864 | 0.000 | 0.000 | False |

![Contact surface_sigma_NORMAL_LOCK_DOWN](assets/geo06/contact_surface_sigma_NORMAL_LOCK_DOWN.jpg)
![Triptych surface_sigma_NORMAL_LOCK_DOWN](assets/geo06/triptych_surface_sigma_NORMAL_LOCK_DOWN.jpg)
![STRADDLE surface_sigma_NORMAL_LOCK_DOWN](assets/geo06/straddle_surface_sigma_NORMAL_LOCK_DOWN.jpg)
![Failure or UNKNOWN surface_sigma_NORMAL_LOCK_DOWN](assets/geo06/failure_surface_sigma_NORMAL_LOCK_DOWN.jpg)
| `surface_sigma/NORMAL_LOCK/LEFT` | `['UNKNOWN', 'OUT']` | 0.889 | 0.000 | 0.000 | False |

![Contact surface_sigma_NORMAL_LOCK_LEFT](assets/geo06/contact_surface_sigma_NORMAL_LOCK_LEFT.jpg)
![Triptych surface_sigma_NORMAL_LOCK_LEFT](assets/geo06/triptych_surface_sigma_NORMAL_LOCK_LEFT.jpg)
![STRADDLE surface_sigma_NORMAL_LOCK_LEFT](assets/geo06/straddle_surface_sigma_NORMAL_LOCK_LEFT.jpg)
![Failure or UNKNOWN surface_sigma_NORMAL_LOCK_LEFT](assets/geo06/failure_surface_sigma_NORMAL_LOCK_LEFT.jpg)
| `surface_sigma/NORMAL_LOCK/RIGHT` | `['UNKNOWN', 'OUT']` | 0.889 | 0.000 | 0.000 | False |

![Contact surface_sigma_NORMAL_LOCK_RIGHT](assets/geo06/contact_surface_sigma_NORMAL_LOCK_RIGHT.jpg)
![Triptych surface_sigma_NORMAL_LOCK_RIGHT](assets/geo06/triptych_surface_sigma_NORMAL_LOCK_RIGHT.jpg)
![STRADDLE surface_sigma_NORMAL_LOCK_RIGHT](assets/geo06/straddle_surface_sigma_NORMAL_LOCK_RIGHT.jpg)
![Failure or UNKNOWN surface_sigma_NORMAL_LOCK_RIGHT](assets/geo06/failure_surface_sigma_NORMAL_LOCK_RIGHT.jpg)
| `surface_sigma/NORMAL_LOCK/UP` | `['UNKNOWN', 'OUT']` | 0.864 | 0.000 | 0.000 | False |

![Contact surface_sigma_NORMAL_LOCK_UP](assets/geo06/contact_surface_sigma_NORMAL_LOCK_UP.jpg)
![Triptych surface_sigma_NORMAL_LOCK_UP](assets/geo06/triptych_surface_sigma_NORMAL_LOCK_UP.jpg)
![STRADDLE surface_sigma_NORMAL_LOCK_UP](assets/geo06/straddle_surface_sigma_NORMAL_LOCK_UP.jpg)
![Failure or UNKNOWN surface_sigma_NORMAL_LOCK_UP](assets/geo06/failure_surface_sigma_NORMAL_LOCK_UP.jpg)

## Representative Frames

| Sequence | Frame ID | Step | Offset (m) | Label | Coverage | Occlusion ratio | Boundary in image | Reason |
|---|---:|---:|---:|---|---:|---:|---|---|
| `surface_omega/NORMAL_LOCK/DOWN` | 4076018 | 15 | -15.00 | OUT | 0.000 | 0.000 | False | independent geometry label |
| `surface_omega/NORMAL_LOCK/DOWN` | 4076003 | 0 | -0.00 | UNKNOWN | 0.645 | 0.645 | False | depth/occlusion or boundary evidence incomplete |
| `surface_omega/NORMAL_LOCK/LEFT` | 4075219 | 25 | -25.00 | OUT | 0.000 | 0.000 | False | independent geometry label |
| `surface_omega/NORMAL_LOCK/LEFT` | 4075194 | 0 | -0.00 | UNKNOWN | 0.645 | 0.645 | False | depth/occlusion or boundary evidence incomplete |
| `surface_omega/NORMAL_LOCK/RIGHT` | 4075533 | 25 | 25.00 | OUT | 0.000 | 0.000 | False | independent geometry label |
| `surface_omega/NORMAL_LOCK/RIGHT` | 4075508 | 0 | 0.00 | UNKNOWN | 0.645 | 0.645 | False | depth/occlusion or boundary evidence incomplete |
| `surface_omega/NORMAL_LOCK/UP` | 4075836 | 10 | 10.00 | STRADDLE | 0.332 | 0.653 | True | independent geometry label |
| `surface_omega/NORMAL_LOCK/UP` | 4075841 | 15 | 15.00 | OUT | 0.000 | 0.000 | False | independent geometry label |
| `surface_omega/NORMAL_LOCK/UP` | 4075826 | 0 | 0.00 | UNKNOWN | 0.645 | 0.645 | False | depth/occlusion or boundary evidence incomplete |
| `surface_sigma/NORMAL_LOCK/DOWN` | 4076724 | 19 | -19.00 | OUT | 0.000 | 0.000 | False | independent geometry label |
| `surface_sigma/NORMAL_LOCK/DOWN` | 4076705 | 0 | -0.00 | UNKNOWN | 0.000 | 0.000 | False | depth/occlusion or boundary evidence incomplete |
| `surface_sigma/NORMAL_LOCK/LEFT` | 4076202 | 24 | -24.00 | OUT | 0.000 | 0.000 | False | independent geometry label |
| `surface_sigma/NORMAL_LOCK/LEFT` | 4076178 | 0 | -0.00 | UNKNOWN | 0.000 | 0.000 | False | depth/occlusion or boundary evidence incomplete |
| `surface_sigma/NORMAL_LOCK/RIGHT` | 4076383 | 24 | 24.00 | OUT | 0.000 | 0.000 | False | independent geometry label |
| `surface_sigma/NORMAL_LOCK/RIGHT` | 4076359 | 0 | 0.00 | UNKNOWN | 0.000 | 0.000 | False | depth/occlusion or boundary evidence incomplete |
| `surface_sigma/NORMAL_LOCK/UP` | 4076567 | 19 | 19.00 | OUT | 0.000 | 0.000 | False | independent geometry label |
| `surface_sigma/NORMAL_LOCK/UP` | 4076548 | 0 | 0.00 | UNKNOWN | 0.000 | 0.000 | False | depth/occlusion or boundary evidence incomplete |

## Rejected Candidate

![Rejected candidate](assets/geo06/rejected_candidate_example.jpg)

Rejected candidate reasons and raycast statistics are recorded in `results/geo06/surface_manifest.json`. Raw RGB-D remains outside Git tracking.
