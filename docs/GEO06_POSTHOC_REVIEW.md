# GEO-0.6 Post-hoc Review

This note interprets the existing GEO-0.6 evidence without modifying its historical `results/geo06/validation.json`.

- `PLANE_BASED_LABEL_PROTOCOL=FAIL`: the protocol compared rendered depth to a single collision plane and is not a reliable outer-contour truth source.
- `surface_omega`: LEFT/RIGHT/UP outer edges are visible in RGB, but windows, recesses and concave geometry create depth mismatches.
- `surface_sigma`: the collision plane disagrees with rendered z-depth, so its `coverage=0` is not evidence that the visual facade is absent.
- `DOWN`: a fixed-normal sweep that crosses the bottom edge can enter the ground. MASK-0 treats this as an AGL-constrained action and reports it infeasible when downward raycast cannot establish `min_agl_m >= 2.0`.
- The old GEO-0.6 `TRAJECTORY_ORDERING=PASS` was an empty pass for most tracks: after filtering UNKNOWN, only OUT remained. It is not evidence of a complete event sequence.

MASK-0 uses CARLA semantic tag `3` (`CityObjectLabel.Buildings`) and instance IDs decoded from raw BGRA. It does not use RGB appearance classification or hand-drawn masks.
