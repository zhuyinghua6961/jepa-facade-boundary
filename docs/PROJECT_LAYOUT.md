# Project Layout

The repository has one implementation of each reusable capability. Experiment differences belong in YAML and result manifests, not in copied `v2`, `v3`, `final`, or `new` source trees.

`boundary_sweep/` owns geometry, CARLA sensor lifecycle, surface schemas, labels and CARLA utility functions. `scripts/` only parses arguments and calls those modules. `results/<run_id>/` stores configuration snapshots, manifests, metrics, logs and a small figure subset; it never contains a source-code copy.

Generated raw RGB-D and full overlays are disposable unless they contain irreplaceable human annotation. Human clicks and fitted physical boundaries remain under the annotation/results records. The cleanup tool only operates on explicit project-local allowlists.
