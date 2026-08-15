# Experiment Policy

1. Use a unique `run_id` for every capture or validation run.
2. Put changes in YAML configuration and result metadata; do not fork source directories.
3. Keep human annotations, fitted physical boundary lines, final metrics, manifests and representative audit figures.
4. Treat full RGB-D, RAW/NPY/BIN and full overlay trees as regenerable outputs. Check references and record hashes before isolation.
5. Run `python3 scripts/clean_artifacts.py --dry-run` at the end of each experiment. Use `--apply` only after reviewing its explicit file list.
6. Never publish tokens, credentials, SSH files, internal hostnames, absolute server paths, model weights or external datasets.
7. A failed gate remains failed in the public report. Do not relabel or remove evidence to turn a failed experiment into a pass.
