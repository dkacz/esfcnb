# Contributing

- Edit and review code in the paired percent script: `notebooks/01_data_spine_pl.py`.
- The `.ipynb` remains the source of truth for headless runs; it is paired with the percent script via Jupytext metadata (`ipynb,py:percent`).
- For clean notebook diffs, enable notebook-aware diffs (e.g., nbdime):
  - Install nbdime and run: `nbdime config-git --enable` in your environment.
- Keep the notebook’s first bootstrap cell that sets `ROOT` and imports the runner.
- Headless automation remains a single command: `python scripts/sfc_pl_runner.py`.

