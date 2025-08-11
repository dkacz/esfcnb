# Empirical SFC Model — Poland Data Spine

Build a quarterly data spine for Poland (PL) that reconciles ESA 2010 sector accounts (Eurostat) with whom‑to‑whom loans (ECB QSA). The spine anchors an empirical stock‑flow consistent (SFC) model by tying non‑financial flows (e.g., net lending B9) to their financial counterparts (B9F), updating balance‑sheet stocks, and checking sectoral identities.

- Single entrypoint: `python scripts/sfc_pl_runner.py`
- Source of truth: `notebooks/01_data_spine_pl.ipynb` (paired with `py:percent` for clean diffs)
- Headless proof: prints `NOTEBOOK_RUN_PROOF <token> <sha256>` bound to the snapshot
- Pass/fail: read from `data/processed/verification_report.json["verify_exit_code"]`

## Scope

Quarterly, tidy coverage for PL:
- NASQ_10_NF_TR (non‑financial transactions)
- NASQ_10_F_TR (financial transactions)
- NASQ_10_F_BS (financial balance sheets)
- NASQ_10_F_GL (revaluations, K.7)
- NASQ_10_F_OC (other changes in volume, K.1–K.6)
- ECB QSA (whom‑to‑whom loans, F4)

Core checks and bridges:
- Resources = Uses on items with both RECV/PAID
- Net lending: B9 (non‑fin) vs B9F [+B9FX9]
- Stocks bridge: ΔAF ≈ F + K.7 + (K.1–K.6)
- Totals: S1 ≈ Σ(S11,S12,S13,S14_S15)
- Whom‑to‑whom symmetry (Loans F4): A(S12→S14_S15) vs L(S14_S15←S12)

## How to run

1) Configure YAML (no env vars): `config/sfc_pl_runner.yml`

```
run:
  execute_notebook: true
io:
  raw_dir: data/raw
  processed_dir: data/processed
  log_file: logs/sfc_pl_runner.log
  notebook_path: notebooks/01_data_spine_pl.ipynb
safety:
  destructive_ops: false
```

2) Execute the single command:

```
python scripts/sfc_pl_runner.py
```

Exit codes: 0 (pass), 6 (verification failed but proof printed), 4 (proof missing), 2 (missing notebook).

## Local Setup & Troubleshooting (Interactive Notebook)

Follow these when opening `notebooks/01_data_spine_pl.ipynb` in VS Code so imports and parquet work reliably.

- Package import: ensure `scripts/__init__.py` exists (it does). Keep the bootstrap cell as the first cell:

```
# --- Project root bootstrap for imports (VS Code/WSL/headless safe) ---
import sys, importlib.util
from pathlib import Path

def find_repo_root() -> Path:
    here = Path.cwd().resolve()
    for parent in (here, *here.parents):
        if (parent / 'scripts' / 'sfc_pl_runner.py').exists() and (parent / 'config' / 'sfc_pl_runner.yml').exists():
            return parent
    raise FileNotFoundError('Could not find project root with scripts/sfc_pl_runner.py and config/sfc_pl_runner.yml')

ROOT = find_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print('Project root:', ROOT)

try:
    from scripts.sfc_pl_runner import load_config, run_from_config, verify
    print('Imported from package:', ROOT / 'scripts' / 'sfc_pl_runner.py')
except ModuleNotFoundError:
    mfp = ROOT / 'scripts' / 'sfc_pl_runner.py'
    spec = importlib.util.spec_from_file_location('scripts.sfc_pl_runner', str(mfp))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    load_config, run_from_config, verify = mod.load_config, mod.run_from_config, mod.verify
    print('Imported via file path:', mfp)
```

- Kernel: in VS Code use "Python: Select Interpreter" or "Jupyter: Select Kernel" and pick your project venv.
- Dependencies (install into the selected interpreter):

```
pip install -r requirements.txt
# or
pip install pandasdmx nbclient nbformat pyarrow pandas
```

- Sanity check inside a cell:

```
import sys, platform
import pandasdmx, nbclient, nbformat, pandas, pyarrow
print('OK:', pandasdmx.__version__, nbclient.__version__, pandas.__version__)
print('exe:', sys.executable, '|', platform.platform())
```

- Paths: the runner executes with repo root as CWD; the notebook uses `ROOT / 'data' / ...` absolute paths (not relative paths).

## Dev UX

- Paired percent script: `notebooks/01_data_spine_pl.py` (`#%%` cells) for code review and quick edits; the `.ipynb` holds rich outputs.
- Notebook diffs: enable nbdime (`pip install nbdime && nbdime config-git --enable`).
- The first notebook cell bootstraps `ROOT` (repo root) and uses absolute `ROOT / 'data' / ...` paths for deterministic execution.

## Outputs

- Parquet: `data/processed/estat_nasq_10_*.parquet`, `data/processed/ecb_QSA_PL.parquet`
- JSON: `data/processed/verification_report.json`, `data/processed/qc_summary.json`
- Executed copy: `notebooks/executed_01_data_spine_pl.ipynb`

## Repository Structure

- `scripts/sfc_pl_runner.py` — headless executor + proof capture + gating
- `notebooks/01_data_spine_pl.ipynb` — canonical pipeline (paired with `py:percent`)
- `config/sfc_pl_runner.yml` — YAML‑only configuration
- `src/` — helpers (SDMX key builder, DSD cache, tidy, QC)
- `data/processed/`, `logs/` — artifacts (gitignored)
