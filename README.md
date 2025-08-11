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

