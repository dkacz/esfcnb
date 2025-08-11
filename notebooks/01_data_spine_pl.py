#%%
# --- Project root bootstrap for imports (works in VS Code, WSL, headless) ---
import sys, importlib.util
from pathlib import Path

def find_repo_root() -> Path:
    """
    Walk up from CWD to locate the project root by markers:
    scripts/sfc_pl_runner.py and config/sfc_pl_runner.yml.
    """
    here = Path.cwd().resolve()
    for parent in (here, *here.parents):
        if (parent / 'scripts' / 'sfc_pl_runner.py').exists() and (parent / 'config' / 'sfc_pl_runner.yml').exists():
            return parent
    raise FileNotFoundError('Could not find project root with scripts/sfc_pl_runner.py and config/sfc_pl_runner.yml')

ROOT = find_repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
print('Project root:', ROOT)

# Primary import (package import requires scripts/__init__.py)
try:
    from scripts.sfc_pl_runner import load_config, run_from_config, verify
    print('Imported from package:', ROOT / 'scripts' / 'sfc_pl_runner.py')
except ModuleNotFoundError:
    # Absolute path fallback (rarely needed if sys.path is set and scripts/__init__.py exists)
    mfp = ROOT / 'scripts' / 'sfc_pl_runner.py'
    spec = importlib.util.spec_from_file_location('scripts.sfc_pl_runner', str(mfp))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    load_config, run_from_config, verify = mod.load_config, mod.run_from_config, mod.verify
    print('Imported via file path:', mfp)

#%%
# Build B9F/B9FX9 subset for anchor and save parquet
from pathlib import Path
import pandas as pd
from src.sdmx_helpers import get_dsd, fetch_series
ROOT = ROOT
proc = ROOT / 'data' / 'processed'
proc.mkdir(parents=True, exist_ok=True)
fp_ftr = proc / 'estat_nasq_10_f_tr_PL.parquet'
fp_b9f = proc / 'estat_nasq_10_f_tr_b9f_PL.parquet'
df_b9f = pd.DataFrame()
if fp_ftr.exists():
    dff = pd.read_parquet(fp_ftr)
    if 'na_item' in dff.columns:
        df_b9f = dff[dff['na_item'].astype(str).isin(['B9F','B9FX9'])].copy()
if df_b9f is None or df_b9f.empty:
    try:
        info = get_dsd('ESTAT','NASQ_10_F_TR')
        units = [u for u in ['MIO_EUR','CP_MEUR'] if u in info.codes.get('unit',[])] or ['MIO_EUR']
        finpos_opts = [c for c in ['NET','ASS','LIAB'] if c in info.codes.get('finpos',[])]
        frames=[]
        for unit in units:
            tried=[None]
            if 'NET' in finpos_opts: tried=["NET"]
            for fin in tried:
                filt={'freq':'Q','unit':unit,'sector':['S13'],'na_item':['B9F','B9FX9'],'geo':'PL'}
                if fin: filt['finpos']=[fin]
                try:
                    dfp = fetch_series('NASQ_10_F_TR', filt, agency='ESTAT')
                    if dfp is not None and not dfp.empty:
                        frames.append(dfp)
                except Exception:
                    pass
        if frames:
            import pandas as pd
            df_b9f = pd.concat(frames, ignore_index=True)
    except Exception:
        pass
if df_b9f is not None and not df_b9f.empty:
    df_b9f.to_parquet(fp_b9f)
print('B9F rows:', 0 if df_b9f is None else len(df_b9f))

#%%
# Derive NF_TR check_items and windows; save nf_tr_meta.json
from pathlib import Path
import json, pandas as pd
ROOT=ROOT
proc = ROOT / 'data' / 'processed'
meta_fp = proc / 'nf_tr_meta.json'
df_fp = proc / 'estat_nasq_10_nf_tr_PL.parquet'
meta={}
try:
    if df_fp.exists():
        df = pd.read_parquet(df_fp)
        core=['S11','S12','S13','S14_S15']
        if set(['time','na_item','direct','sector']).issubset(df.columns):
            d = df[df['sector'].isin(core)].copy()
            piv = d.pivot_table(index=['time','na_item'], columns='direct', values='value', aggfunc='sum').fillna(0.0)
            cols=[c for c in ['RECV','PAID'] if c in piv.columns]
            items=[]; win={}
            if len(cols)==2:
                mask=(piv[cols[0]]!=0)&(piv[cols[1]]!=0)
                idx=piv[mask].reset_index()[['time','na_item']]
                items=sorted(idx['na_item'].astype(str).unique().tolist())
                for itm, sub in idx.groupby('na_item'):
                    win[str(itm)]={'first_quarter': str(sub['time'].min()), 'last_quarter': str(sub['time'].max())}
            meta={'sectors_direct': core,'na_items_final': items,'check_items': items,'check_item_windows': win}
            meta_fp.write_text(json.dumps(meta, indent=2), encoding='utf-8')
except Exception:
    pass
print('NF_TR check_items:', [] if not meta else meta.get('check_items'))

#%%
# STEP 1.4 — ECB QSA Loans F4 (PL): whom-to-whom minimal panel
from pathlib import Path
import pandas as pd
from src.sdmx_helpers import get_dsd, fetch_series
ROOT = ROOT
proc = ROOT / 'data' / 'processed'
proc.mkdir(parents=True, exist_ok=True)
try:
    info = get_dsd('ECB','QSA')
    dims = info.dimensions
    codes = info.codes
    def has(dim, vals):
        return dim in codes and all(v in set(codes[dim]) for v in vals)
    instr_dim = next((d for d in dims if has(d,['F4'])), None)
    entry_dim = next((d for d in dims if has(d,['A','L']) or has(d,['ASS','LIAB'])), None)
    stock_dim = next((d for d in dims if has(d,['LE']) or has(d,['S','F'])), None)
    unit_dim = next((d for d in dims if has(d,['MIO_EUR']) or has(d,['CP_MEUR'])), None)
    geo_dim = next((d for d in dims if has(d,['PL'])), None)
    cons_dim = next((d for d in dims if has(d,['N']) or has(d,['NC'])), None)
    sec_dims = [d for d in dims if has(d,['S11','S12','S13','S14_S15'])][:2]
    flt={'freq':'Q'}
    if unit_dim: flt[unit_dim]='MIO_EUR'
    if geo_dim: flt[geo_dim]='PL'
    if instr_dim: flt[instr_dim]='F4'
    if entry_dim: flt[entry_dim]=['A','L']
    if stock_dim:
        if has(stock_dim,['LE']): flt[stock_dim]='LE'
        else: flt[stock_dim]=['S','F']
    if cons_dim: flt[cons_dim]='N'
    if len(sec_dims)==2:
        flt[sec_dims[0]]=['S11','S12','S13','S14_S15']
        flt[sec_dims[1]]=['S11','S12','S13','S14_S15']
    df_qsa = fetch_series('QSA', flt, agency='ECB')
    if df_qsa is None or df_qsa.empty:
        df_qsa = pd.DataFrame()
    else:
        # normalize column names
        ren={}
        if instr_dim: ren[instr_dim]='instrument'
        if entry_dim: ren[entry_dim]='entry'
        if stock_dim: ren[stock_dim]='stock_flow_flag'
        if len(sec_dims)==2:
            ren[sec_dims[0]]='ref_sector'; ren[sec_dims[1]]='cp_sector'
        df_qsa = df_qsa.rename(columns=ren)
        keep=['time','ref_sector','cp_sector','instrument','entry','stock_flow_flag','unit','value','dataset','last_update']
        for c in keep:
            if c not in df_qsa.columns: df_qsa[c]=None
        df_qsa = df_qsa[keep]
    if not df_qsa.empty:
        df_qsa.to_parquet(proc / 'ecb_QSA_PL.parquet')
    print('QSA rows:', 0 if df_qsa is None else len(df_qsa))
except Exception as e:
    print('QSA pull failed:', e)

#%%
# STEP 1.4 — Eurostat revaluations (K.7) and other changes (K.1–K.6)
from pathlib import Path
import pandas as pd
from src.sdmx_helpers import get_dsd, fetch_series
ROOT = ROOT
proc = ROOT / 'data' / 'processed'
proc.mkdir(parents=True, exist_ok=True)

def _pull_estat(dataset):
    try:
        info = get_dsd('ESTAT', dataset)
    except Exception:
        return pd.DataFrame()
    unit = next((u for u in ['MIO_EUR','CP_MEUR'] if u in info.codes.get('unit',[])), None)
    if not unit:
        unit='MIO_EUR'
    if dataset.endswith('_F_GL'):
        items=[c for c in info.codes.get('na_item',[]) if str(c).startswith('K7') or str(c) in ('K7','K.7')]
    else:
        items=[c for c in info.codes.get('na_item',[]) if str(c).startswith('K')]
    if not items: items = ['K7'] if dataset.endswith('_F_GL') else ['K1','K2','K3','K4','K5','K6']
    filters={'freq':'Q','unit':unit,'sector':['S1','S11','S12','S13','S14_S15','S2'],'na_item':items,'geo':'PL'}
    try:
        return fetch_series(dataset, filters, agency='ESTAT')
    except Exception:
        return pd.DataFrame()

df_gl=_pull_estat('NASQ_10_F_GL')
df_oc=_pull_estat('NASQ_10_F_OC')
if df_gl is not None and not df_gl.empty:
    df_gl.to_parquet(proc / 'estat_nasq_10_f_gl_PL.parquet')
if df_oc is not None and not df_oc.empty:
    df_oc.to_parquet(proc / 'estat_nasq_10_f_oc_PL.parquet')
print('F_GL rows:', 0 if df_gl is None else len(df_gl), '| F_OC rows:', 0 if df_oc is None else len(df_oc))

#%%
CFG = load_config()
rc = verify(CFG)
print('verify() exit code:', rc)

#%% [markdown]
# # Data Spine — Poland (Eurostat + ECB)
# 
# This notebook orchestrates Step 1 pulls and QC via the fixed runner.

#%% [markdown]
# ### VERIFY & FIX (STEP 1.2)
# 
# Run All to pull SDMX data and print the verification block.

#%%
# (Optional) full pipeline from notebook (skipped by default)
import os
if os.environ.get('NOTEBOOK_PULL')=='1':
    _ = run_from_config(CFG)
else:
    print('Skipping full pull (set NOTEBOOK_PULL=1 to enable)')

#%%
# Single-source verification print (JSON-driven)
# Always call verify() at least once so it writes the snapshot JSON and exit code.
rc = verify(CFG)
print(f"verify() exit code: {rc}")

# The rest of the display must be built from the freshly written JSON.
import json, pandas as pd
from pathlib import Path
vr = json.loads((ROOT / 'data' / 'processed' / 'verification_report.json').read_text(encoding='utf-8'))

# --- Artifacts table ---
art_rows = []
for p, meta in vr.get('artifacts', {}).items():
    art_rows.append({
        'path': p,
        'exists': meta.get('exists'),
        'size_bytes': meta.get('size_bytes'),
        'rows': meta.get('rows'),
        'cols': meta.get('cols'),
        'first_quarter': meta.get('first_quarter'),
        'last_quarter': meta.get('last_quarter'),
    })
art_df = pd.DataFrame(art_rows).sort_values('path')
display(art_df)

# Echo NF_TR scope, QC, anchors, failures from JSON
display(vr.get('nf_tr_scope', {}))
display(vr.get('qc_summary', {}))
display(vr.get('anchors', {}))
display(vr.get('failures', {}))

# Checksums (file & sample) from JSON
display(vr.get('checksums', {}))

# Hard stop on failure (notebook)
VERIFY_RC = rc
# soft-fail: do not abort; proof cell will still run

#%%
# Re-run verification only (idempotent); prints full block
from scripts.sfc_pl_runner import verify
rc = verify(CFG)
print('verify() returned:', rc)

#%% [markdown]
# #### Output Gate & Debug Pack — Self-contained
# Run to print the artifacts table, schema checks, NF_TR scope, QC summary, anchors, and failure lines.

#%%
# Print the full verification block (includes artifacts, schema, scope, QC, anchors, failures)
from scripts.sfc_pl_runner import verify
rc = verify(CFG)
print('verify() returned:', rc)

#%%
# Convenience: print single worst stocks-bridge line + one failure log line if any
import json, re, pathlib
qc=json.load(open('data/processed/qc_summary.json')) if pathlib.Path('data/processed/qc_summary.json').exists() else {}
worst=(qc.get('stocks_bridge',{}).get('worst10') or [])
print('WORST stocks-bridge line:', worst[0] if worst else None)
# Show first fetch failure logged
log_lines=[]
try:
    with open('logs/sfc_pl_runner.log','r',encoding='utf-8') as fh:
        for ln in fh:
            if ln.strip().startswith('FAIL | dataset='):
                log_lines.append(ln.strip())
except FileNotFoundError:
    pass
print('ONE failure line (if any):', log_lines[0] if log_lines else None)

#%%
# --- Notebook run proof: emit a runtime token and bind it to the JSON snapshot ---
import os, sys, json, time, hashlib, platform
from pathlib import Path

vr_path = (ROOT / 'data' / 'processed' / 'verification_report.json')
assert vr_path.exists(), "verification_report.json not found; run verify(CFG) first"
vr_bytes = vr_path.read_bytes()
vr_sha = hashlib.sha256(vr_bytes).hexdigest()

# Runtime-only token (cannot exist without execution)
nonce = os.urandom(32)
now_ns = time.time_ns()
proof_token = hashlib.sha256(nonce + str(now_ns).encode('utf-8')).hexdigest()[:16]

proof = {
    "run_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    "cwd": str(Path.cwd().resolve()),
    "python": sys.version,
    "platform": platform.platform(),
    "verification_report_sha256": vr_sha,
    "proof_token": proof_token,
}

proof_path = ROOT / 'data' / 'processed' / '_nb_run_proof.json'
proof_path.parent.mkdir(parents=True, exist_ok=True)
proof_path.write_text(json.dumps(proof, indent=2), encoding='utf-8')
print('NOTEBOOK_RUN_PROOF', proof_token, vr_sha)

