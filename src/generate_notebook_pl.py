#!/usr/bin/env python3
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path


def build_notebook() -> nbf.NotebookNode:
    nb = new_notebook()

    md_intro = (
        "# Step 1 — Data spine for Poland (SDMX via Eurostat & ECB)\n"
        "This notebook pulls quarterly sector accounts for Poland (geo=PL), tidies and maps ESA→SFC variables, runs QC checks, and saves Parquet outputs and a minimal README."
    )

    cell1 = r"""
# Cell 1 — Imports & config
import warnings
import sys, time, json, logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from pandasdmx import Request
from dateutil import parser as dateparser

# Version printouts
print('Python:', sys.version.split()[0])
print('pandas:', pd.__version__)
try:
    import pandasdmx as _pdmx
    print('pandasdmx:', _pdmx.__version__)
except Exception as e:
    print('pandasdmx: <import issue>', e)
try:
    import pyarrow as pa
    print('pyarrow:', pa.__version__)
except Exception as e:
    print('pyarrow: <not available>', e)

# Helper for timestamps
def now_iso():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('sfc_pl')

# Paths
ROOT = Path('.').resolve()
DIR_RAW_ESTAT = ROOT / 'data' / 'raw' / 'estat'
DIR_RAW_ECB = ROOT / 'data' / 'raw' / 'ecb'
DIR_PROCESSED = ROOT / 'data' / 'processed'
DIR_NOTEBOOKS = ROOT / 'notebooks'
for d in [DIR_RAW_ESTAT, DIR_RAW_ECB, DIR_PROCESSED, DIR_NOTEBOOKS]:
    d.mkdir(parents=True, exist_ok=True)

# SDMX clients
estat = Request('ESTAT')
ecb = Request('ECB')

# Constants
GEO = 'PL'
S_ADJ = 'NSA'
FREQ = 'Q'
SECTORS = ['S11','S12','S13','S14_S15','S2','S1']
NF_ITEMS = [
    'B1G','B2G_B3G','B6G','B7G','B8G','B9',
    'D1','D2','D3','D4','D41','D421','D5','D61','D62','D8',
    'P3','P31','P51G','P52_P53'
]
UNITS_PREFERRED = ['CP_MEUR','MIO_EUR']  # keep units as published

DATASETS = {
    'nasq_10_nf_tr': 'Non-financial transactions by sector (quarterly)',
    'nasq_10_f_tr': 'Financial transactions by sector (quarterly)',
    'nasq_10_f_bs': 'Financial balance sheets by sector (quarterly)',
    'nasq_10_f_gl': 'Revaluation account (K.7) (optional)',
    'nasq_10_f_oc': 'Other changes in volume (K.1–K.6) (optional)',
    'namq_10_gdp': 'GDP & main components (store for deflators)'
}

# Helpers
def sdmx_get(client: Request, flow: str, key=None, params=None, retries: int = 3, backoff: float = 1.5):
    params = params or {}
    params.setdefault('compressed', True)
    attempt = 0
    tried_upper = False
    while True:
        try:
            resp = client.data(resource_id=flow, key=key or {}, params=params)
            return resp
        except Exception as e:
            # If dataflow is missing, try uppercase once
            if not tried_upper:
                try:
                    alt = flow.upper()
                    if alt != flow:
                        logger.warning(f"SDMX retry with uppercase flow id: {alt}")
                        resp = client.data(resource_id=alt, key=key or {}, params=params)
                        return resp
                except Exception:
                    pass
                tried_upper = True
            attempt += 1
            if attempt >= retries:
                logger.error(f'Failed SDMX fetch for {flow} after {retries} attempts: {e}')
                raise
            sleep = backoff ** attempt
            logger.warning(f'SDMX retry {attempt}/{retries} for {flow} in {sleep:.1f}s due to: {e}')
            time.sleep(sleep)

def sdmx_save_raw(resp, path: Path):
    path = Path(path)
    content = None
    try:
        content = resp.response.content
    except Exception:
        try:
            content = resp.msg.to_xml().encode('utf-8')
        except Exception:
            content = None
    if content is not None:
        path.write_bytes(content)
        logger.info(f'Wrote raw SDMX: {path}')
    else:
        logger.warning(f'Could not extract raw content for {path.name}')

def to_pandas_tidy(resp, dataset_id: str, rename_map: dict | None = None, add_cols: dict | None = None) -> pd.DataFrame:
    s = resp.to_pandas()
    if isinstance(s, pd.Series):
        df = s.to_frame('value')
    else:
        if 'value' not in s.columns and s.shape[1] == 1:
            s.columns = ['value']
        df = s
    df = df.reset_index()
    # Standardize time column
    time_cols = [c for c in df.columns if c.upper() in ('TIME_PERIOD', 'TIME')]
    tc = time_cols[0] if time_cols else ('time' if 'time' in df.columns else None)
    if tc is None:
        raise RuntimeError(f'Could not find time column in {dataset_id}; got columns: {df.columns.tolist()}')
    df = df.rename(columns={tc: 'time'})
    # Common renames
    if 'sect' in df.columns and 'sector' not in df.columns:
        df = df.rename(columns={'sect': 'sector'})
    if 'direction' in df.columns and 'direct' not in df.columns:
        df = df.rename(columns={'direction': 'direct'})
    if 'DIRECT' in df.columns and 'direct' not in df.columns:
        df = df.rename(columns={'DIRECT': 'direct'})
    if 'SECTOR' in df.columns and 'sector' not in df.columns:
        df = df.rename(columns={'SECTOR': 'sector'})
    if rename_map:
        df = df.rename(columns=rename_map)
    # Parse time to quarterly Period
    df['time'] = pd.PeriodIndex(pd.to_datetime(df['time']), freq='Q')
    # Add dataset and last update
    try:
        lu = resp.msg.header.prepared.isoformat()
    except Exception:
        lu = None
    df['dataset'] = dataset_id
    df['last_update'] = lu
    # Add constant columns if missing
    if add_cols:
        for k, v in add_cols.items():
            if k not in df.columns:
                df[k] = v
    return df

def filter_preferred_units(df: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    if 'unit' not in df.columns:
        return df
    present = list(pd.unique(df['unit']))
    for u in candidates:
        if u in present:
            return df[df['unit'] == u].copy()
    eur_units = [u for u in present if isinstance(u, str) and 'EUR' in u]
    if eur_units:
        return df[df['unit'].isin(eur_units)].copy()
    return df

def summarize_coverage(df: pd.DataFrame, label: str):
    tmin = str(df['time'].min()) if df is not None and not df.empty else 'NA'
    tmax = str(df['time'].max()) if df is not None and not df.empty else 'NA'
    print(f"{label}: rows={len(df) if df is not None else 0:,}, time=[{tmin} → {tmax}]")

# Placeholders for dataframes
df_nf = None
df_f_tr = None
df_f_bs = None
df_f_gl = None
df_f_oc = None
df_qsa = None
"""

    cell2 = r"""
# Cell 2 — Eurostat pull: non-financial (nasq_10_nf_tr)
flow = 'NASQ_10_NF_TR'
key = {
    'freq': FREQ,
    'geo': GEO,
    's_adj': S_ADJ,
    'sector': '+'.join(SECTORS),
    'na_item': '+'.join(NF_ITEMS),
    'direct': 'RECV+PAID',
    'unit': 'CP_MEUR+MIO_EUR',
}
resp_nf = sdmx_get(estat, flow, key=key, params={'compressed': True})
raw_fn = DIR_RAW_ESTAT / f'{flow}_{now_iso()}.xml'
sdmx_save_raw(resp_nf, raw_fn)
df_nf = to_pandas_tidy(resp_nf, dataset_id=flow)
# Standardize direct column name
if 'direct' not in df_nf.columns:
    for alt in ['direction', 'DIRECT', 'Direction']:
        if alt in df_nf.columns:
            df_nf = df_nf.rename(columns={alt: 'direct'})
            break
if 'direct' not in df_nf.columns:
    raise RuntimeError('DIRECT dimension not found in nasq_10_nf_tr pull')
# Keep preferred units
df_nf = filter_preferred_units(df_nf, UNITS_PREFERRED)
# Ensure essential columns exist
expected_cols = {'time','sector','na_item','direct','s_adj','unit','value'}
missing = expected_cols - set(df_nf.columns)
if missing:
    raise RuntimeError(f'Missing expected columns in nasq_10_nf_tr: {missing}')
df_nf.head(3)
"""

    cell3 = r"""
# Cell 3 — Eurostat pull: financial transactions (nasq_10_f_tr)
flow = 'NASQ_10_F_TR'
key = {
    'freq': FREQ,
    'geo': GEO,
    's_adj': S_ADJ,
    'sector': '+'.join(SECTORS),
    'unit': 'CP_MEUR+MIO_EUR',
}
resp_ftr = sdmx_get(estat, flow, key=key, params={'compressed': True})
raw_fn = DIR_RAW_ESTAT / f'{flow}_{now_iso()}.xml'
sdmx_save_raw(resp_ftr, raw_fn)
df_f_tr = to_pandas_tidy(resp_ftr, dataset_id=flow)
# Add direct column (not applicable) to align schema where needed
if 'direct' not in df_f_tr.columns:
    df_f_tr['direct'] = pd.NA
df_f_tr = filter_preferred_units(df_f_tr, UNITS_PREFERRED)
# Check B9F present
if not (df_f_tr.get('na_item') == 'B9F').any():
    warnings.warn('B9F not found in nasq_10_f_tr pull; Net lending QC may be limited.')
df_f_tr.head(3)
"""

    cell4 = r"""
# Cell 4 — Eurostat pull: financial balance sheets (nasq_10_f_bs)
flow = 'NASQ_10_F_BS'
key = {
    'freq': FREQ,
    'geo': GEO,
    's_adj': S_ADJ,
    'sector': '+'.join(SECTORS),
    'unit': 'CP_MEUR+MIO_EUR',
}
resp_fbs = sdmx_get(estat, flow, key=key, params={'compressed': True})
raw_fn = DIR_RAW_ESTAT / f'{flow}_{now_iso()}.xml'
sdmx_save_raw(resp_fbs, raw_fn)
df_f_bs = to_pandas_tidy(resp_fbs, dataset_id=flow)
df_f_bs = filter_preferred_units(df_f_bs, UNITS_PREFERRED)
df_f_bs.head(3)
"""

    cell5 = r"""
# Cell 5 — Optional: Revaluations & other changes (nasq_10_f_gl, nasq_10_f_oc)
for flow in ['NASQ_10_F_GL', 'NASQ_10_F_OC']:
    try:
        key = {
            'freq': FREQ,
            'geo': GEO,
            's_adj': S_ADJ,
            'sector': '+'.join(SECTORS),
            'unit': 'CP_MEUR+MIO_EUR',
        }
        resp = sdmx_get(estat, flow, key=key, params={'compressed': True})
        raw_fn = DIR_RAW_ESTAT / f'{flow}_{now_iso()}.xml'
        sdmx_save_raw(resp, raw_fn)
        df_tmp = to_pandas_tidy(resp, dataset_id=flow)
        df_tmp = filter_preferred_units(df_tmp, UNITS_PREFERRED)
        if flow.endswith('_gl'):
            df_f_gl = df_tmp
        else:
            df_f_oc = df_tmp
        logger.info(f'Pulled optional dataset {flow}: {len(df_tmp):,} rows')
    except Exception as e:
        warnings.warn(f'Skipping optional dataset {flow} due to: {e}')
        continue

df_f_gl.head(3) if isinstance(df_f_gl, pd.DataFrame) else None
"""

    cell6 = r"""
# Cell 6 — ECB QSA whom-to-whom (F2,F3,F4; sectors S11,S12,S13,S14_S15; ref area PL)
def try_fetch_qsa():
    mappings = [
        {'freq':'FREQ','area':'REF_AREA','ref':'REF_SECTOR','cp':'COUNTERPART_SECTOR','instr':'INSTR_ASSET'},
        {'freq':'FREQ','area':'REF_AREA','ref':'SECTOR','cp':'COUNTERPART_SECTOR','instr':'INSTR_ASSET'},
        {'freq':'FREQ','area':'REF_AREA','ref':'REF_SECTOR','cp':'C_SECTOR','instr':'INSTR_ASSET'},
        {'freq':'FREQ','area':'REF_AREA','ref':'SECTOR','cp':'C_SECTOR','instr':'INSTRUMENT'},
    ]
    last_exc = None
    for m in mappings:
        key = {
            m['freq']: 'Q',
            m['area']: GEO,
            m['ref']: '+'.join([s for s in SECTORS if s != 'S1']),
            m['cp']: '+'.join([s for s in SECTORS if s != 'S1']),
            m['instr']: 'F2+F3+F4',
        }
        try:
            resp = sdmx_get(ecb, 'QSA', key=key, params={'compressed': True})
            return resp
        except Exception as e:
            last_exc = e
            logger.warning(f'ECB QSA fetch failed with mapping {m}: {e}')
            continue
    raise last_exc if last_exc else RuntimeError('ECB QSA fetch failed with all mappings')

resp_qsa = try_fetch_qsa()
raw_fn = DIR_RAW_ECB / f"QSA_{now_iso()}.xml"
sdmx_save_raw(resp_qsa, raw_fn)
df_qsa = resp_qsa.to_pandas()
if isinstance(df_qsa, pd.Series):
    df_qsa = df_qsa.to_frame('value')
df_qsa = df_qsa.reset_index()
cols = {c.upper(): c for c in df_qsa.columns}
def find_col(substrs, exclude=None):
    exclude = exclude or []
    for U, c in cols.items():
        if all(s in U for s in substrs) and not any(e in U for e in exclude):
            return c
    return None
col_time = find_col(['TIME']) or find_col(['TIME_PERIOD']) or 'time'
if col_time not in df_qsa.columns:
    raise RuntimeError(f'ECB QSA: time column not found; got {df_qsa.columns.tolist()}')
col_ref = find_col(['SECTOR'], exclude=['COUNTER']) or find_col(['REF','SECTOR'])
col_cp = find_col(['COUNTER','SECTOR']) or find_col(['C_SECTOR'])
col_instr = find_col(['INSTR'])
col_pos = find_col(['A','L']) or find_col(['POSITION']) or find_col(['A_L'])
col_sf = find_col(['STOCK']) or find_col(['TRANSACT']) or find_col(['FLOW'])
col_unit = find_col(['UNIT']) or find_col(['MEASURE'])
ren = {}
ren[col_time] = 'time'
if col_ref: ren[col_ref] = 'ref_sector'
if col_cp: ren[col_cp] = 'cp_sector'
if col_instr: ren[col_instr] = 'instrument'
if col_pos: ren[col_pos] = 'entry'
if col_sf: ren[col_sf] = 'stock_flow_flag'
if col_unit: ren[col_unit] = 'unit'
df_qsa = df_qsa.rename(columns=ren)
df_qsa['time'] = pd.PeriodIndex(pd.to_datetime(df_qsa['time']), freq='Q')
try:
    lu = resp_qsa.msg.header.prepared.isoformat()
except Exception:
    lu = None
df_qsa['dataset'] = 'QSA'
df_qsa['last_update'] = lu
keep_cols = ['time','ref_sector','cp_sector','instrument','entry','stock_flow_flag','unit','value','dataset','last_update']
df_qsa = df_qsa[[c for c in keep_cols if c in df_qsa.columns]]
df_qsa.head(3)
"""

    cell7 = r"""
# Cell 7 — ESA→SFC mapping layer
sector_map = {
    'S14_S15': 'HH',
    'S11': 'NFC',
    'S12': 'FC',
    'S13': 'GOV',
    'S2': 'ROW',
    'S1': 'TOT',
}
flow_map_examples = {
    ('P31','HH'): 'C',
    ('P3','GOV'): 'G',
    ('P51G', None): 'I',
    ('B6G','HH'): 'YD_HH',
    ('B8G','HH'): 'S_HH',
    ('B9', None): 'NL_S',
    ('B9F', None): 'NLF_S',
}
def map_item_sfc(na_item: str, sector_code: str | None) -> str | None:
    sec_sfc = sector_map.get(sector_code) if sector_code else None
    key = (na_item, sec_sfc)
    if key in flow_map_examples:
        return flow_map_examples[key]
    key_any = (na_item, None)
    return flow_map_examples.get(key_any)

if isinstance(df_nf, pd.DataFrame):
    df_nf['sector_sfc'] = df_nf['sector'].map(sector_map)
    df_nf['item_sfc'] = [map_item_sfc(a, b) for a, b in zip(df_nf.get('na_item', pd.Series([None]*len(df_nf))), df_nf.get('sector', pd.Series([None]*len(df_nf))))]
if isinstance(df_f_tr, pd.DataFrame):
    df_f_tr['sector_sfc'] = df_f_tr['sector'].map(sector_map)
    df_f_tr['item_sfc'] = [map_item_sfc(a, None) for a in df_f_tr.get('na_item', pd.Series([None]*len(df_f_tr)))]
if isinstance(df_f_bs, pd.DataFrame):
    df_f_bs['sector_sfc'] = df_f_bs['sector'].map(sector_map)
    df_f_bs['item_sfc'] = [map_item_sfc(a, None) for a in df_f_bs.get('na_item', pd.Series([None]*len(df_f_bs)))]

df_nf[['sector','sector_sfc','na_item','item_sfc']].drop_duplicates().head(8)
"""

    cell8 = r"""
# Cell 8 — QC checks
qc_summary = {}
tol = 0.5  # tolerance in dataset's unit (million EUR)

def qc_resources_uses(df: pd.DataFrame, tol: float = 0.5):
    if df is None or df.empty:
        return {'available': False, 'message': 'No non-financial data.'}
    tmp = df.copy()
    def norm_direct(x: str):
        if pd.isna(x):
            return x
        x = str(x).upper()
        if x.startswith('R') or 'REC' in x or 'RES' in x:
            return 'RESOURCES'
        if x.startswith('P') or 'PAY' in x or 'USE' in x:
            return 'USES'
        return x
    tmp['direct_norm'] = tmp['direct'].map(norm_direct)
    tmp = tmp[tmp['sector'] != 'S1']
    grp = tmp.groupby(['time','na_item','unit','s_adj','direct_norm'], dropna=False)['value'].sum().unstack('direct_norm')
    grp = grp.fillna(0.0)
    if 'RESOURCES' not in grp.columns or 'USES' not in grp.columns:
        return {'available': False, 'message': 'DIRECT categories missing after normalization.'}
    grp['diff'] = grp['RESOURCES'] - grp['USES']
    grp['abs_diff'] = grp['diff'].abs()
    max_abs = float(grp['abs_diff'].max()) if not grp.empty else 0.0
    failures = grp[grp['abs_diff'] > tol].sort_values('abs_diff', ascending=False).head(10)
    failures_list = [
        {'time': str(ix[0]), 'na_item': ix[1], 'unit': ix[2], 'abs_diff': float(row['abs_diff'])}
        for ix, row in failures.iterrows()
    ]
    return {'available': True, 'max_abs_diff': max_abs, 'n_fail': int((grp['abs_diff'] > tol).sum()), 'fail_examples': failures_list}

qc1 = qc_resources_uses(df_nf, tol=tol)
qc_summary['resources_vs_uses'] = qc1

def qc_net_lending(df_nf: pd.DataFrame, df_ftr: pd.DataFrame, tol: float = 0.5):
    if df_nf is None or df_nf.empty or df_ftr is None or df_ftr.empty:
        return {'available': False, 'message': 'Missing NF or F_TR data.'}
    nf = df_nf[df_nf['na_item'] == 'B9'].copy()
    f = df_ftr[df_ftr['na_item'].isin(['B9F','B9FX9'])].copy()
    if nf.empty or f.empty:
        return {'available': False, 'message': 'B9 or B9F/B9FX9 missing.'}
    f_piv = f.pivot_table(index=['time','sector','unit','s_adj'], columns='na_item', values='value', aggfunc='sum')
    f_piv = f_piv.fillna(0.0)
    nf_k = nf[['time','sector','unit','s_adj','value']].rename(columns={'value':'B9'})
    m = nf_k.merge(f_piv, left_on=['time','sector','unit','s_adj'], right_index=True, how='left')
    for col in ['B9F','B9FX9']:
        if col not in m.columns:
            m[col] = 0.0
    m['diff'] = m['B9'] - (m['B9F'] + m['B9FX9'])
    m['abs_diff'] = m['diff'].abs()
    max_abs = float(m['abs_diff'].max()) if not m.empty else 0.0
    offenders = m.sort_values('abs_diff', ascending=False).head(10)
    offenders_list = [
        {'time': str(r.time), 'sector': r.sector, 'unit': r.unit, 'abs_diff': float(r.abs_diff)}
        for r in offenders.itertuples(index=False)
    ]
    return {'available': True, 'max_abs_diff': max_abs, 'n_fail': int((m['abs_diff'] > tol).sum()), 'fail_examples': offenders_list}

qc2 = qc_net_lending(df_nf, df_f_tr, tol=tol)
qc_summary['net_lending_recon'] = qc2

def qc_stocks(df_bs: pd.DataFrame, df_ftr: pd.DataFrame, df_gl: pd.DataFrame | None, df_oc: pd.DataFrame | None):
    if df_bs is None or df_bs.empty or df_ftr is None or df_ftr.empty:
        return {'available': False, 'message': 'Missing balance sheets or financial transactions.'}
    def inst_from_na(x):
        try:
            if isinstance(x, str) and x.startswith('AF'):
                return 'F' + x[2:]
        except Exception:
            pass
        return None
    bs = df_bs.copy()
    if 'na_item' not in bs.columns:
        return {'available': False, 'message': 'No na_item in balance sheets.'}
    bs['instr'] = bs['na_item'].map(inst_from_na)
    bs = bs[bs['instr'].notna()].copy()
    bs = bs.sort_values(['sector','instr','unit','s_adj','time'])
    bs['delta'] = bs.groupby(['sector','instr','unit','s_adj'])['value'].diff()
    fl = df_ftr.copy()
    fl = fl[fl['na_item'].str.startswith('F', na=False)].copy()
    fl = fl.rename(columns={'na_item':'instr'})
    fl_g = fl.groupby(['time','sector','instr','unit','s_adj'], as_index=False)['value'].sum()
    m = bs.merge(fl_g, on=['time','sector','instr','unit','s_adj'], how='left', suffixes=('_bs','_f'))
    m['value_f'] = m['value_f'].fillna(0.0)
    def prep_k(dfk, code_prefix):
        if dfk is None or dfk.empty:
            return None
        d = dfk.copy()
        if 'na_item' not in d.columns:
            return None
        d = d.rename(columns={'na_item':'k_item'})
        return d
    k7 = prep_k(df_gl, 'K7')
    k1_6 = prep_k(df_oc, 'K')
    if k7 is not None:
        k7['instr'] = k7.get('k_item').map(lambda x: 'F' + x.split('_')[-1] if isinstance(x, str) and '_' in x and x.split('_')[-1].startswith('F') else None)
        m = m.merge(k7[['time','sector','unit','s_adj','instr','value']], on=['time','sector','unit','s_adj','instr'], how='left')
        m = m.rename(columns={'value':'k7'})
    else:
        m['k7'] = 0.0
    if k1_6 is not None:
        k1_6['instr'] = k1_6.get('k_item').map(lambda x: 'F' + x.split('_')[-1] if isinstance(x, str) and '_' in x and x.split('_')[-1].startswith('F') else None)
        m = m.merge(k1_6[['time','sector','unit','s_adj','instr','value']], on=['time','sector','unit','s_adj','instr'], how='left')
        m = m.rename(columns={'value':'k1_6'})
    else:
        m['k1_6'] = 0.0
    m['k7'] = m.get('k7', 0.0).fillna(0.0)
    m['k1_6'] = m.get('k1_6', 0.0).fillna(0.0)
    m['residual'] = m['delta'] - (m['value_f'] + m['k7'] + m['k1_6'])
    m['abs_residual'] = m['residual'].abs()
    by_combo = m.groupby(['sector','instr','unit','s_adj'])['abs_residual'].max().reset_index().sort_values('abs_residual', ascending=False)
    max_abs = float(by_combo['abs_residual'].max()) if not by_combo.empty else 0.0
    top = by_combo.head(10)
    examples = [
        {'sector': r.sector, 'instr': r.instr, 'unit': r.unit, 'max_abs_residual': float(r.abs_residual)}
        for r in top.itertuples(index=False)
    ]
    return {'available': True, 'max_abs_residual': max_abs, 'examples': examples}

qc3 = qc_stocks(df_f_bs, df_f_tr, df_f_gl, df_f_oc)
qc_summary['stocks_reconciliation'] = qc3

qc_path = DIR_PROCESSED / 'qc_summary.json'
qc_path.write_text(json.dumps(qc_summary, indent=2))

print('QC-1 Resources=Uses: available=', qc1.get('available'), ', max_abs_diff=', qc1.get('max_abs_diff'))
if qc1.get('available'):
    print('  Fail examples:', qc1.get('fail_examples')[:5])
print('QC-2 Net lending: available=', qc2.get('available'), ', max_abs_diff=', qc2.get('max_abs_diff'))
if qc2.get('available'):
    print('  Offenders:', qc2.get('fail_examples')[:5])
print('QC-3 Stocks recon: available=', qc3.get('available'), ', max_abs_residual=', qc3.get('max_abs_residual', qc3.get('max_abs', None)))
if qc3.get('available'):
    print('  Examples:', qc3.get('examples')[:5])
"""

    cell9 = r'''
# Cell 9 — Persist tidy tables & README; print acceptance outputs
# Save Parquet files
if isinstance(df_nf, pd.DataFrame):
    df_nf.to_parquet(DIR_PROCESSED / 'estat_nasq_10_nf_tr_PL.parquet', index=False)
if isinstance(df_f_tr, pd.DataFrame):
    df_f_tr.to_parquet(DIR_PROCESSED / 'estat_nasq_10_f_tr_PL.parquet', index=False)
if isinstance(df_f_bs, pd.DataFrame):
    df_f_bs.to_parquet(DIR_PROCESSED / 'estat_nasq_10_f_bs_PL.parquet', index=False)
if isinstance(df_f_gl, pd.DataFrame) and not df_f_gl.empty:
    df_f_gl.to_parquet(DIR_PROCESSED / 'estat_nasq_10_f_gl_PL.parquet', index=False)
if isinstance(df_f_oc, pd.DataFrame) and not df_f_oc.empty:
    df_f_oc.to_parquet(DIR_PROCESSED / 'estat_nasq_10_f_oc_PL.parquet', index=False)
if isinstance(df_qsa, pd.DataFrame):
    df_qsa.to_parquet(DIR_PROCESSED / 'ecb_QSA_PL.parquet', index=False)

# README_data.md
readme_path = ROOT / 'README_data.md'
last_updates = {}
for name, df in [('nasq_10_nf_tr', df_nf), ('nasq_10_f_tr', df_f_tr), ('nasq_10_f_bs', df_f_bs), ('nasq_10_f_gl', df_f_gl), ('nasq_10_f_oc', df_f_oc), ('QSA', df_qsa)]:
    if isinstance(df, pd.DataFrame) and 'last_update' in df.columns:
        lu = df['last_update'].dropna().unique().tolist()
        last_updates[name] = ', '.join(lu[:1])
    else:
        last_updates[name] = ''
readme = f"""# Data spine — Poland (SDMX)
Run timestamp (UTC): {now_iso()}

Sources: Eurostat (ESTAT), ECB (QSA) via SDMX.

Datasets and filters:
- Eurostat nasq_10_nf_tr: freq=Q, geo=PL, s_adj=NSA, unit in {{CP_MEUR,MIO_EUR}}, sectors={SECTORS}, na_item includes minimal core set. Last update: {last_updates.get('nasq_10_nf_tr','')}
- Eurostat nasq_10_f_tr: freq=Q, geo=PL, s_adj=NSA, unit in {{CP_MEUR,MIO_EUR}}, sectors={SECTORS}. Last update: {last_updates.get('nasq_10_f_tr','')}
- Eurostat nasq_10_f_bs: freq=Q, geo=PL, s_adj=NSA, unit in {{CP_MEUR,MIO_EUR}}, sectors={SECTORS}. Last update: {last_updates.get('nasq_10_f_bs','')}
- Eurostat nasq_10_f_gl (optional): freq=Q, geo=PL, s_adj=NSA. Last update: {last_updates.get('nasq_10_f_gl','')}
- Eurostat nasq_10_f_oc (optional): freq=Q, geo=PL, s_adj=NSA. Last update: {last_updates.get('nasq_10_f_oc','')}
- ECB QSA whom-to-whom: ref_area=PL, freq=Q, instruments F2,F3,F4; ref/cp sectors in {{S11,S12,S13,S14_S15}}. Last update: {last_updates.get('QSA','')}

Outputs:
- data/processed/*.parquet; raw SDMX payloads under data/raw/estat and data/raw/ecb; QC summary at data/processed/qc_summary.json.
"""
readme_path.write_text(readme)

print('Row counts and time coverage:')
if isinstance(df_nf, pd.DataFrame): summarize_coverage(df_nf, 'nasq_10_nf_tr')
if isinstance(df_f_tr, pd.DataFrame): summarize_coverage(df_f_tr, 'nasq_10_f_tr')
if isinstance(df_f_bs, pd.DataFrame): summarize_coverage(df_f_bs, 'nasq_10_f_bs')
if isinstance(df_qsa, pd.DataFrame): summarize_coverage(df_qsa, 'ECB QSA')

print('\nQC summary:')
print(json.dumps(qc_summary, indent=2))

print('\nHead of non-financial tidy table:')
if isinstance(df_nf, pd.DataFrame):
    display(df_nf[['time','sector','na_item','direct','unit','value']].head())
else:
    print('Non-financial DF not available')
'''

    md_verify = "### VERIFY_STEP_1 — SDMX-only verification"

    cell_verify = r"""
# Verification block: builds a concise report and saves data/processed/verification_report.json
ver = {
    'run_timestamp': now_iso(),
    'row_coverage': {},
    'checks': {},
    'provider_last_update': {},
    'anchor': {}
}
warns = []

def row_cov(df: pd.DataFrame, name: str):
    try:
        if df is None or df.empty:
            return {'rows': 0, 'first_quarter': None, 'last_quarter': None, 'distinct_sectors': 0, 'units': []}
        tmin = str(df['time'].min())
        tmax = str(df['time'].max())
        if 'sector' in df.columns:
            dsec = int(df['sector'].nunique())
        elif {'ref_sector','cp_sector'} <= set(df.columns):
            dsec = int(pd.Index(df['ref_sector']).union(pd.Index(df['cp_sector'])).nunique())
        else:
            dsec = 0
        units = sorted([str(u) for u in pd.unique(df['unit'])]) if 'unit' in df.columns else []
        return {'rows': int(len(df)), 'first_quarter': tmin, 'last_quarter': tmax, 'distinct_sectors': dsec, 'units': units}
    except Exception as e:
        warns.append(f'Row coverage failed for {name}: {e}')
        return {'error': str(e)}

ver['row_coverage']['nasq_10_nf_tr'] = row_cov(df_nf, 'nasq_10_nf_tr')
ver['row_coverage']['nasq_10_f_tr'] = row_cov(df_f_tr, 'nasq_10_f_tr')
ver['row_coverage']['nasq_10_f_bs'] = row_cov(df_f_bs, 'nasq_10_f_bs')
ver['row_coverage']['ecb_QSA'] = row_cov(df_qsa, 'ecb_QSA')

def verify_time_geo(df: pd.DataFrame, name: str):
    ok_time = False
    ok_geo = True
    try:
        if df is not None and not df.empty:
            ok_time = isinstance(df['time'].dtype, pd.PeriodDtype) and str(df['time'].dtype.freq).startswith('Q')
            if 'geo' in df.columns:
                geos = set(df['geo'].dropna().unique())
                ok_geo = (geos == {GEO})
    except Exception as e:
        warns.append(f'Time/geo verification failed for {name}: {e}')
    return {'quarterly_period': bool(ok_time), 'geo_PL_only': bool(ok_geo)}

ver['checks']['time_geo'] = {
    'nasq_10_nf_tr': verify_time_geo(df_nf, 'nasq_10_nf_tr'),
    'nasq_10_f_tr': verify_time_geo(df_f_tr, 'nasq_10_f_tr'),
    'nasq_10_f_bs': verify_time_geo(df_f_bs, 'nasq_10_f_bs'),
    'ecb_QSA': verify_time_geo(df_qsa, 'ecb_QSA'),
}

try:
    qcA = qc_resources_uses(df_nf, tol=0.5)
    ver['checks']['identity_A_resources_uses'] = qcA
except Exception as e:
    warns.append(f'Identity A failed: {e}')
    ver['checks']['identity_A_resources_uses'] = {'error': str(e)}

try:
    qcB = qc_net_lending(df_nf, df_f_tr, tol=0.5)
    ver['checks']['identity_B_net_lending'] = qcB
except Exception as e:
    warns.append(f'Identity B failed: {e}')
    ver['checks']['identity_B_net_lending'] = {'error': str(e)}

try:
    qcC = qc_stocks(df_f_bs, df_f_tr, df_f_gl, df_f_oc)
    ver['checks']['identity_C_stocks_bridge'] = qcC
except Exception as e:
    warns.append(f'Identity C failed: {e}')
    ver['checks']['identity_C_stocks_bridge'] = {'error': str(e)}

try:
    cross = {}
    if isinstance(df_qsa, pd.DataFrame) and not df_qsa.empty:
        qsa = df_qsa.copy()
        qsa = qsa[(qsa.get('instrument') == 'F4') &
                  (qsa.get('ref_sector').isin(['S12','S14_S15','S11','S13'])) &
                  (qsa.get('cp_sector').isin(['S12','S14_S15','S11','S13']))]
        aset = qsa[(qsa.get('ref_sector') == 'S12') & (qsa.get('cp_sector') == 'S14_S15') & (qsa.get('entry','').astype(str).str.upper().str.startswith('A'))]
        lset = qsa[(qsa.get('ref_sector') == 'S14_S15') & (qsa.get('cp_sector') == 'S12') & (qsa.get('entry','').astype(str).str.upper().str.startswith('L'))]
        if not aset.empty and not lset.empty:
            tq = min(aset['time'].max(), lset['time'].max())
            a_val = float(aset[aset['time'] == tq]['value'].sum())
            l_val = float(lset[lset['time'] == tq]['value'].sum())
            residual = a_val - l_val
            cross = {
                'available': True,
                'time': str(tq),
                'instrument': 'F4',
                'dims': {
                    'assets': {'ref_sector':'S12','cp_sector':'S14_S15','entry':'A'},
                    'liabilities': {'ref_sector':'S14_S15','cp_sector':'S12','entry':'L'}
                },
                'unit': (aset['unit'].dropna().unique().tolist() or lset['unit'].dropna().unique().tolist() or [''])[0],
                'asset_value': a_val,
                'liability_value': l_val,
                'residual': residual
            }
        else:
            cross = {'available': False, 'message': 'Required entries not found (A vs L).'}
    else:
        cross = {'available': False, 'message': 'QSA dataset missing.'}
    ver['checks']['cross_source_w2w_loans'] = cross
except Exception as e:
    warns.append(f'Cross-source coherence failed: {e}')
    ver['checks']['cross_source_w2w_loans'] = {'error': str(e)}

try:
    totals_items = ['B1G','D1','P3','P31','P51G','B9']
    tot_res = {}
    if isinstance(df_nf, pd.DataFrame) and not df_nf.empty:
        tmp = df_nf[df_nf['na_item'].isin(totals_items)].copy()
        res = []
        for item in totals_items:
            sub = tmp[tmp['na_item'] == item]
            if sub.empty:
                res.append({'na_item': item, 'max_abs_diff': None})
                continue
            s1 = sub[sub['sector']=='S1'].groupby(['time','direct','unit','s_adj'], dropna=False)['value'].sum().rename('S1')
            sr = sub[sub['sector'].isin(['S11','S12','S13','S14_S15'])].groupby(['time','direct','unit','s_adj'], dropna=False)['value'].sum().rename('sum_residents')
            m = s1.to_frame().join(sr, how='inner')
            m['diff'] = m['S1'] - m['sum_residents']
            mad = float(m['diff'].abs().max()) if not m.empty else None
            res.append({'na_item': item, 'max_abs_diff': mad})
        tot_res = {r['na_item']: r['max_abs_diff'] for r in res}
    else:
        tot_res = {'error': 'Non-financial dataset missing.'}
    ver['checks']['totals_S1_equals_sum_residents'] = tot_res
except Exception as e:
    warns.append(f'Totals check failed: {e}')
    ver['checks']['totals_S1_equals_sum_residents'] = {'error': str(e)}

try:
    def lu(df):
        if isinstance(df, pd.DataFrame) and 'last_update' in df.columns:
            vals = [v for v in df['last_update'].dropna().unique().tolist() if v]
            return vals[0] if vals else None
        return None
    ver['provider_last_update'] = {
        'nasq_10_nf_tr': lu(df_nf),
        'nasq_10_f_tr': lu(df_f_tr),
        'nasq_10_f_bs': lu(df_f_bs),
        'nasq_10_f_gl': lu(df_f_gl),
        'nasq_10_f_oc': lu(df_f_oc),
        'ecb_QSA': lu(df_qsa),
    }
except Exception as e:
    warns.append(f'Provider last-update capture failed: {e}')

try:
    anchor = {}
    if isinstance(df_nf, pd.DataFrame):
        a = df_nf[(df_nf['na_item']=='B9') & (df_nf['sector']=='S13')].copy()
        if not a.empty:
            tq = a['time'].max()
            r = a[a['time']==tq].sort_values('time').head(1).iloc[0]
            anchor = {
                'time': str(r['time']),
                'sector': 'S13',
                'na_item': 'B9',
                'direct': r.get('direct', None),
                'unit': r.get('unit', None),
                'value': float(r['value']),
                'sdmx_key': {
                    'dataset': 'nasq_10_nf_tr',
                    'freq': FREQ, 's_adj': S_ADJ, 'geo': GEO, 'sector': 'S13', 'na_item': 'B9', 'direct': r.get('direct', None), 'unit': r.get('unit', None)
                }
            }
        else:
            anchor = {'message': 'No B9 S13 rows found.'}
    ver['anchor'] = anchor
except Exception as e:
    warns.append(f'Anchor selection failed: {e}')
    ver['anchor'] = {'error': str(e)}

(DIR_PROCESSED).mkdir(parents=True, exist_ok=True)
ver_path = DIR_PROCESSED / 'verification_report.json'
ver_path.write_text(json.dumps({**ver, 'warnings': warns}, indent=2))

print('## VERIFY_STEP_1')
print('\n### Row counts & coverage')
for k, v in ver['row_coverage'].items():
    print(f"- {k}: rows={v.get('rows')}, first={v.get('first_quarter')}, last={v.get('last_quarter')}, sectors={v.get('distinct_sectors')}, units={v.get('units')}")
print('\n### Time/geo checks')
for k, v in ver['checks']['time_geo'].items():
    print(f"- {k}: quarterly_period={v.get('quarterly_period')}, geo_PL_only={v.get('geo_PL_only')}")
print('\n### Identity A — Resources=Uses')
ia = ver['checks'].get('identity_A_resources_uses', {})
print(f"max_abs_diff={ia.get('max_abs_diff')}, n_fail={ia.get('n_fail')}\nFail examples: {ia.get('fail_examples')}")
print('\n### Identity B — Net lending recon')
ib = ver['checks'].get('identity_B_net_lending', {})
print(f"max_abs_diff={ib.get('max_abs_diff')}, n_fail={ib.get('n_fail')}\nOffenders: {ib.get('fail_examples')}")
print('\n### Identity C — Stocks/flows bridge')
ic = ver['checks'].get('identity_C_stocks_bridge', {})
print(f"available={ic.get('available')}, max_abs_residual={ic.get('max_abs_residual')}")
print('\n### Cross-source coherence — Loans F4 (ECB QSA)')
cs = ver['checks'].get('cross_source_w2w_loans', {})
print(f"available={cs.get('available')}, time={cs.get('time')}, residual={cs.get('residual')}\nDims: {cs.get('dims')}, unit={cs.get('unit')}")
print('\n### Totals check — S1 vs residents')
tt = ver['checks'].get('totals_S1_equals_sum_residents', {})
print(json.dumps(tt, indent=2))
print('\n### Provider last-update stamps')
print(json.dumps(ver['provider_last_update'], indent=2))
print('\n### Anchor — latest B9, S13 (General government)')
print(json.dumps(ver['anchor'], indent=2))
if warns:
    print('\n### WARN')
    for w in warns:
        print('-', w)
"""

    nb.cells = [
        new_markdown_cell(md_intro),
        new_code_cell(cell1),
        new_code_cell(cell2),
        new_code_cell(cell3),
        new_code_cell(cell4),
        new_code_cell(cell5),
        new_code_cell(cell6),
        new_code_cell(cell7),
        new_code_cell(cell8),
        new_code_cell(cell9),
        new_markdown_cell(md_verify),
        new_code_cell(cell_verify),
    ]
    nb.metadata.setdefault('kernelspec', {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3',
    })
    nb.metadata.setdefault('language_info', {'name': 'python'})
    return nb


def main():
    nb = build_notebook()
    out = Path('notebooks/01_data_spine_pl.ipynb')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f'Wrote notebook: {out}')


if __name__ == '__main__':
    main()
