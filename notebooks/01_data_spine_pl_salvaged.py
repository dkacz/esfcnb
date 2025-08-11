# Salvaged from malformed notebook at 20250810T114514Z

# --- code cell 1 ---
# Cell 1 — Imports & config\nimport warnings\nimport sys, time, json, logging\nfrom pathlib import Path\nfrom datetime import datetime, timezone\n\nimport pandas as pd\nimport numpy as np\nfrom pandasdmx import Request\nfrom dateutil import parser as dateparser\n\n# Version printouts\n

# --- code cell 2 ---
# Cell 2 — Eurostat pull: non-financial (nasq_10_nf_tr)\nflow = 'nasq_10_nf_tr'\nkey = {\n    'freq': FREQ,\n    'geo': GEO,\n    's_adj': S_ADJ,\n    'sector': '+'.join(SECTORS),\n    'na_item': '+'.join(NF_ITEMS),\n}\nresp_nf = sdmx_get(estat, flow, key=key, params={'compressed': True})\nraw_fn = DIR_RAW_ESTAT / f'{flow}_{now_iso()}.xml'\nsdmx_save_raw(resp_nf, raw_fn)\ndf_nf = to_pandas_tidy(resp_nf, dataset_id=flow)\n# Standardize direct column name\nif 'direct' not in df_nf.columns:\n

# --- code cell 3 ---
# Cell 3 — Eurostat pull: financial transactions (nasq_10_f_tr)\nflow = 'nasq_10_f_tr'\nkey = {\n    'freq': FREQ,\n    'geo': GEO,\n    's_adj': S_ADJ,\n    'sector': '+'.join(SECTORS),\n}\nresp_ftr = sdmx_get(estat, flow, key=key, params={'compressed': True})\nraw_fn = DIR_RAW_ESTAT / f'{flow}_{now_iso()}.xml'\nsdmx_save_raw(resp_ftr, raw_fn)\ndf_f_tr = to_pandas_tidy(resp_ftr, dataset_id=flow)\n# Add direct column (not applicable) to align schema where needed\nif 'direct' not in df_f_tr.columns:\n

# --- code cell 4 ---
# Cell 4 — Eurostat pull: financial balance sheets (nasq_10_f_bs)
flow = 'nasq_10_f_bs'
key = {
    'freq': FREQ,
    'geo': GEO,
    's_adj': S_ADJ,
    'sector': '+'.join(SECTORS),
}
resp_fbs = sdmx_get(estat, flow, key=key, params={'compressed': True})
raw_fn = DIR_RAW_ESTAT / f'{flow}_{now_iso()}.xml'
sdmx_save_raw(resp_fbs, raw_fn)
df_f_bs = to_pandas_tidy(resp_fbs, dataset_id=flow)
df_f_bs = filter_preferred_units(df_f_bs, UNITS_PREFERRED)
df_f_bs.head(3)

# --- code cell 5 ---
# Cell 5 — Optional: Revaluations & other changes (nasq_10_f_gl, nasq_10_f_oc)\n

# --- code cell 6 ---
# Cell 6 — ECB QSA whom-to-whom (F2,F3,F4; sectors S11,S12,S13,S14_S15; ref area PL)\ndef try_fetch_qsa():\n    # Try alternative dimension ID mappings for robustness\n    mappings = [\n        {'freq':'FREQ','area':'REF_AREA','ref':'REF_SECTOR','cp':'COUNTERPART_SECTOR','instr':'INSTR_ASSET'},\n        {'freq':'FREQ','area':'REF_AREA','ref':'SECTOR','cp':'COUNTERPART_SECTOR','instr':'INSTR_ASSET'},\n        {'freq':'FREQ','area':'REF_AREA','ref':'REF_SECTOR','cp':'C_SECTOR','instr':'INSTR_ASSET'},\n        {'freq':'FREQ','area':'REF_AREA','ref':'SECTOR','cp':'C_SECTOR','instr':'INSTRUMENT'},\n

# --- code cell 7 ---
# Cell 7 — ESA→SFC mapping layer\nsector_map = {\n    'S14_S15': 'HH',\n    'S11': 'NFC',\n    'S12': 'FC',\n    'S13': 'GOV',\n    'S2': 'ROW',\n    'S1': 'TOT',\n}\nflow_map_examples = {\n    # examples for later modeling\n    ('P31','HH'): 'C',\n    ('P3','GOV'): 'G',\n    ('P51G', None): 'I',\n    ('B6G','HH'): 'YD_HH',\n    ('B8G','HH'): 'S_HH',\n    ('B9', None): 'NL_S',\n    ('B9F', None): 'NLF_S',\n}\ndef map_item_sfc(na_item: str, sector_code: str | None) -> str | None:\n    sec_sfc = sector_map.get(sector_code) if sector_code else None\n    key = (na_item, sec_sfc)\n    if key in flow_map_examples:\n

# --- code cell 8 ---
# Cell 8 — QC checks\nqc_summary = {}\ntol = 0.5  # tolerance in dataset's unit (million EUR)\n\n# QC-1: Resources = Uses (by na_item, quarter)\ndef qc_resources_uses(df: pd.DataFrame, tol: float = 0.5):\n    if df is None or df.empty:\n        return {'available': False, 'message': 'No non-financial data.'}\n    tmp = df.copy()\n    # Normalize direct labels\n    def norm_direct(x: str):\n        if pd.isna(x):\n            return x\n        x = str(x).upper()\n        if x.startswith('R') or 'REC' in x or 'RES' in x:\n            return 'RESOURCES'\n        if x.startswith('P') or 'PAY' in x or 'USE' in x:\n            return 'USES'\n        return x\n

# --- code cell 9 ---
# Cell 9 — Persist tidy tables & README; print acceptance outputs\n# Save Parquet files\nif isinstance(df_nf, pd.DataFrame):\n    (DIR_PROCESSED / 'estat_nasq_10_nf_tr_PL.parquet').write_bytes(b'') if False else None\n    df_nf.to_parquet(DIR_PROCESSED / 'estat_nasq_10_nf_tr_PL.parquet', index=False)\nif isinstance(df_f_tr, pd.DataFrame):\n    df_f_tr.to_parquet(DIR_PROCESSED / 'estat_nasq_10_f_tr_PL.parquet', index=False)\nif isinstance(df_f_bs, pd.DataFrame):\n    df_f_bs.to_parquet(DIR_PROCESSED / 'estat_nasq_10_f_bs_PL.parquet', index=False)\nif isinstance(df_f_gl, pd.DataFrame) and not df_f_gl.empty:\n    df_f_gl.to_parquet(DIR_PROCESSED / 'estat_nasq_10_f_gl_PL.parquet', index=False)\nif isinstance(df_f_oc, pd.DataFrame) and not df_f_oc.empty:\n    df_f_oc.to_parquet(DIR_PROCESSED / 'estat_nasq_10_f_oc_PL.parquet', index=False)\nif isinstance(df_qsa, pd.DataFrame):\n    df_qsa.to_parquet(DIR_PROCESSED / 'ecb_QSA_PL.parquet', index=False)\n\n# README_data.md\nreadme_path = ROOT / 'README_data.md'\nlast_updates = {}\n

# --- code cell 10 ---
# Verification block: builds a concise report and saves data/processed/verification_report.json\nver = {\n    'run_timestamp': now_iso(),\n    'row_coverage': {},\n    'checks': {},\n    'provider_last_update': {},\n    'anchor': {}\n}\n

