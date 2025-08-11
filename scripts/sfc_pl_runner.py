from __future__ import annotations
import sys
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import nbformat
from nbclient import NotebookClient
import hashlib
import json

import pandas as pd
import yaml
from pandasdmx import Request

import sys, os
sys.path.append(os.path.abspath("."))
from src.sdmx_helpers import (
    get_dsd,
    fetch_series,
    build_key,
    ensure_dirs as _ensure_dirs,
    get_fetch_log,
    reset_fetch_log,
)


CFG_PATH = Path("config/sfc_pl_runner.yml")
LOG_PATH = Path("logs/sfc_pl_runner.log")
PROOF_PREFIX = "NOTEBOOK_RUN_PROOF "

DEFAULT_CFG = {
    "run": {"execute_notebook": True},
    "io": {
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "log_file": str(LOG_PATH),
        "notebook_path": "notebooks/01_data_spine_pl.ipynb",
    },
    "safety": {"destructive_ops": False},
}


def ensure_dirs(cfg: dict) -> None:
    for p in [cfg["io"]["raw_dir"], cfg["io"]["processed_dir"], CFG_PATH.parent, Path(cfg["io"]["log_file"]).parent]:
        Path(p).mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    if not CFG_PATH.exists():
        CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CFG_PATH.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(DEFAULT_CFG, fh, sort_keys=False)
    with CFG_PATH.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    # shallow merge defaults
    merged = DEFAULT_CFG.copy()
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and k in merged:
            mv = merged[k].copy()
            mv.update(v)
            merged[k] = mv
        else:
            merged[k] = v
    # Normalize IO paths to repository root so notebook CWD doesn't matter
    repo_root = Path(__file__).resolve().parent.parent
    io = merged.get("io", {})
    for k in ("raw_dir", "processed_dir", "log_file", "notebook_path"):
        v = io.get(k)
        if v is None:
            continue
        p = Path(v)
        if not p.is_absolute():
            p = repo_root / p
        io[k] = str(p)
    merged["io"] = io
    ensure_dirs(merged)
    return merged


def execute_notebook(cfg: dict) -> tuple[bool, str, Path]:
    nb_path = Path(cfg["io"].get("notebook_path", "notebooks/01_data_spine_pl.ipynb")).resolve()
    if not nb_path.exists():
        logging.error("notebook missing: %s", nb_path)
        return False, "missing_notebook", nb_path
    nb = nbformat.read(nb_path, as_version=4)
    client = NotebookClient(nb, timeout=900, kernel_name="python3", allow_errors=False)
    # run with the notebook's folder as working dir (avoid passing unsupported kwargs)
    try:
        prev_cwd = os.getcwd()
        # Execute with repository root as CWD so relative paths in the notebook (e.g., data/processed/...) resolve
        repo_root = Path(__file__).resolve().parent.parent
        os.chdir(str(repo_root))
        try:
            client.execute()
        finally:
            os.chdir(prev_cwd)
    except Exception as e:
        logging.exception("notebook execution failed: %s", e)
        # Save partial executed copy if any
        out_fail = nb_path.with_name(f"executed_{nb_path.name}")
        try:
            nbformat.write(nb, out_fail)
        except Exception:
            pass
        return False, "exec_error", out_fail
    # Save executed copy next to the original
    out = nb_path.with_name(f"executed_{nb_path.name}")
    nbformat.write(nb, out)
    # Scan cell outputs for the NOTEBOOK_RUN_PROOF line
    proof_line = None
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        for outp in cell.get("outputs", []) or []:
            texts = []
            if isinstance(outp, dict):
                if outp.get("text"):
                    texts.append(outp["text"])
                data = outp.get("data", {}) or {}
                for v in data.values():
                    if isinstance(v, str):
                        texts.append(v)
            for t in texts:
                for line in str(t).splitlines():
                    s = line.strip()
                    if s.startswith(PROOF_PREFIX):
                        proof_line = s
                        break
                if proof_line:
                    break
        if proof_line:
            break
    if not proof_line:
        logging.error("proof line not found in executed notebook")
        return False, "missing_proof", out
    logging.info("NOTEBOOK: %s", nb_path)
    logging.info("EXECUTED: %s", out)
    logging.info("PROOF: %s", proof_line)
    print(proof_line)
    return True, proof_line, out


def setup_logging(log_file: str) -> None:
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


# ----------------- Pulls -----------------

SECTORS_CORE = ["S11", "S12", "S13", "S14_S15", "S2"]
SECTORS_TOTAL = ["S1", "S11", "S12", "S13", "S14_S15", "S2"]
NF_TR_META: dict = {"na_items_final": [], "sectors_direct": ["S11","S12","S13","S14_S15"], "check_items": [], "check_item_windows": {}}


def _prefer_unit(units: List[str]) -> str:
    for u in ("MIO_EUR", "CP_MEUR"):
        if u in units:
            return u
    return units[0] if units else "MIO_EUR"


def pull_nf_tr_pl(cfg: dict) -> pd.DataFrame:
    ds = "NASQ_10_NF_TR"
    info = get_dsd("ESTAT", ds)
    # Discover items that have both RECV and PAID at quarterly PL level
    try:
        restr = cfg.get("nf_tr", {}).get("restrict_na_items") or []
        if restr:
            both_items = list(restr)
        else:
            rk = Request('ESTAT').series_keys(ds)
            seen: dict[str, set[str]] = {}
            for sk in rk:
                s = str(sk)
                if "geo=PL" not in s or "freq=Q" not in s:
                    continue
                parts = {p.split('=')[0].strip(): p.split('=')[1].strip() for p in s.strip('<>').split(':',1)[1].split(',') if '=' in p}
                # Only count NSA to avoid s_adj mismatches
                if parts.get('s_adj') and parts['s_adj'] != 'NSA':
                    continue
                na = parts.get('na_item'); direct = parts.get('direct')
                if na and direct:
                    seen.setdefault(na, set()).add(direct)
            both_items = sorted([k for k,v in seen.items() if {'RECV','PAID'}.issubset(v)])
    except Exception:
        both_items = ["D1","D2","D3","D4","D5","D6","D7","D8","D9"]
    NF_TR_META["na_items_final"] = list(both_items)
    NF_TR_META["sectors_direct"] = ["S11","S12","S13","S14_S15"]

    def _pull_direct(unit_code: str) -> Optional[pd.DataFrame]:
        if not both_items:
            return None
        frames: List[pd.DataFrame] = []
        for itm in both_items:
            filters = {
                "freq": "Q",
                "unit": unit_code,
                "direct": ["RECV", "PAID"],
                "sector": NF_TR_META["sectors_direct"],
                "na_item": [itm],
                "s_adj": "NSA",
                "geo": "PL",
            }
            try:
                mc = int(cfg.get("nf_tr", {}).get("max_combinations", 20))
                dfp = fetch_series(ds, filters, agency="ESTAT", max_combinations=mc)
                if not dfp.empty:
                    frames.append(dfp)
            except Exception:
                continue
        if frames:
            return pd.concat(frames, ignore_index=True)
        return None

    def _pull_b9(unit_code: str) -> Optional[pd.DataFrame]:
        filters = {
            "freq": "Q",
            "unit": unit_code,
            "sector": SECTORS_CORE + ["S1"],
            "na_item": ["B9"],
            "s_adj": "NSA",
            "geo": "PL",
        }
        try:
            mc = int(cfg.get("nf_tr", {}).get("max_combinations", 20))
            return fetch_series(ds, filters, agency="ESTAT", max_combinations=mc)
        except Exception:
            return None

    dfs: List[pd.DataFrame] = []
    for unit_code in ("CP_MEUR","MIO_EUR"):
        d1 = _pull_direct(unit_code)
        d2 = _pull_b9(unit_code)
        if d1 is not None and not d1.empty:
            dfs.append(d1)
        if d2 is not None and not d2.empty:
            dfs.append(d2)
        if dfs:
            break
    if not dfs:
        raise RuntimeError("Failed to fetch NASQ_10_NF_TR for PL")
    out = pd.concat(dfs, ignore_index=True)
    # Derive check_items and coverage windows from pulled data (presence of both RECV & PAID within a quarter)
    try:
        d = out[out["sector"].isin(NF_TR_META["sectors_direct"])].copy()
        piv = d.pivot_table(index=["time", "na_item"], columns="direct", values="value", aggfunc="sum").fillna(0.0)
        cols = [c for c in ["RECV","PAID"] if c in piv.columns]
        if len(cols) == 2:
            mask = (piv[cols[0]] != 0) & (piv[cols[1]] != 0)
            idx = piv[mask].reset_index()[["time","na_item"]]
            NF_TR_META["check_items"] = sorted(idx["na_item"].unique().tolist())
            win = {}
            for itm, sub in idx.groupby("na_item"):
                win[str(itm)] = {"first_quarter": str(sub["time"].min()), "last_quarter": str(sub["time"].max())}
            NF_TR_META["check_item_windows"] = win
    except Exception:
        pass
    return out


def pull_f_tr_pl() -> pd.DataFrame:
    ds = "NASQ_10_F_TR"
    info = get_dsd("ESTAT", ds)
    unit = _prefer_unit(info.codes.get("unit", []))
    na_codes = set(info.codes.get("na_item", []))
    f_items = [f"F{i}" for i in range(1, 9) if f"F{i}" in na_codes]
    for extra in ("B9F", "B9FX9"):
        if extra in na_codes:
            f_items.append(extra)
    if not f_items:
        f_items = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "B9F"]
    finpos = [c for c in ["ASS", "LIAB", "NET"] if c in info.codes.get("finpos", [])]
    if not finpos:
        finpos = ["ASS", "LIAB"]
    filters = {
        "freq": "Q",
        "unit": unit,
        "sector": SECTORS_TOTAL,
        "finpos": finpos,
        "na_item": f_items,
        "geo": "PL",
    }
    df = fetch_series(ds, filters, agency="ESTAT")
    return df


def pull_f_bs_pl() -> pd.DataFrame:
    ds = "NASQ_10_F_BS"
    info = get_dsd("ESTAT", ds)
    unit = _prefer_unit(info.codes.get("unit", []))
    # Instruments: try F* (works for Eurostat F_BS) limited to common set
    na_items = [c for c in info.codes.get("na_item", []) if c.startswith("F")][:40]
    if not na_items:
        na_items = ["F2", "F3", "F4", "F5", "F6", "F7", "F8"]
    finpos = [c for c in ["ASS", "LIAB", "NET"] if c in info.codes.get("finpos", [])] or ["ASS", "LIAB"]
    filters = {
        "freq": "Q",
        "unit": unit,
        "sector": SECTORS_TOTAL,
        "finpos": finpos,
        "na_item": na_items,
        "geo": "PL",
    }
    df = fetch_series(ds, filters, agency="ESTAT")
    return df


def pull_f_gl_pl() -> Optional[pd.DataFrame]:
    ds = "NASQ_10_F_GL"
    try:
        info = get_dsd("ESTAT", ds)
    except Exception:
        return None
    unit = _prefer_unit(info.codes.get("unit", []))
    items = [c for c in info.codes.get("na_item", []) if c.startswith("K7") or c == "K.7"]
    if not items:
        items = ["K7"]
    filters = {
        "freq": "Q",
        "unit": unit,
        "sector": SECTORS_TOTAL,
        "na_item": items,
        "geo": "PL",
    }
    try:
        return fetch_series(ds, filters, agency="ESTAT")
    except Exception:
        return None


def pull_f_oc_pl() -> Optional[pd.DataFrame]:
    ds = "NASQ_10_F_OC"
    try:
        info = get_dsd("ESTAT", ds)
    except Exception:
        return None
    unit = _prefer_unit(info.codes.get("unit", []))
    items = [c for c in info.codes.get("na_item", []) if c.startswith("K")]
    if not items:
        items = ["K1", "K2", "K3", "K4", "K5", "K6"]
    filters = {
        "freq": "Q",
        "unit": unit,
        "sector": SECTORS_TOTAL,
        "na_item": items,
        "geo": "PL",
    }
    try:
        return fetch_series(ds, filters, agency="ESTAT")
    except Exception:
        return None


def pull_b9f_pl() -> pd.DataFrame:
    ds = "NASQ_10_F_TR"
    info = get_dsd("ESTAT", ds)
    # Determine available B9F-like codes
    na_avail = set(info.codes.get("na_item", []))
    cand = [c for c in ["B9F", "B9FX9", "B9X9F"] if c in na_avail] or ["B9F"]
    units_try = ["MIO_EUR", "CP_MEUR"]
    sectors = ["S1", "S11", "S12", "S13", "S14_S15"]
    frames: List[pd.DataFrame] = []
    last_err = None
    for unit in units_try:
        for sec in sectors:
            for trial in ("net", "omit"):
                filters = {
                    "freq": "Q",
                    "unit": unit,
                    "sector": [sec],
                    "na_item": cand,
                    "geo": "PL",
                }
                if trial == "net":
                    filters["finpos"] = ["NET"]
                try:
                    df = fetch_series(ds, filters, agency="ESTAT")
                    if not df.empty:
                        frames.append(df)
                        break
                except Exception as e:
                    last_err = e
            # continue next sector
    if frames:
        return pd.concat(frames, ignore_index=True)
    if last_err:
        raise last_err
    return pd.DataFrame()


def pull_qsa_loans_pl() -> Optional[pd.DataFrame]:
    try:
        info = get_dsd("ECB", "QSA")
    except Exception:
        return None
    dims = info.dimensions
    codes = info.codes
    # Identify dims
    def has(dim: str, vals: List[str]) -> bool:
        return dim in codes and all(v in codes[dim] for v in vals)
    instr_dim = next((d for d in dims if has(d, ["F4"])), None)
    # pick two sector dims
    sec_dims = [d for d in dims if has(d, ["S11", "S12", "S13", "S14_S15"])][:2]
    entry_dim = next((d for d in dims if has(d, ["A"]) or has(d, ["L"]) or has(d, ["ASS"])) , None)
    flow_dim = next((d for d in dims if has(d, ["F"]) or has(d, ["S"])) , None)
    unit_dim = next((d for d in dims if has(d, ["MIO_EUR"])) , None)
    geo_dim = next((d for d in dims if has(d, ["PL"])) , None)
    if not instr_dim or len(sec_dims) < 2:
        return None
    filters: Dict[str, object] = {"freq": "Q"}
    filters[instr_dim] = "F4"
    if unit_dim:
        filters[unit_dim] = "MIO_EUR"
    if flow_dim:
        filters[flow_dim] = ["S", "F"]
    if entry_dim:
        filters[entry_dim] = ["A", "L"]
    if geo_dim:
        filters[geo_dim] = "PL"
    filters[sec_dims[0]] = ["S11", "S12", "S13", "S14_S15"]
    filters[sec_dims[1]] = ["S11", "S12", "S13", "S14_S15"]
    try:
        df = fetch_series("QSA", filters, agency="ECB")
    except Exception:
        return None
    # normalize column names
    ren = {instr_dim: "instrument", sec_dims[0]: "ref_sector", sec_dims[1]: "cp_sector"}
    if entry_dim:
        ren[entry_dim] = "entry"
    if flow_dim:
        ren[flow_dim] = "stock_flow_flag"
    df = df.rename(columns=ren)
    return df


def pull(cfg: dict) -> None:
    processed = Path(cfg["io"]["processed_dir"]) 
    processed.mkdir(parents=True, exist_ok=True)

    logging.info("Pulling Eurostat + ECB datasets for PL (Q)")

    df_nf = pull_nf_tr_pl(cfg)
    df_nf.to_parquet(processed / "estat_nasq_10_nf_tr_PL.parquet")
    # Persist NF_TR meta used for traceability
    try:
        (processed / "nf_tr_meta.json").write_text(json.dumps(NF_TR_META, ensure_ascii=False, indent=2))
    except Exception:
        logging.warning("failed to write nf_tr_meta.json")

    df_ftr = pull_f_tr_pl()
    df_ftr.to_parquet(processed / "estat_nasq_10_f_tr_PL.parquet")

    df_fbs = pull_f_bs_pl()
    df_fbs.to_parquet(processed / "estat_nasq_10_f_bs_PL.parquet")

    df_gl = pull_f_gl_pl()
    if df_gl is not None and not df_gl.empty:
        df_gl.to_parquet(processed / "estat_nasq_10_f_gl_PL.parquet")

    df_oc = pull_f_oc_pl()
    if df_oc is not None and not df_oc.empty:
        df_oc.to_parquet(processed / "estat_nasq_10_f_oc_PL.parquet")

    df_b9f = pull_b9f_pl()
    if not df_b9f.empty:
        # merge into f_tr parquet as it’s same dataset, or save separate view
        df_b9f.to_parquet(processed / "estat_nasq_10_f_tr_b9f_PL.parquet")

    df_qsa = pull_qsa_loans_pl()
    if df_qsa is not None and not df_qsa.empty:
        df_qsa.to_parquet(processed / "ecb_QSA_PL.parquet")


# ----------------- Verification -----------------

def _coverage(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    if df is None or df.empty:
        return None, None
    tmin = str(df["time"].min()) if "time" in df.columns else None
    tmax = str(df["time"].max()) if "time" in df.columns else None
    return tmin, tmax


def _resources_uses(df_nf: pd.DataFrame) -> Dict[str, object]:
    # Filter to core sectors (exclude S1)
    d = df_nf[df_nf["sector"].isin(SECTORS_CORE)].copy()
    # Items present on both sides within same quarter
    piv = d.pivot_table(index=["time", "na_item"], columns="direct", values="value", aggfunc="sum")
    piv = piv.fillna(0.0)
    # Keep only quarters/items where both sides were observed (non-zero presence). We use sign-insensitive presence.
    mask_both = (piv.columns.isin(["RECV", "PAID"]))
    cols = [c for c in ["RECV", "PAID"] if c in piv.columns]
    if len(cols) < 2:
        return {"max_abs_resid": None, "breaches": []}
    both = piv[(piv[cols[0]] != 0) & (piv[cols[1]] != 0)]
    both["resid"] = (both.get("RECV", 0.0) - both.get("PAID", 0.0))
    both["abs_resid"] = both["resid"].abs()
    max_abs = float(both["abs_resid"].max()) if not both.empty else 0.0
    breaches = both[both["abs_resid"] > 0.5]
    out_list = [{"time": str(i[0]), "na_item": i[1], "residual": float(v)} for i, v in breaches["resid"].items()]
    return {"max_abs_resid": max_abs, "breaches": out_list, "n_checks": int(len(both))}


def _totals_check(df_nf: pd.DataFrame) -> Dict[Tuple[str, str], float]:
    # For each (na_item, direct): value(S1) vs sum(S11,S12,S13,S14_S15). Exclude S2.
    res: Dict[Tuple[str, str], float] = {}
    if df_nf.empty:
        return res
    base = df_nf[df_nf["sector"].isin(["S1", "S11", "S12", "S13", "S14_S15"])]
    for (item, direct), sub in base.groupby(["na_item", "direct" ]):
        tot = sub[sub["sector"] == "S1"].set_index("time")["value"]
        parts = sub[sub["sector"].isin(["S11", "S12", "S13", "S14_S15"])].groupby("time")["value"].sum()
        align = pd.concat([tot, parts], axis=1, keys=["S1", "sum_parts"]).dropna()
        if align.empty:
            continue
        resid = float((align["S1"] - align["sum_parts"]).abs().max())
        res[(item, direct)] = resid
    return res


def _net_lending(df_nf: pd.DataFrame, df_b9f: pd.DataFrame) -> Dict[str, object]:
    out = {"max_abs_diff": None, "top_offenders": []}
    if df_nf.empty or df_b9f.empty:
        return out
    b9_nf = df_nf[df_nf["na_item"] == "B9"].groupby(["time", "sector"], as_index=False)["value"].sum()
    fx = df_b9f[df_b9f["na_item"].isin(["B9F", "B9FX9"])].groupby(["time", "sector"], as_index=False)["value"].sum().rename(columns={"value": "B9F_sum"})
    comp = b9_nf.merge(fx, on=["time", "sector"], how="inner")
    if comp.empty:
        return out
    comp["diff"] = (comp["value"] - comp["B9F_sum"]).abs()
    out["max_abs_diff"] = float(comp["diff"].max())
    offenders = comp.sort_values("diff", ascending=False).head(10)
    out["top_offenders"] = [{"time": str(r.time), "sector": r.sector, "abs_diff": float(r.diff)} for r in offenders.itertuples()]
    out["n_quarters"] = int(len(comp))
    return out


def _stocks_bridge(df_fbs: pd.DataFrame, df_ftr: pd.DataFrame, df_gl: Optional[pd.DataFrame], df_oc: Optional[pd.DataFrame]) -> Dict[str, object]:
    out = {"max": None, "p95": None, "median": None, "worst10": []}
    if df_fbs.empty or df_ftr.empty:
        return out
    # Compute delta stocks per (sector, finpos, na_item)
    fbs = df_fbs.sort_values(["sector", "finpos", "na_item", "time"]).set_index(["sector", "finpos", "na_item", "time"]).copy()
    delta = fbs.groupby(level=[0, 1, 2])["value"].diff().rename("delta_AF").reset_index()
    flow = df_ftr.rename(columns={"value": "F"})[["time", "sector", "finpos", "na_item", "F"]]
    comp = delta.merge(flow, on=["time", "sector", "finpos", "na_item"], how="left")
    comp["K"] = 0.0
    if df_gl is not None and not df_gl.empty:
        k7 = df_gl.groupby(["time", "sector"], as_index=False)["value"].sum().rename(columns={"value": "K7"})
        comp = comp.merge(k7, on=["time", "sector"], how="left")
        comp["K"] = comp["K"].fillna(0.0) + comp["K7"].fillna(0.0)
    if df_oc is not None and not df_oc.empty:
        k16 = df_oc.groupby(["time", "sector"], as_index=False)["value"].sum().rename(columns={"value": "K16"})
        comp = comp.merge(k16, on=["time", "sector"], how="left")
        comp["K"] = comp["K"].fillna(0.0) + comp["K16"].fillna(0.0)
    comp["F"] = comp["F"].fillna(0.0)
    comp["delta_AF"] = comp["delta_AF"].fillna(0.0)
    comp["resid"] = comp["delta_AF"] - (comp["F"] + comp["K"]) 
    comp["abs_resid"] = comp["resid"].abs()
    if comp.empty:
        return out
    out["max"] = float(comp["abs_resid"].max())
    out["median"] = float(comp["abs_resid"].median())
    out["p95"] = float(comp["abs_resid"].quantile(0.95))
    w10 = comp.sort_values("abs_resid", ascending=False).head(10)
    out["worst10"] = [
        {"time": str(r.time), "sector": r.sector, "finpos": r.finpos, "na_item": r.na_item, "abs_resid": float(r.abs_resid)}
        for r in w10.itertuples()
    ]
    return out


def _qsa_symmetry(df_qsa: Optional[pd.DataFrame]) -> Dict[str, object]:
    out = {"residual": None, "time": None, "dims": None}
    if df_qsa is None or df_qsa.empty:
        return out
    loans = df_qsa[df_qsa.get("instrument").eq("F4")] if "instrument" in df_qsa.columns else df_qsa
    a = loans[(loans.get("entry") == "A") & (loans.get("ref_sector") == "S12") & (loans.get("cp_sector") == "S14_S15")]
    l = loans[(loans.get("entry") == "L") & (loans.get("ref_sector") == "S14_S15") & (loans.get("cp_sector") == "S12")]
    if a.empty or l.empty:
        return out
    idx = a.merge(l, on=["time"], how="inner", suffixes=("_a", "_l"))
    if idx.empty:
        return out
    t = idx["time"].max()
    row = idx[idx["time"] == t].tail(1)
    resid = float((row["value_a"].values[0] - row["value_l"].values[0]))
    out.update({
        "residual": resid,
        "time": str(t),
        "dims": {"ref_sector": "S12", "cp_sector": "S14_S15", "instrument": "F4", "entry": ["A","L"]}
    })
    return out


def verify(cfg: dict) -> None:
    processed = Path(cfg["io"]["processed_dir"]) 
    log_path = Path(cfg["io"]["log_file"]) 

    # Helper: consistent sample hashing for top N rows (deterministic)
    def _sample_hash(df: Optional[pd.DataFrame], n: int = 100) -> Optional[str]:
        if df is None or getattr(df, 'empty', True):
            return None
        sdf = df.copy()
        # Preferred stable sort order
        pref = [
            "time",
            "sector",
            "na_item",
            "direct",
            "finpos",
            "instrument",
            "ref_sector",
            "cp_sector",
            "unit",
            "value",
        ]
        sort_cols = [c for c in pref if c in sdf.columns]
        if not sort_cols:
            sort_cols = sorted([c for c in sdf.columns])
        try:
            sdf = sdf.sort_values(sort_cols, kind="mergesort").head(n)
        except Exception:
            sdf = sdf.head(n)
        # Normalize dtypes
        for c in list(sdf.columns):
            if str(sdf[c].dtype) in ("float64", "float32"):
                sdf[c] = sdf[c].round(6)
        cols = sorted(sdf.columns)
        payload = sdf[cols].to_csv(index=False, lineterminator='\n').encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    # Helper: file hash
    def _file_hash(p: Path) -> Optional[str]:
        if not p.exists():
            return None
        h = hashlib.sha256()
        with p.open('rb') as fh:
            for chunk in iter(lambda: fh.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()

    # Helper: load if exists
    def _maybe(fp: str) -> Optional[pd.DataFrame]:
        p = processed / fp
        if p.exists():
            return pd.read_parquet(p)
        return None

    # Build a fresh snapshot from disk
    df_nf = _maybe("estat_nasq_10_nf_tr_PL.parquet")
    df_ftr = _maybe("estat_nasq_10_f_tr_PL.parquet")
    df_fbs = _maybe("estat_nasq_10_f_bs_PL.parquet")
    df_gl = _maybe("estat_nasq_10_f_gl_PL.parquet")
    df_oc = _maybe("estat_nasq_10_f_oc_PL.parquet")
    df_b9f = _maybe("estat_nasq_10_f_tr_b9f_PL.parquet")
    if df_b9f is None:
        df_b9f = pd.DataFrame()
    df_qsa = _maybe("ecb_QSA_PL.parquet")

    # Coverage
    cov = {
        "NF_TR": {"rows": 0 if df_nf is None else len(df_nf), "first": _coverage(df_nf)[0] if df_nf is not None else None, "last": _coverage(df_nf)[1] if df_nf is not None else None},
        "F_TR": {"rows": 0 if df_ftr is None else len(df_ftr), "first": _coverage(df_ftr)[0] if df_ftr is not None else None, "last": _coverage(df_ftr)[1] if df_ftr is not None else None},
        "F_TR_B9F": {"rows": len(df_b9f) if not df_b9f.empty else 0, "first": _coverage(df_b9f)[0] if not df_b9f.empty else None, "last": _coverage(df_b9f)[1] if not df_b9f.empty else None},
        "F_BS": {"rows": 0 if df_fbs is None else len(df_fbs), "first": _coverage(df_fbs)[0] if df_fbs is not None else None, "last": _coverage(df_fbs)[1] if df_fbs is not None else None},
        "F_GL": {"rows": 0 if df_gl is None else len(df_gl), "first": _coverage(df_gl)[0] if df_gl is not None else None, "last": _coverage(df_gl)[1] if df_gl is not None else None},
        "F_OC": {"rows": 0 if df_oc is None else len(df_oc), "first": _coverage(df_oc)[0] if df_oc is not None else None, "last": _coverage(df_oc)[1] if df_oc is not None else None},
        "QSA": {"rows": 0 if df_qsa is None else len(df_qsa), "first": _coverage(df_qsa)[0] if df_qsa is not None else None, "last": _coverage(df_qsa)[1] if df_qsa is not None else None},
    }

    # A) Resources = Uses (scoped)
    res_uses = _resources_uses(df_nf if df_nf is not None else pd.DataFrame())

    # B) Net lending
    netlend = _net_lending(df_nf if df_nf is not None else pd.DataFrame(), df_b9f)

    # C) Stocks bridge
    # F_BS schema sanity: prefer AF.* instruments for stocks; warn if mixed
    df_fbs_for_bridge = df_fbs if df_fbs is not None else pd.DataFrame()
    warn_fbs_mixed = False
    if df_fbs_for_bridge is not None and not df_fbs_for_bridge.empty and "na_item" in df_fbs_for_bridge.columns:
        has_af = df_fbs_for_bridge["na_item"].astype(str).str.startswith("AF").any()
        has_f = df_fbs_for_bridge["na_item"].astype(str).str.match(r"^F(?!X)").any()
        if has_af and has_f:
            warn_fbs_mixed = True
            df_fbs_for_bridge = df_fbs_for_bridge[df_fbs_for_bridge["na_item"].astype(str).str.startswith("AF")] 
    stocks = _stocks_bridge(df_fbs_for_bridge, df_ftr if df_ftr is not None else pd.DataFrame(), df_gl, df_oc)

    # D) QSA symmetry
    qsa_sym = _qsa_symmetry(df_qsa)

    # Totals check
    totals = _totals_check(df_nf if df_nf is not None else pd.DataFrame())

    # Anchor rows
    anchor_nf = None
    anchor_b9f = None
    if df_nf is not None and not df_nf.empty:
        a = df_nf[(df_nf["na_item"] == "B9") & (df_nf["sector"] == "S13")]
        if not a.empty:
            t = a["time"].max()
            anchor_nf = a[a["time"] == t].tail(1)
    if not df_b9f.empty:
        b = df_b9f[(df_b9f["na_item"].isin(["B9F", "B9FX9"])) & (df_b9f["sector"] == "S13")]
        if not b.empty:
            t = b["time"].max()
            anchor_b9f = b[b["time"] == t].tail(1)

    # Save JSON reports
    processed = Path(cfg["io"]["processed_dir"]) 
    # Build snapshot metrics per dataset
    def _vc(series, top=5):
        try:
            return series.astype(str).value_counts().head(top).to_dict()
        except Exception:
            return {}

    def _metrics(df: Optional[pd.DataFrame]) -> Dict[str, object]:
        if df is None or getattr(df, 'empty', True):
            return {"rows": 0, "cols": 0, "first_quarter": None, "last_quarter": None, "unit_counts": {}, "sector_counts": {}}
        first, last = _coverage(df)
        return {
            "rows": int(len(df)),
            "cols": int(len(df.columns)),
            "first_quarter": first,
            "last_quarter": last,
            "unit_counts": _vc(df["unit"]) if "unit" in df.columns else {},
            "sector_counts": _vc(df["sector"]) if "sector" in df.columns else {},
        }

    artifacts = {
        "data/processed/estat_nasq_10_nf_tr_PL.parquet": _metrics(df_nf),
        "data/processed/estat_nasq_10_f_tr_PL.parquet": _metrics(df_ftr),
        "data/processed/estat_nasq_10_f_bs_PL.parquet": _metrics(df_fbs),
        "data/processed/estat_nasq_10_f_gl_PL.parquet": _metrics(df_gl),
        "data/processed/estat_nasq_10_f_oc_PL.parquet": _metrics(df_oc),
        "data/processed/ecb_QSA_PL.parquet": _metrics(df_qsa),
    }
    # add exists and size_bytes for each artifact
    for path in list(artifacts.keys()):
        p = Path(path)
        meta = artifacts[path]
        meta["exists"] = p.exists()
        meta["size_bytes"] = (p.stat().st_size if p.exists() else 0)

    # NF_TR scope from data + optional meta file
    nf_meta_fp = processed / "nf_tr_meta.json"
    nf_meta = {}
    try:
        if nf_meta_fp.exists():
            nf_meta = json.loads(nf_meta_fp.read_text())
    except Exception:
        nf_meta = {}
    na_items_present = sorted(df_nf["na_item"].astype(str).unique().tolist()) if df_nf is not None and not df_nf.empty and "na_item" in df_nf.columns else []
    s_adj_counts = _vc(df_nf["s_adj"]) if df_nf is not None and not df_nf.empty and "s_adj" in df_nf.columns else {}
    # Derive check_items and windows if missing/empty in nf_meta
    if (not nf_meta.get("check_items")) and df_nf is not None and not df_nf.empty and set(["time","na_item","direct","sector"]).issubset(df_nf.columns):
        try:
            core = nf_meta.get("sectors_direct") or ["S11","S12","S13","S14_S15"]
            d = df_nf[df_nf["sector"].isin(core)].copy()
            piv = d.pivot_table(index=["time","na_item"], columns="direct", values="value", aggfunc="sum").fillna(0.0)
            cols = [c for c in ["RECV","PAID"] if c in piv.columns]
            items = []
            win = {}
            if len(cols) == 2:
                mask = (piv[cols[0]] != 0) & (piv[cols[1]] != 0)
                idx = piv[mask].reset_index()[["time","na_item"]]
                items = sorted(idx["na_item"].astype(str).unique().tolist())
                for itm, sub in idx.groupby("na_item"):
                    win[str(itm)] = {"first_quarter": str(sub["time"].min()), "last_quarter": str(sub["time"].max())}
            nf_meta.update({"sectors_direct": core, "na_items_final": items, "check_items": items, "check_item_windows": win})
        except Exception:
            pass

    # Schema sanity flags
    schema = {
        "nf_tr_has_cols": all(c in (df_nf.columns if df_nf is not None else []) for c in ["time","sector","na_item","direct","unit","value","dataset","last_update"]),
        "nf_tr_has_TIME_PERIOD": ("TIME_PERIOD" in (df_nf.columns if df_nf is not None else [])),
        "f_tr_has_cols": all(c in (df_ftr.columns if df_ftr is not None else []) for c in ["time","sector","na_item","finpos","unit","value"]),
        "f_bs_has_AF_only": False,
        "f_bs_mixed_warn": False,
    }
    if df_fbs is not None and not df_fbs.empty and "na_item" in df_fbs.columns:
        has_af = df_fbs["na_item"].astype(str).str.startswith("AF").any()
        has_f = df_fbs["na_item"].astype(str).str.match(r"^F(?!X)").any()
        schema["f_bs_mixed_warn"] = bool(has_af and has_f)
        schema["f_bs_has_AF_only"] = bool(has_af and not has_f)

    # QC summary (same as before)
    qc_summary = {
        "coverage": cov,
        "resources_uses": res_uses,
        "net_lending": netlend,
        "stocks_bridge": stocks,
        "totals_check": {f"{k[0]}|{k[1]}": v for k, v in totals.items()},
        "qsa_symmetry": qsa_sym,
    }

    # Failure summary from in-memory events or log scraping
    events = get_fetch_log()
    fail_lines = []
    if not events and log_path.exists():
        try:
            for ln in log_path.read_text(encoding='utf-8', errors='ignore').splitlines():
                if ln.strip().startswith("FAIL | dataset="):
                    fail_lines.append(ln.strip())
        except Exception:
            pass
    by_ds = {}
    for ev in events:
        ds = ev.get("dataset")
        d = by_ds.setdefault(ds, {"attempted":0,"ok":0,"failed":0})
        d["attempted"] += 1
        if ev.get("ok"):
            d["ok"] += 1
        else:
            d["failed"] += 1

    # Build checksums for parquets
    checksums = {}
    for path, df in artifacts.items():
        p = Path(path)
        checksums[path] = {
            "file": _file_hash(p),
            "sample": _sample_hash(_maybe(Path(path).name)),
        }

    snapshot = {
        "artifacts": artifacts,
        "schema": schema,
        "nf_tr_scope": {
            "sectors_direct": nf_meta.get("sectors_direct", ["S11","S12","S13","S14_S15"]),
            "na_items_used": nf_meta.get("na_items_final", na_items_present),
            "check_items": nf_meta.get("check_items", []),
            "check_item_windows": nf_meta.get("check_item_windows", {}),
            "s_adj_counts": s_adj_counts,
            "unit_counts": artifacts.get("data/processed/estat_nasq_10_nf_tr_PL.parquet", {}).get("unit_counts", {}),
        },
        "qc_summary": qc_summary,
        "anchors": {},
        "failures": {"by_dataset": by_ds, "lines": fail_lines},
        "checksums": checksums,
        "run_ts": datetime.utcnow().isoformat(),
    }

    # Anchors and non-null guard
    anchor_nf = None
    anchor_b9f = None
    if df_nf is not None and not df_nf.empty:
        a = df_nf[(df_nf.get("na_item").astype(str) == "B9") & (df_nf.get("sector").astype(str) == "S13")]
        if not a.empty:
            t = a["time"].max()
            anchor_nf = a[a["time"] == t].tail(1)
    if df_b9f is not None and not df_b9f.empty:
        b = df_b9f[(df_b9f.get("na_item").astype(str).isin(["B9F","B9FX9"])) & (df_b9f.get("sector").astype(str) == "S13")]
        if not b.empty:
            t = b["time"].max()
            anchor_b9f = b[b["time"] == t].tail(1)

    def _row_to_dict(df: Optional[pd.DataFrame], cols: List[str]) -> Optional[Dict[str, object]]:
        if df is None or df.empty:
            return None
        r = df.iloc[0]
        out = {}
        for c in cols:
            if c in df.columns:
                v = r[c]
                try:
                    # Normalize non-JSON types
                    if isinstance(v, (pd.Period, )):
                        v = str(v)
                except Exception:
                    pass
                out[c] = v
        return out

    snapshot["anchors"]["nf_tr_b9_s13"] = _row_to_dict(anchor_nf, ["geo","s_adj","unit","sector","na_item","direct","time","value"])
    snapshot["anchors"]["f_tr_b9f_s13"] = _row_to_dict(anchor_b9f, ["geo","unit","sector","na_item","finpos","time","value"])

    # Assertions: anchors present and non-null
    assertions_failed = []
    if not snapshot["anchors"]["nf_tr_b9_s13"] or pd.isna(snapshot["anchors"]["nf_tr_b9_s13"].get("value")):
        assertions_failed.append("FAIL (anchor missing): NF_TR B9 S13 latest")
    if not snapshot["anchors"]["f_tr_b9f_s13"] or pd.isna(snapshot["anchors"]["f_tr_b9f_s13"].get("value")):
        assertions_failed.append("FAIL (anchor missing): F_TR B9F S13 latest")

    # Units & s_adj sanity assertions
    nf_units = set(artifacts.get("data/processed/estat_nasq_10_nf_tr_PL.parquet", {}).get("unit_counts", {}).keys())
    if nf_units and "CP_MEUR" not in nf_units:
        assertions_failed.append("FAIL (units): NF_TR missing CP_MEUR")
    ftr_units = set(artifacts.get("data/processed/estat_nasq_10_f_tr_PL.parquet", {}).get("unit_counts", {}).keys())
    if ftr_units and "MIO_EUR" not in ftr_units:
        assertions_failed.append("FAIL (units): F_TR missing MIO_EUR")
    fbs_units = set(artifacts.get("data/processed/estat_nasq_10_f_bs_PL.parquet", {}).get("unit_counts", {}).keys())
    if fbs_units and "MIO_EUR" not in fbs_units:
        assertions_failed.append("FAIL (units): F_BS missing MIO_EUR")
    if s_adj_counts and "NSA" not in s_adj_counts:
        assertions_failed.append("FAIL (s_adj): NF_TR missing NSA")

    # Write report JSON only from this snapshot
    vr_fp = processed / "verification_report.json"
    qc_fp = processed / "qc_summary.json"
    vr_rel = "data/processed/verification_report.json"
    qc_rel = "data/processed/qc_summary.json"
    vr_fp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))
    qc_fp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))

    # Now that JSONs exist, include them in artifacts with exists/size (use relative paths)
    artifacts[vr_rel] = {"rows": None, "cols": None, "first_quarter": None, "last_quarter": None,
                         "unit_counts": {}, "sector_counts": {}, "exists": vr_fp.exists(), "size_bytes": (vr_fp.stat().st_size if vr_fp.exists() else 0)}
    artifacts[qc_rel] = {"rows": None, "cols": None, "first_quarter": None, "last_quarter": None,
                         "unit_counts": {}, "sector_counts": {}, "exists": qc_fp.exists(), "size_bytes": (qc_fp.stat().st_size if qc_fp.exists() else 0)}
    snapshot["artifacts"] = artifacts

    # After writing, add checksum for JSON itself and rewrite (relative keys)
    snapshot["checksums"][vr_rel] = {"file": _file_hash(vr_fp), "sample": None}
    snapshot["checksums"][qc_rel] = {"file": _file_hash(qc_fp), "sample": None}
    vr_fp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))
    qc_fp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))

    # Shadow recompute and deep-compare to JSON
    # Recompute minimal snapshot again
    df_nf2 = _maybe("estat_nasq_10_nf_tr_PL.parquet")
    df_ftr2 = _maybe("estat_nasq_10_f_tr_PL.parquet")
    df_fbs2 = _maybe("estat_nasq_10_f_bs_PL.parquet")
    df_gl2 = _maybe("estat_nasq_10_f_gl_PL.parquet")
    df_oc2 = _maybe("estat_nasq_10_f_oc_PL.parquet")
    df_qsa2 = _maybe("ecb_QSA_PL.parquet")
    artifacts2 = {
        "data/processed/estat_nasq_10_nf_tr_PL.parquet": _metrics(df_nf2),
        "data/processed/estat_nasq_10_f_tr_PL.parquet": _metrics(df_ftr2),
        "data/processed/estat_nasq_10_f_bs_PL.parquet": _metrics(df_fbs2),
        "data/processed/estat_nasq_10_f_gl_PL.parquet": _metrics(df_gl2),
        "data/processed/estat_nasq_10_f_oc_PL.parquet": _metrics(df_oc2),
        "data/processed/ecb_QSA_PL.parquet": _metrics(df_qsa2),
        vr_rel: {"rows": None, "cols": None, "first_quarter": None, "last_quarter": None,
                 "unit_counts": {}, "sector_counts": {}, "exists": vr_fp.exists(), "size_bytes": (vr_fp.stat().st_size if vr_fp.exists() else 0)},
        qc_rel: {"rows": None, "cols": None, "first_quarter": None, "last_quarter": None,
                 "unit_counts": {}, "sector_counts": {}, "exists": qc_fp.exists(), "size_bytes": (qc_fp.stat().st_size if qc_fp.exists() else 0)},
    }
    rep = json.loads(vr_fp.read_text())
    if artifacts2 != rep.get("artifacts"):
        logging.warning("printed≠recomputed: artifacts snapshot mismatch (non-fatal)")

    # Now print everything from rep (not from DataFrames)
    print("=== Coverage Recap ===")
    for path, m in rep.get("artifacts", {}).items():
        print(f"{path}: rows={m.get('rows')} first={m.get('first_quarter')} last={m.get('last_quarter')}")

    print("\n=== Artifacts (existence + schema) ===")
    for path, m in rep.get("artifacts", {}).items():
        p = Path(path)
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        print(f"{path} | exists={exists} size={size} rows={m.get('rows')} cols={m.get('cols')} first={m.get('first_quarter')} last={m.get('last_quarter')}")
    for path in (processed / "verification_report.json", processed / "qc_summary.json"):
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        print(f"{path} | exists={exists} size={size}")

    print("\n=== Schema Assertions ===")
    sc = rep.get("schema", {})
    print(f"NF_TR columns: {'OK' if sc.get('nf_tr_has_cols') else 'FAIL'}; TIME_PERIOD present? {sc.get('nf_tr_has_TIME_PERIOD')}")
    print(f"F_TR columns: {'OK' if sc.get('f_tr_has_cols') else 'FAIL'}")
    if sc.get("f_bs_mixed_warn"):
        print("F_BS: WARN mixed AF.* and F.*; filtered to AF.* for bridge")
    else:
        print(f"F_BS: {'OK instruments AF.*' if sc.get('f_bs_has_AF_only') else 'WARN instruments not AF.*'}")

    print("\n=== NF_TR Scope & Check Items ===")
    scope = rep.get("nf_tr_scope", {})
    print("DIRECT sectors:", scope.get("sectors_direct"))
    print("NA_ITEM list used:", scope.get("na_items_used"))
    print("Derived check_items:", scope.get("check_items"))
    for itm, w in (scope.get("check_item_windows") or {}).items():
        print(f"{itm}: first={w.get('first_quarter')} last={w.get('last_quarter')}")

    print("\n=== Resources=Uses (scoped) ===")
    ru = rep.get("qc_summary", {}).get("resources_uses", {})
    print("max_abs_resid:", ru.get("max_abs_resid"), "| n_checks:", ru.get("n_checks"))
    if ru.get("breaches"):
        print("breaches > 0.5 (time, na_item, residual):")
        for b in ru.get("breaches"):
            if abs(b.get("residual", 0.0)) > 0.5:
                print(b)

    print("\n=== Net lending (B9 vs B9F [+B9FX9]) ===")
    nl = rep.get("qc_summary", {}).get("net_lending", {})
    print("max_abs_diff:", nl.get("max_abs_diff"), "| n_quarters:", nl.get("n_quarters"))
    for o in nl.get("top_offenders", []) or []:
        print(o)

    print("\n=== Stocks bridge (ΔAF ≈ F + K.7 + (K.1–K.6)) ===")
    sb = rep.get("qc_summary", {}).get("stocks_bridge", {})
    print({k: sb.get(k) for k in ("max", "p95", "median")})
    for w in (sb.get("worst10") or []):
        print(w)

    print("\n=== QSA symmetry (Loans F4) ===")
    print(rep.get("qc_summary", {}).get("qsa_symmetry"))

    print("\n=== Totals check (S1 vs Σ parts by (na_item,direct)) ===")
    totals = rep.get("qc_summary", {}).get("totals_check", {})
    if totals:
        mk = max(totals, key=lambda k: totals[k])
        print(f"max_residual_pair={mk} value={totals[mk]}")
    for k, v in totals.items():
        print(f"{k}: {v}")

    print("\n=== Anchor rows ===")
    anc = rep.get("anchors", {})
    if anc.get("nf_tr_b9_s13"):
        print("NF_TR B9 S13 latest (full dims):")
        print(anc.get("nf_tr_b9_s13"))
    if anc.get("f_tr_b9f_s13"):
        print("F_TR B9F S13 latest (full dims):")
        print(anc.get("f_tr_b9f_s13"))

    print("\n=== Fetch Failures Summary ===")
    fails = rep.get("failures", {})
    for ds, stats in (fails.get("by_dataset") or {}).items():
        print(f"{ds}: attempted={stats.get('attempted')} ok={stats.get('ok')} failed={stats.get('failed')}")
    for line in fails.get("lines") or []:
        print(line)

    # Print checksums
    print("\n=== Checksums ===")
    for path, h in (rep.get("checksums") or {}).items():
        print(f"{path}: sha256(file)={h.get('file')} sha256(sample)={h.get('sample')}")

    # Persist verify_exit_code into the JSON snapshot (always)
    try:
        vr_fp = processed / "verification_report.json"
        rep2 = {}
        if vr_fp.exists():
            rep2 = json.loads(vr_fp.read_text(encoding='utf-8'))
        rep2["verify_exit_code"] = 0 if not assertions_failed else 2
        vr_fp.write_text(json.dumps(rep2, ensure_ascii=False, indent=2))
    except Exception:
        pass

    # Return non-zero on assertions
    if assertions_failed:
        for msg in assertions_failed:
            print(msg)
        return 2
    return 0

    # Print verification block
    print("=== Coverage Recap ===")
    for k, v in cov.items():
        print(f"{k}: rows={v['rows']} first={v['first']} last={v['last']}")

    print("\n=== Resources=Uses (scoped) ===")
    print("max_abs_resid:", res_uses.get("max_abs_resid"), "| n_checks:", res_uses.get("n_checks"))
    breaches = res_uses.get("breaches", [])
    if breaches:
        print("breaches > 0.5 (time, na_item, residual):")
        for b in breaches:
            if abs(b.get("residual", 0.0)) > 0.5:
                print(b)

    print("\n=== Net lending (B9 vs B9F [+B9FX9]) ===")
    print("max_abs_diff:", netlend.get("max_abs_diff"), "| n_quarters:", netlend.get("n_quarters"))
    for o in netlend.get("top_offenders", []):
        print(o)

    print("\n=== Stocks bridge (ΔAF ≈ F + K.7 + (K.1–K.6)) ===")
    print({k: stocks.get(k) for k in ("max", "p95", "median")})
    if stocks.get("worst10"):
        for w in stocks["worst10"]:
            print(w)

    print("\n=== QSA symmetry (Loans F4) ===")
    print(qsa_sym)

    print("\n=== Totals check (S1 vs Σ parts by (na_item,direct)) ===")
    if totals:
        # print max residual first
        mk = max(totals, key=lambda k: totals[k])
        print(f"max_residual_pair={mk} value={totals[mk]}")
    for k, v in totals.items():
        print(f"{k}: {v}")

    print("\n=== Anchor rows ===")
    if anchor_nf is not None:
        print("NF_TR B9 S13 latest (full dims):")
        cols = [c for c in ["geo","s_adj","unit","sector","na_item","direct","time","value"] if c in anchor_nf.columns]
        print(anchor_nf[cols].to_string(index=False))
    if anchor_b9f is not None:
        print("F_TR B9F S13 latest (full dims):")
        cols = [c for c in ["geo","unit","sector","na_item","finpos","time","value"] if c in anchor_b9f.columns]
        print(anchor_b9f[cols].to_string(index=False))

    # --------- PRINT THIS BLOCK (artifacts, scope, schema, failures) ---------
    print("\n=== Artifacts (existence + schema) ===")
    artifacts = [
        ("data/processed/estat_nasq_10_nf_tr_PL.parquet", df_nf),
        ("data/processed/estat_nasq_10_f_tr_PL.parquet", df_ftr),
        ("data/processed/estat_nasq_10_f_bs_PL.parquet", df_fbs),
        ("data/processed/estat_nasq_10_f_gl_PL.parquet", df_gl),
        ("data/processed/estat_nasq_10_f_oc_PL.parquet", df_oc),
        ("data/processed/ecb_QSA_PL.parquet", df_qsa),
    ]
    import os
    def _first_last(df):
        if df is None or df is pd.DataFrame() or df is None or getattr(df, 'empty', True):
            return None, None
        if "time" in df.columns:
            return str(df["time"].min()), str(df["time"].max())
        return None, None
    for path, df in artifacts:
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        rows = (len(df) if df is not None else 0)
        cols = (len(df.columns) if df is not None else 0)
        f, l = _first_last(df if df is not None else pd.DataFrame())
        print(f"{path} | exists={exists} size={size} rows={rows} cols={cols} first={f} last={l}")
    for path in ("data/processed/qc_summary.json","data/processed/verification_report.json"):
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        print(f"{path} | exists={exists} size={size}")

    # Tidy schema assertions
    print("\n=== Schema Assertions ===")
    def _assert_cols(df, req, name):
        if df is None or df.empty:
            print(f"{name}: SKIP (empty)")
            return
        missing = [c for c in req if c not in df.columns]
        if missing:
            print(f"{name}: FAIL missing {missing}")
        else:
            print(f"{name}: OK")
    _assert_cols(df_nf, ["time","sector","na_item","direct","unit","value","dataset","last_update"], "NF_TR columns")
    if df_nf is not None and not df_nf.empty:
        if "TIME_PERIOD" in df_nf.columns:
            print("NF_TR: FAIL TIME_PERIOD present as column")
        # Time dtype check (best-effort)
        try:
            ok = str(df_nf["time"].dtype).startswith("period[Q]")
            print(f"NF_TR time dtype: {'OK' if ok else 'WARN'} ({df_nf['time'].dtype})")
        except Exception:
            pass
    _assert_cols(df_ftr, ["time","sector","na_item","finpos","unit","value"], "F_TR columns")
    if df_ftr is not None and not df_ftr.empty and "unit" in df_ftr.columns:
        units = sorted(set(df_ftr["unit"].astype(str)))
        print(f"F_TR units: {units}")
        if "MIO_EUR" not in units:
            print("F_TR: WARN expected MIO_EUR in units")
    if df_fbs is not None and not df_fbs.empty:
        has_af = df_fbs["na_item"].astype(str).str.startswith("AF").any() if "na_item" in df_fbs.columns else False
        has_f = df_fbs["na_item"].astype(str).str.match(r"^F(?!X)").any() if "na_item" in df_fbs.columns else False
        if has_af and has_f:
            print("F_BS: WARN mixed AF.* and F.*; filtered to AF.* for bridge")
        elif has_af:
            print("F_BS: OK instruments AF.*")
        else:
            print("F_BS: WARN instruments not AF.*")

    # NF_TR scope & check_items trace
    print("\n=== NF_TR Scope & Check Items ===")
    print("DIRECT sectors:", NF_TR_META.get("sectors_direct"))
    print("NA_ITEM list used:", NF_TR_META.get("na_items_final"))
    print("Derived check_items:", NF_TR_META.get("check_items"))
    win = NF_TR_META.get("check_item_windows", {})
    for itm, w in win.items():
        print(f"{itm}: first={w.get('first_quarter')} last={w.get('last_quarter')}")

    # Failures summary
    print("\n=== Fetch Failures Summary ===")
    events = get_fetch_log()
    by_ds = {}
    for ev in events:
        ds = ev.get("dataset")
        d = by_ds.setdefault(ds, {"attempted":0,"ok":0,"failed":0})
        d["attempted"] += 1
        if ev.get("ok"):
            d["ok"] += 1
        else:
            d["failed"] += 1
    for ds, stats in by_ds.items():
        print(f"{ds}: attempted={stats['attempted']} ok={stats['ok']} failed={stats['failed']}")
    for ev in events:
        if not ev.get("ok"):
            line = f"FAIL | dataset={ev.get('dataset')} key={ev.get('key')} params={ev.get('params')} status={ev.get('status')} url={ev.get('url')}"
            print(line)
            logging.error(line)


def run_from_config(cfg: dict) -> int:
    start = datetime.utcnow().isoformat()
    logging.info("sfc_pl_runner start | utc=%s | cfg=%s", start, {k: cfg.get(k) for k in ("run","io","safety")})
    if cfg["safety"].get("destructive_ops", False):
        logging.error("destructive_ops must remain False; aborting")
        return 2
    try:
        # Only execute the notebook and surface proof
        if not cfg["run"].get("execute_notebook", True):
            logging.error("execute_notebook must be true for this project")
            return 5
        ok, msg, executed = execute_notebook(cfg)
        if not ok:
            logging.error("Notebook execution failed: %s | executed_copy=%s", msg, executed)
            return 2 if msg == "missing_notebook" else 4
        # After proof printed, gate on JSON verify_exit_code
        processed = Path(cfg["io"]["processed_dir"]) 
        vr = processed / "verification_report.json"
        if not vr.exists():
            logging.error("verification_report.json missing after notebook execution: %s", vr)
            return 4
        try:
            rep = json.loads(vr.read_text(encoding="utf-8"))
        except Exception:
            logging.exception("failed to read verification_report.json")
            return 4
        vrc = int(rep.get("verify_exit_code", -1))
        if vrc == 0:
            logging.info("verification passed (verify_exit_code=0)")
            return 0
        else:
            logging.error("verification failed (verify_exit_code=%s)", vrc)
            return 6
    except Exception:
        logging.exception("runner failed")
        return 1
    logging.info("sfc_pl_runner done")
    return 0


if __name__ == "__main__":
    cfg = load_config()
    setup_logging(cfg["io"]["log_file"]) 
    rc = run_from_config(cfg)
    sys.exit(rc)
