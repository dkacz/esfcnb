import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional, Iterable
import logging
from urllib.parse import urlencode

import pandas as pd
from pandasdmx import Request
from requests import HTTPError
import threading


CACHE_DIR = os.path.join("data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# In-memory fetch log (accumulates per run)
_FETCH_EVENTS_LOCK = threading.Lock()
_FETCH_EVENTS: List[Dict[str, Any]] = []


def reset_fetch_log() -> None:
    with _FETCH_EVENTS_LOCK:
        _FETCH_EVENTS.clear()


def get_fetch_log() -> List[Dict[str, Any]]:
    with _FETCH_EVENTS_LOCK:
        return list(_FETCH_EVENTS)


def _push_fetch_event(ev: Dict[str, Any]) -> None:
    with _FETCH_EVENTS_LOCK:
        _FETCH_EVENTS.append(ev)


@dataclass
class DSDInfo:
    dataset: str
    agency: str
    dimensions: List[str]
    codes: Dict[str, List[str]]  # dim_id -> list of codes
    prepared: Optional[str] = None


def _cache_path(agency: str, dataset: str) -> str:
    return os.path.join(CACHE_DIR, f"dsd_{agency}_{dataset}.json")


def get_dsd(agency: str, dataset: str) -> DSDInfo:
    """Fetch DSD and codelists. Returns dimension order and valid codes, with small disk cache."""
    cache_fp = _cache_path(agency, dataset)
    if os.path.exists(cache_fp):
        with open(cache_fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        return DSDInfo(
            dataset=data["dataset"],
            agency=data["agency"],
            dimensions=data["dimensions"],
            codes=data["codes"],
            prepared=data.get("prepared"),
        )

    req = Request(agency)
    msg = req.datastructure(dataset)

    dsd = msg.structure[dataset]
    dims = [d.id for d in dsd.dimensions]

    # Build code lists per dimension if available
    codes: Dict[str, List[str]] = {}
    # Some structural dimensions like TIME_PERIOD might not have codelists
    for dim in dsd.dimensions:
        dim_id = dim.id
        try:
            cl_id = dim.local_representation.enumeration.id
            cl = msg.codelist[cl_id]
            codes[dim_id] = [c.id for c in cl]
        except Exception:
            # Fallback to empty; validation will skip for dims without code lists
            codes[dim_id] = []

    prepared = None
    try:
        prepared = str(msg.header.prepared)
    except Exception:
        pass

    info = DSDInfo(dataset=dataset, agency=agency, dimensions=dims, codes=codes, prepared=prepared)

    with open(cache_fp, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": info.dataset,
            "agency": info.agency,
            "dimensions": info.dimensions,
            "codes": info.codes,
            "prepared": info.prepared,
        }, f, ensure_ascii=False, indent=2)

    return info


def _normalize_filter_value(val: Any) -> str:
    """Normalize filter values to SDMX key segment string.
    - list/tuple -> join with '+'
    - '-' kept as wildcard
    - other -> str
    """
    # Empty slot wildcard: empty string per Eurostat SDMX 2.1 guidance
    if val is None or val == "-" or val == "":
        return ""
    if isinstance(val, (list, tuple, set)):
        parts = []
        for v in val:
            if v is None:
                continue
            s = str(v).strip()
            if s:
                parts.append(s)
        return "+".join(parts) if parts else "-"
    return str(val).strip()


def build_key(dataset: str, filters: Dict[str, Any], agency: str = "ESTAT") -> Tuple[str, List[str]]:
    """Compose the dot-key in DSD order from filters.
    - Supports multi-values via list (joined by '+') and wildcard '-'.
    - Validates codes using DSD codelists when available.
    - Ignores unknown dimensions present in filters.

    Returns (key, used_dims) where used_dims is the ordered list of non-time dimensions.
    """
    info = get_dsd(agency, dataset)
    dims = [d for d in info.dimensions if d != "TIME_PERIOD"]

    # Validation against codelists
    def validate_dim_value(dim: str, value_expr: str) -> str:
        if value_expr == "":
            return value_expr
        allowed = set(info.codes.get(dim, []))
        # If no codelist, accept as-is
        if not allowed:
            return value_expr
        parts = value_expr.split("+")
        bad = [p for p in parts if p not in allowed]
        if bad:
            raise ValueError(f"Invalid code(s) for {dataset}.{dim}: {bad}; allowed example(s): {sorted(list(allowed))[:8]}")
        return value_expr

    used = []
    key_parts = []
    for dim in dims:
        raw = None
        # Support common aliases in filters (case-insensitive)
        for k in (dim, dim.lower(), dim.upper()):
            if k in filters:
                raw = filters[k]
                break
        val = _normalize_filter_value(raw if raw is not None else "")
        val = validate_dim_value(dim, val)
        key_parts.append(val)
        used.append(dim)

    return ".".join(key_parts), used


def _chunk_multi_values(key_expr: str, max_combinations: int = 50) -> List[str]:
    """Split a key expression into smaller chunks if there are too many combinations.
    This splits on the first dimension that has multiple values joined by '+'."""
    segs = key_expr.split(".")

    # Compute combinations
    comb = 1
    for seg in segs:
        if seg == "":
            continue
        comb *= (len(seg.split("+")) if "+" in seg else 1)
    if comb <= max_combinations:
        return [key_expr]

    # Split the widest segment into chunks
    sizes = [0 if seg == "" else len(seg.split("+")) for seg in segs]
    if max(sizes) == 1:
        return [key_expr]
    idx = sizes.index(max(sizes))
    vals = segs[idx].split("+")
    chunk_size = max(1, max_combinations // max(1, (comb // len(vals))))
    chunks = []
    for i in range(0, len(vals), chunk_size):
        segs_i = list(segs)
        segs_i[idx] = "+".join(vals[i:i+chunk_size])
        chunks.append(".".join(segs_i))
    return chunks


def _to_tidy(msg, dataset: str) -> pd.DataFrame:
    """Convert pandasdmx DataMessage to tidy DataFrame with Period[Q] index and value column."""
    ser = msg.to_pandas()
    if not isinstance(ser, (pd.Series, pd.DataFrame)):
        raise ValueError("Unexpected conversion; expected Series or DataFrame")
    if isinstance(ser, pd.DataFrame):
        # Some structures may already produce DF; melt to series
        ser = ser.stack()
    ser = ser.rename("value")
    df = ser.reset_index()
    # Normalize time column
    if "TIME_PERIOD" in df.columns:
        # Coerce to quarterly Period
        df["time"] = pd.PeriodIndex(df["TIME_PERIOD"], freq="Q")
        df = df.drop(columns=["TIME_PERIOD"])
    # Keep freq for completeness but not mandatory; ensure it's 'Q'
    # Attach dataset
    df["dataset"] = dataset
    # Last update timestamp if available
    last_update = None
    try:
        last_update = str(msg.header.prepared)
    except Exception:
        pass
    df["last_update"] = last_update
    return df


def fetch_series(dataset: str, filters: Dict[str, Any], agency: str = "ESTAT", retries: int = 3,
                 backoff_base: float = 1.0, start_period: Optional[str] = None,
                 end_period: Optional[str] = None, max_combinations: int = 50) -> pd.DataFrame:
    """Call SDMX 2.1 via pandasdmx, retry, chunk long queries, return tidy DataFrame.

    Adds:
    - max_combinations: limit for chunk splitting of multi-value keys (default 50).
    - logs request context and a best-effort URL (for ESTAT) on failures for easier debugging.
    """
    key_expr, used_dims = build_key(dataset, filters, agency=agency)
    chunks = _chunk_multi_values(key_expr, max_combinations=max_combinations)

    req = Request(agency)
    all_rows: List[pd.DataFrame] = []
    last_status = None
    for key in chunks:
        attempt = 0
        while True:
            try:
                params = {"compressed": False, "format": "SDMX-CSV"}
                if start_period:
                    params["startPeriod"] = start_period
                if end_period:
                    params["endPeriod"] = end_period
                # Best-effort URL (reliable for ESTAT); pandasdmx may not expose response before request
                best_url = None
                if agency.upper() == "ESTAT":
                    base = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data"
                    best_url = f"{base}/{dataset}/{key}?{urlencode(params)}"

                msg = req.data(dataset, key=key, params=params)
                # Simple emptiness check
                if not getattr(msg, "data", None):
                    raise ValueError(f"Empty data for {dataset} with key={key}")
                # Track HTTP status if available
                try:
                    last_status = msg.response.status_code
                except Exception:
                    pass
                df = _to_tidy(msg, dataset)
                all_rows.append(df)
                # success event
                ev = {
                    "dataset": dataset,
                    "agency": agency,
                    "key": key,
                    "params": params,
                    "status": last_status,
                    "url": best_url,
                    "ok": True,
                }
                logging.info("fetch_series ok: %s", {k: ev[k] for k in ("dataset","key","status","url")})
                _push_fetch_event(ev)
                break
            except Exception as e:
                attempt += 1
                # failure event (retryable)
                try:
                    ev = {
                        "dataset": dataset,
                        "agency": agency,
                        "key": key,
                        "params": params,
                        "status": last_status,
                        "url": best_url,
                        "ok": False,
                        "error": str(e),
                        "attempt": attempt,
                    }
                    logging.warning("fetch_series retry: %s", {k: ev[k] for k in ("dataset","key","status","url","attempt")})
                    _push_fetch_event(ev)
                except Exception:
                    pass
                if attempt > retries:
                    # Raise with enriched context for easier debugging
                    ctx = {
                        "agency": agency,
                        "dataset": dataset,
                        "key": key,
                        "params": params,
                        "status": last_status,
                    }
                    if best_url:
                        ctx["url"] = best_url
                    logging.error("fetch_series failed: %s", ctx, exc_info=True)
                    raise RuntimeError(f"fetch_series error: {ctx}") from e
                time.sleep(backoff_base * (2 ** (attempt - 1)))
    if not all_rows:
        raise ValueError(f"No data returned for {dataset}")
    out = pd.concat(all_rows, ignore_index=True)
    # Place time first if present
    cols = list(out.columns)
    if "time" in cols:
        cols = ["time"] + [c for c in cols if c != "time"]
        out = out[cols]
    return out


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def pick_anchor_row(df_nf: pd.DataFrame) -> pd.DataFrame:
    """Select the single anchor row: latest B9, S13."""
    if df_nf.empty:
        return df_nf
    mask = (df_nf.get("na_item").eq("B9")) & (df_nf.get("sector").eq("S13"))
    sdf = df_nf.loc[mask].copy()
    if sdf.empty:
        return sdf
    latest_time = sdf["time"].max()
    return sdf.loc[sdf["time"].eq(latest_time)].sort_values(by=["time"]).tail(1)
