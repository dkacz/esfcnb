#!/usr/bin/env python3
"""
Repair a malformed Jupyter notebook (.ipynb) with a robust, staged approach:

1) Backup the broken file
2) Try restoring from checkpoint (validate with nbformat)
3) Try programmatic JSON repairs (trailing commas, NaN/Infinity, control chars)
4) As a last resort, salvage code/markdown sources and regenerate a minimal valid notebook

Outputs:
- <nb>.bak backup
- <nb> repaired in place if successful
- notebooks/01_data_spine_pl_salvaged.py if regeneration was needed
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def backup_file(src: Path) -> Path:
    cand = src.with_suffix(src.suffix + ".bak")
    if cand.exists():
        cand = src.with_suffix(src.suffix + f".bak.{now_iso()}")
    shutil.copy2(src, cand)
    return cand


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_nb_from_dict(doc: dict) -> Tuple[bool, str]:
    try:
        import nbformat
    except Exception as e:
        return False, f"nbformat import failed: {e}"
    try:
        # from_dict ensures proper object; validate enforces schema
        nb = nbformat.from_dict(doc)
        nbformat.validate(nb)
        return True, "ok"
    except Exception as e:
        return False, f"nbformat validate error: {e}"


def validate_nb_file(path: Path) -> Tuple[bool, str]:
    try:
        with path.open("r", encoding="utf-8") as f:
            doc = json.load(f)
        return validate_nb_from_dict(doc)
    except Exception as e:
        return False, f"json load error: {e}"


def try_checkpoint_restore(target: Path) -> Tuple[bool, str]:
    chk = target.parent / ".ipynb_checkpoints" / (target.stem + "-checkpoint" + target.suffix)
    if not chk.exists():
        return False, "no checkpoint"
    try:
        doc = load_json(chk)
    except Exception as e:
        return False, f"checkpoint json error: {e}"
    ok, msg = validate_nb_from_dict(doc)
    if ok:
        with target.open("w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        return True, f"restored from checkpoint: {chk}"
    else:
        return False, msg


@dataclass
class RepairResult:
    changed: bool
    text: str
    notes: List[str]


def basic_text_repairs(text: str) -> RepairResult:
    notes: List[str] = []
    orig = text

    # 1) Remove BOM and non-printable control chars (except tab, newline)
    if text and text[0] == "\ufeff":
        text = text.lstrip("\ufeff")
        notes.append("removed UTF-8 BOM")

    ctrl_re = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
    if ctrl_re.search(text):
        text = ctrl_re.sub("", text)
        notes.append("stripped control characters")

    # 2) Replace NaN/Infinity with null (JSON-valid)
    repl_map = {
        ": NaN": ": null",
        ": -NaN": ": null",
        ": Infinity": ": null",
        ": -Infinity": ": null",
    }
    for k, v in repl_map.items():
        if k in text:
            cnt = text.count(k)
            text = text.replace(k, v)
            notes.append(f"replaced {cnt} occurrence(s) of {k} -> {v}")

    # 3) Remove trailing commas before } or ]
    #    This is a common corruption when editing JSON by hand
    tc_pat = re.compile(r",\s*(?=[}\]])")
    text2 = tc_pat.sub("", text)
    if text2 != text:
        notes.append("removed trailing commas before } or ]")
        text = text2

    # 4) Fix unescaped lone backslashes in strings (simple heuristic)
    #    Replace occurrences of "\\\n" with "\\\\\n" within the whole text as a best-effort
    if "\\\n" in text:
        cnt = text.count("\\\n")
        text = text.replace("\\\n", "\\\\\n")
        notes.append(f"escaped {cnt} lone backslash-newline occurrences")

    return RepairResult(changed=(text != orig), text=text, notes=notes)


def attempt_programmatic_repair(target: Path) -> Tuple[bool, str, List[str]]:
    notes: List[str] = []
    raw = target.read_text(encoding="utf-8", errors="ignore")

    rr = basic_text_repairs(raw)
    notes.extend(rr.notes)
    candidate = rr.text

    # Try to load after basic repairs
    try:
        doc = json.loads(candidate)
        ok, msg = validate_nb_from_dict(doc)
        if ok:
            target.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            return True, "programmatic repair via basic heuristics", notes
    except Exception as e:
        notes.append(f"json after basic repairs still invalid: {e}")

    # Fallback: try a tolerant parse by extracting cell-like structures (still as a repair step)
    return False, "programmatic repair failed", notes


def salvage_cells_from_text(text: str) -> Tuple[List[str], List[str]]:
    code_cells: List[str] = []
    md_cells: List[str] = []

    # Regex to capture cell_type then source array or string (non-greedy)
    cell_re = re.compile(
        r"\{[^\{\}]*?\"cell_type\"\s*:\s*\"(code|markdown)\"[\s\S]*?\"source\"\s*:\s*(\[[\s\S]*?\]|\"[\s\S]*?\")",
        re.MULTILINE,
    )

    # Helper: convert JSON-like source to a plain text string
    def normalize_source(src: str) -> str:
        src = src.strip()
        if src.startswith("["):
            # Attempt to parse a JSON array of strings; if that fails, degrade to manual join
            try:
                arr = json.loads(src)
                if isinstance(arr, list):
                    return "".join(arr)
            except Exception:
                pass
            # Strip brackets and split by quotes — crude fallback
            parts = re.findall(r"\"(.*?)\"", src, flags=re.DOTALL)
            return "".join([p for p in parts])
        else:
            # Single string value (strip leading/trailing quotes)
            if src.startswith('"') and src.endswith('"'):
                try:
                    return json.loads(src)
                except Exception:
                    return src.strip('"')
            return src

    for m in cell_re.finditer(text):
        ctype = m.group(1)
        src = normalize_source(m.group(2))
        if ctype == "code":
            code_cells.append(src)
        else:
            md_cells.append(src)

    return code_cells, md_cells


def regenerate_minimal_notebook(target: Path, code_cells: List[str], md_cells: List[str]) -> Tuple[bool, str, int, int, Path | None]:
    try:
        import nbformat
        from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell
    except Exception as e:
        return False, f"nbformat not available for regeneration: {e}", 0, 0, None

    nb = new_notebook()
    nb.cells = []
    # Intro cell
    nb.cells.append(new_markdown_cell("Recovered notebook — automated repair at " + now_iso()))
    # Add markdown
    for s in md_cells:
        nb.cells.append(new_markdown_cell(s))
    # Add code
    for s in code_cells:
        nb.cells.append(new_code_cell(s))
    # Minimal metadata
    nb.metadata.setdefault("kernelspec", {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    })
    nb.metadata.setdefault("language_info", {"name": "python", "version": f"{sys.version_info.major}.{sys.version_info.minor}"})

    try:
        nbformat.validate(nb)
    except Exception as e:
        return False, f"nbformat validation failed after regeneration: {e}", 0, 0, None

    # Write
    with target.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    return True, "regenerated minimal valid notebook from salvaged cells", len(code_cells), len(md_cells), None


def main():
    parser = argparse.ArgumentParser(description="Repair a malformed Jupyter .ipynb notebook")
    parser.add_argument("path", nargs="?", default="notebooks/01_data_spine_pl.ipynb", help="Path to the notebook to repair")
    args = parser.parse_args()

    target = Path(args.path).resolve()
    if not target.exists():
        print(f"ERROR: Target notebook not found: {target}")
        sys.exit(1)

    print(f"Repairing: {target}")

    # 1) Backup
    backup = backup_file(target)
    print(f"Backup created: {backup}")

    # 2) Try checkpoint restore
    ok, msg = try_checkpoint_restore(target)
    if ok:
        ok_val, vmsg = validate_nb_file(target)
        print(f"SUCCESS: {msg}; validation: {ok_val} ({vmsg})")
        print("Result: restored from checkpoint")
        return
    else:
        print(f"Checkpoint restore not used: {msg}")

    # 3) Programmatic JSON repair
    ok, msg, notes = attempt_programmatic_repair(target)
    if ok:
        ok_val, vmsg = validate_nb_file(target)
        print(f"SUCCESS: {msg}; validation: {ok_val} ({vmsg})")
        if notes:
            print("Repairs applied:")
            for n in notes:
                print("-", n)
        print("Result: repaired original file in place")
        return
    else:
        print(f"Programmatic repair failed: {msg}")
        if notes:
            print("Notes:")
            for n in notes:
                print("-", n)

    # 4) Minimal regeneration with salvage
    raw = target.read_text(encoding="utf-8", errors="ignore")
    code_cells, md_cells = salvage_cells_from_text(raw)

    # Save salvaged code for audit
    salvaged_py = target.parent / f"{target.stem}_salvaged.py"
    try:
        with salvaged_py.open("w", encoding="utf-8") as f:
            f.write("# Salvaged from malformed notebook at " + now_iso() + "\n\n")
            for i, s in enumerate(code_cells, 1):
                f.write(f"# --- code cell {i} ---\n")
                f.write(s)
                if not s.endswith("\n"):
                    f.write("\n")
                f.write("\n")
        print(f"Salvaged code written: {salvaged_py}")
    except Exception as e:
        print(f"WARN: Could not write salvaged code file: {e}")

    ok, msg, n_code, n_md, _ = regenerate_minimal_notebook(target, code_cells, md_cells)
    ok_val, vmsg = validate_nb_file(target)
    print(f"Regeneration: {msg}")
    print(f"Validation: {ok_val} ({vmsg})")
    print(f"Recovered cells — code: {n_code}, markdown: {n_md}")
    if ok and ok_val:
        print("Result: regenerated minimal valid notebook")
    else:
        print("Result: regeneration attempted but validation did not fully pass — manual review needed")


if __name__ == "__main__":
    main()

