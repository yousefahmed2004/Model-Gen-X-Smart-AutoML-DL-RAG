"""
Dataset inspection: load a file (CSV / Excel), extract metadata,
and produce a JSON-safe preview for the UI.

For files >500 MB we'd ideally hand off to Dask — that path is stubbed
below and clearly marked so a future engineer can plug it in.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SAMPLE_PREVIEW_ROWS = 10
BIG_FILE_THRESHOLD_MB = 500   # > this → use Dask (stub)


def _json_safe(v: Any) -> Any:
    """Pandas/Numpy values aren't always JSON-serializable."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if math.isnan(f) else f
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def load_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a tabular file based on its extension."""
    p = Path(path)
    suffix = p.suffix.lower()

    size_mb = p.stat().st_size / (1024 * 1024)
    if size_mb > BIG_FILE_THRESHOLD_MB:
        # TODO: replace with Dask streaming load. For now we still try pandas
        # but warn the caller.
        pass

    if suffix in {".csv", ".tsv", ".txt"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(p, sep=sep)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(p)
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix == ".json":
        return pd.read_json(p)
    raise ValueError(f"Unsupported file type: {suffix}")


def inspect_dataset(path: str | Path) -> dict[str, Any]:
    """Return rich metadata + preview rows for a dataset."""
    df = load_dataframe(path)

    columns: list[dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        n_missing = int(s.isna().sum())
        sample_vals = [_json_safe(v) for v in s.dropna().head(3).tolist()]
        unique = int(s.nunique(dropna=True))
        columns.append({
            "name": str(col),
            "dtype": dtype,
            "n_missing": n_missing,
            "n_unique": unique,
            "sample": sample_vals,
        })

    head = [
        {str(k): _json_safe(v) for k, v in row.items()}
        for row in df.head(SAMPLE_PREVIEW_ROWS).to_dict(orient="records")
    ]

    return {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": columns,
        "head": head,
    }


def suggest_target(columns: list[dict[str, Any]]) -> str | None:
    """Heuristic target-column suggestion: last column wins, then any
    column literally named 'target' / 'label' / 'class' / 'y'."""
    if not columns:
        return None
    names = [c["name"] for c in columns]
    for cand in ("target", "label", "class", "y", "outcome"):
        for n in names:
            if n.lower() == cand:
                return n
    return names[-1]
