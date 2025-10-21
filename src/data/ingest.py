import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .schemas import ProcessedSchema

RAW_PATH = Path("data/raw/spare_parts_sales.csv")
OUT_PATH = Path("data/processed/train.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

REQUIRED_COLS: Dict[str, List[str]] = {
    "sku": ["sku", "product_code", "product", "item", "product_id", "code", "sku_id"],
    "ds": ["ds", "date", "order_date", "timestamp"],
    "y": ["y", "sales", "order_demand", "demand", "qty", "quantity"],
}

OPTIONAL_COLS: Dict[str, List[str]] = {
    "warehouse": ["warehouse", "store", "location", "whse", "depot"],
    "promo": ["promo", "onpromotion", "promotion", "is_promo"],
    "open": ["open", "is_open", "store_open"],
    "state_holiday": ["state_holiday", "stateholiday"],
    "school_holiday": ["school_holiday", "schoolholiday"],
    "petrol_price": ["petrol_price", "petrolprice", "gas_price", "fuel_price"],
    "product_category": ["product_category", "category"],
    "product_id": ["product_id", "sku_id"],
}


def find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    """Return the actual column in df matching any candidate name."""
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_cols:
            return lower_cols[cand.lower()]
    return ""


def resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Resolve canonical column names to actual dataframe columns."""
    resolved: Dict[str, str] = {}
    missing_required: List[str] = []

    for canonical, candidates in REQUIRED_COLS.items():
        match = find_col(df, candidates)
        if match:
            resolved[canonical] = match
        else:
            missing_required.append(canonical)

    if missing_required:
        print(
            f"[ingest] ERROR: Missing required columns: {missing_required}",
            file=sys.stderr,
        )
        print("[ingest] Available columns:", list(df.columns), file=sys.stderr)
        return {}

    for canonical, candidates in OPTIONAL_COLS.items():
        match = find_col(df, candidates)
        if match:
            resolved[canonical] = match
        else:
            print(f"[ingest] INFO: Optional column '{canonical}' not found")

    mapping_info = ", ".join(f"{k}={resolved[k]}" for k in sorted(resolved.keys()))
    print(f"[ingest] INFO: Column mapping -> {mapping_info}")
    return resolved


def coerce_optional_int(df: pd.DataFrame, column: str) -> None:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].clip(lower=0)
        df[column] = df[column].round().astype("Int64")


def coerce_optional_float(df: pd.DataFrame, column: str) -> None:
    if column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].astype(float)


def main() -> int:
    if not RAW_PATH.exists():
        print(
            f"[ingest] ERROR: Raw CSV file not found at '{RAW_PATH}'", file=sys.stderr
        )
        return 2

    if not os.access(RAW_PATH, os.R_OK):
        print(
            f"[ingest] ERROR: No read permissions for file '{RAW_PATH}'",
            file=sys.stderr,
        )
        return 5

    try:
        df = pd.read_csv(RAW_PATH, sep=";")
    except pd.errors.EmptyDataError:
        print(
            f"[ingest] ERROR: CSV file '{RAW_PATH}' is empty or malformed",
            file=sys.stderr,
        )
        return 3
    except pd.errors.ParserError as exc:
        print(
            f"[ingest] ERROR: Failed to parse CSV file '{RAW_PATH}': {exc}",
            file=sys.stderr,
        )
        return 7
    except Exception as exc:  # pylint: disable=broad-except
        print(
            f"[ingest] ERROR: Unexpected error reading '{RAW_PATH}': {exc}",
            file=sys.stderr,
        )
        return 8

    if df.empty:
        print(
            f"[ingest] ERROR: No data found in CSV file '{RAW_PATH}'", file=sys.stderr
        )
        return 3

    print(
        f"[ingest] INFO: Found CSV file with {len(df)} rows and {len(df.columns)} columns"
    )
    print(f"[ingest] INFO: Available columns: {list(df.columns)}")

    resolved = resolve_columns(df)
    if not resolved:
        return 4

    rename_map = {actual: canonical for canonical, actual in resolved.items()}
    df = df.rename(columns=rename_map)

    keep_cols = list(REQUIRED_COLS.keys()) + [
        col for col in OPTIONAL_COLS if col in df.columns
    ]
    df = df[keep_cols]

    df["sku"] = df["sku"].astype(str).str.strip()
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df[df["ds"].notna()]

    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df[df["y"].notna()]
    df.loc[df["y"] < 0, "y"] = 0
    df["y"] = df["y"].astype(float)

    if "warehouse" in df.columns:
        df["warehouse"] = df["warehouse"].astype(str).str.strip()

    for text_col in ["product_category", "product_id"]:
        if text_col in df.columns:
            df[text_col] = df[text_col].astype(str).str.strip()

    for col in ["promo", "open", "state_holiday", "school_holiday"]:
        coerce_optional_int(df, col)

    coerce_optional_float(df, "petrol_price")

    entity_key = ["sku"]
    if "warehouse" in df.columns:
        entity_key.append("warehouse")

    df = (
        df.sort_values(entity_key + ["ds"])
        .drop_duplicates(subset=entity_key + ["ds"])
        .reset_index(drop=True)
    )

    if df["y"].isna().any():
        print(
            "[ingest] ERROR: Found missing target values after coercion",
            file=sys.stderr,
        )
        return 6

    ProcessedSchema.validate(df, lazy=True)

    df.to_parquet(OUT_PATH, index=False)

    n_rows = len(df)
    n_entities = df[entity_key].drop_duplicates().shape[0]
    tmin, tmax = df["ds"].min(), df["ds"].max()
    print(
        f"[ingest] wrote {OUT_PATH} | rows={n_rows} | entities={n_entities} "
        f"| range=[{tmin.date()} .. {tmax.date()}] | key={entity_key}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
