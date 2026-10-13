"""
Data Quality Validation and Quarantine Pipeline.
Enforces business rules, filters invalid/duplicate records, and logs rejected rows to quarantine.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import json
import logging
import datetime
import numpy as np
import pandas as pd
from sqlalchemy import text
from src.etl.db import DATA_DIR, get_engine
from src.etl.standardize import parse_accident_date, parse_accident_time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("data_validation")

# Rules are applied in order; a record is quarantined with the first rule it fails.
RULES = [
    ("VR-01", "INVALID_COORDINATES: Lat/Lon missing or outside UK bounds (49.0-61.5 N, -8.5-2.5 E)"),
    ("VR-03", "INVALID_DATETIME: Date (DD/MM/YYYY) or time (HH:MM) missing or unparseable"),
    ("VR-02", "INVALID_SPEED_LIMIT: Speed limit missing or not in range [10, 100] (DfT -1 = missing)"),
    ("VR-04", "INVALID_SEVERITY: Severity not mapped to 1, 2, or 3"),
    ("VR-05", "DUPLICATE_ACCIDENT_INDEX: Index missing, repeated in batch, or already loaded"),
]


def rule_failures(df: pd.DataFrame, existing_indices: set) -> dict:
    """Returns {rule_id: boolean Series of failing rows}."""
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    speed = pd.to_numeric(df["speed_limit"], errors="coerce")
    severity = pd.to_numeric(df["accident_severity"], errors="coerce")
    index = df["accident_index"].astype("string").str.strip()

    return {
        "VR-01": lat.isna() | lon.isna() | ~lat.between(49.0, 61.5) | ~lon.between(-8.5, 2.5),
        "VR-03": parse_accident_date(df["accident_date"]).isna() | parse_accident_time(df["accident_time"]).isna(),
        "VR-02": speed.isna() | ~speed.between(10, 100),
        "VR-04": ~severity.isin([1, 2, 3]),
        "VR-05": index.isna() | (index == "") | index.duplicated(keep="first") | index.isin(existing_indices),
    }


def validate_and_quarantine(batch_id: str = None, persist: bool = True) -> dict:
    """
    Validates records in stg_accidents_raw (one batch, or everything staged).
    persist=True writes failures to quarantine_rejected_records and updates the
    batch's audit row; persist=False only returns the split (used when the DAG has
    already run the quality gate as its own task).
    """
    engine = get_engine()

    if batch_id:
        df = pd.read_sql(text("SELECT * FROM stg_accidents_raw WHERE batch_id = :b"), con=engine, params={"b": batch_id})
    else:
        df = pd.read_sql("SELECT * FROM stg_accidents_raw", con=engine)

    if df.empty:
        logger.warning("No records found in staging to validate.")
        return {"total": 0, "valid": 0, "rejected": 0, "valid_df": df, "batches": set()}

    total_records = len(df)
    logger.info(f"Starting quality checks on {total_records:,} records...")

    existing = set(pd.read_sql("SELECT accident_index FROM fact_accidents", con=engine)["accident_index"].astype(str))
    failures = rule_failures(df, existing)

    reason = pd.Series(np.nan, index=df.index, dtype=object)
    for rule_id, message in RULES:
        reason = reason.where(reason.notna() | ~failures[rule_id], message)

    valid_mask = reason.isna()
    valid_df = df[valid_mask].copy()
    rejected_df = df[~valid_mask].copy()
    breakdown = reason.dropna().value_counts().to_dict()

    if persist and not rejected_df.empty:
        now = datetime.datetime.now(datetime.timezone.utc)
        quarantine = pd.DataFrame({
            "batch_id": rejected_df["batch_id"].fillna("UNKNOWN"),
            "source_entity": "stg_accidents_raw",
            "record_identifier": rejected_df["accident_index"].fillna("").astype(str),
            "rejection_reason": reason[~valid_mask],
            "raw_payload": rejected_df.apply(lambda r: r.to_json(), axis=1),
            "rejected_at": now,
        })
        quarantine.to_sql("quarantine_rejected_records", con=engine, if_exists="append", index=False, chunksize=5000)

        # Record rejected counts against each batch's ingestion audit row
        with engine.begin() as conn:
            for b, count in rejected_df["batch_id"].value_counts().items():
                conn.execute(
                    text("UPDATE ingestion_audit SET records_rejected = :n WHERE batch_id = :b"),
                    {"n": int(count), "b": b},
                )

        rejected_dir = DATA_DIR / "rejected"
        rejected_dir.mkdir(parents=True, exist_ok=True)
        summary_path = rejected_dir / f"quarantine_summary_{batch_id or 'all'}_{datetime.date.today()}.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.datetime.now().isoformat(),
                "batch_id": batch_id,
                "total_evaluated": total_records,
                "passed_validation": len(valid_df),
                "quarantined_count": len(rejected_df),
                "reasons_breakdown": breakdown,
            }, f, indent=2)

        logger.warning(f"Quarantined {len(rejected_df):,} invalid records out of {total_records:,}. Logged to quarantine_rejected_records.")
    elif rejected_df.empty:
        logger.info("All records passed data quality checks!")

    return {
        "total": total_records,
        "valid": len(valid_df),
        "rejected": len(rejected_df),
        "reasons": breakdown,
        "valid_df": valid_df,
        "batches": set(df["batch_id"].dropna()),
    }


if __name__ == "__main__":
    validate_and_quarantine()
