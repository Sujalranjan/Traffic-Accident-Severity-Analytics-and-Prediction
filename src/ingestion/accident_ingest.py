"""
Ingestion script for UK DfT Road Safety (STATS19) collision records.

1. Downloads the public DfT collision CSV into datasets/ if it is not already there.
2. Selects only collisions that have not been loaded or quarantined before
   (so a scheduled re-run picks up new records instead of duplicating old ones).
3. Writes an immutable raw copy of the batch to data/raw/.
4. Appends the batch to stg_accidents_raw and records the run in ingestion_audit.
"""

import os
import sys
import uuid
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import requests
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.etl.db import PROJECT_ROOT, DATA_DIR, get_engine, init_db
from src.etl.standardize import parse_accident_date

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("accident_ingest")

SOURCE_NAME = "UK DfT Road Safety Data (STATS19 collisions)"
DFT_COLLISION_URL = os.getenv(
    "DFT_COLLISION_URL",
    "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-last-5-years.csv",
)
DEFAULT_SOURCE_FILE = Path(os.getenv(
    "DFT_COLLISION_FILE",
    str(PROJECT_ROOT / "datasets" / "dft-road-casualty-statistics-collision-last-5-years.csv"),
))

# Staging column -> accepted DfT header names (newer releases renamed accident_* to collision_*)
COLUMN_ALIASES = {
    "accident_index": ["collision_index", "accident_index"],
    "accident_date": ["date"],
    "accident_time": ["time"],
    "latitude": ["latitude"],
    "longitude": ["longitude"],
    "accident_severity": ["collision_severity", "accident_severity"],
    "number_of_vehicles": ["number_of_vehicles"],
    "number_of_casualties": ["number_of_casualties"],
    "speed_limit": ["speed_limit"],
    "road_type": ["road_type"],
    "light_conditions": ["light_conditions"],
    "weather_conditions": ["weather_conditions"],
    "road_surface_conditions": ["road_surface_conditions"],
    "urban_or_rural_area": ["urban_or_rural_area"],
    "local_authority": ["local_authority_ons_district", "local_authority_district"],
}


def download_source_file(dest: Path = DEFAULT_SOURCE_FILE, url: str = DFT_COLLISION_URL) -> Path:
    """Streams the DfT collision CSV to disk (written to a .part file first, then renamed)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    logger.info(f"Downloading DfT collision data from {url} ...")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        with open(part, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    part.replace(dest)
    logger.info(f"Downloaded {dest.stat().st_size / 1e6:.1f} MB to {dest}")
    return dest


def read_dft_collisions(source_file: Path) -> pd.DataFrame:
    """Reads the DfT CSV using its own header row and maps it to the staging schema."""
    header = pd.read_csv(source_file, nrows=0).columns.str.strip().str.lower().tolist()
    selected = {}
    for target, aliases in COLUMN_ALIASES.items():
        match = next((a for a in aliases if a in header), None)
        if match is None:
            raise ValueError(f"Source file is missing a column for '{target}' (looked for {aliases})")
        selected[match] = target

    df = pd.read_csv(source_file, dtype=str, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()
    df = df[list(selected)].rename(columns=selected)
    for col in ["latitude", "longitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["number_of_vehicles", "number_of_casualties", "speed_limit"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def _already_seen_indices(engine) -> set:
    """Collision indices already loaded, quarantined, or waiting in staging."""
    seen = set()
    for query in [
        "SELECT accident_index FROM fact_accidents",
        "SELECT accident_index FROM stg_accidents_raw",
        "SELECT record_identifier FROM quarantine_rejected_records WHERE source_entity = 'stg_accidents_raw'",
    ]:
        seen.update(pd.read_sql(query, con=engine).iloc[:, 0].dropna().astype(str))
    return seen


def _write_audit(engine, batch_id, file_or_endpoint, status, records, started, error=None):
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO ingestion_audit
                (batch_id, source_name, file_or_endpoint, status, records_ingested, records_rejected,
                 started_at, completed_at, error_message)
                VALUES (:batch_id, :source, :path, :status, :records, 0, :started, :completed, :err)
            """),
            {
                "batch_id": batch_id,
                "source": SOURCE_NAME,
                "path": str(file_or_endpoint),
                "status": status,
                "records": records,
                "started": started,
                "completed": datetime.now(timezone.utc),
                "err": error,
            },
        )


def ingest_accidents(source_file=None, batch_size=None, download_if_missing: bool = True) -> str:
    """
    Ingests the next batch of unseen DfT collisions.

    batch_size: maximum number of new collisions to take, oldest first. None takes all
    unseen rows. Set ACCIDENT_BATCH_SIZE to make each scheduled run load the next slice,
    which simulates data arriving over time.
    Returns the batch_id (also used by the DAG via XCom).
    """
    batch_id = f"batch_acc_{uuid.uuid4().hex[:8]}"
    source_file = Path(source_file) if source_file else DEFAULT_SOURCE_FILE
    if batch_size is None and os.getenv("ACCIDENT_BATCH_SIZE"):
        batch_size = int(os.getenv("ACCIDENT_BATCH_SIZE"))

    engine = get_engine()
    init_db(engine)
    start_time = datetime.now(timezone.utc)

    try:
        if not source_file.exists():
            if not download_if_missing:
                raise FileNotFoundError(f"DfT source file not found: {source_file}")
            download_source_file(source_file)

        df = read_dft_collisions(source_file)
        total_in_file = len(df)

        seen = _already_seen_indices(engine)
        df = df[~df["accident_index"].astype(str).isin(seen)].copy()

        # Oldest first, so incremental batches follow the calendar
        df["_sort_date"] = parse_accident_date(df["accident_date"])
        df = df.sort_values(["_sort_date", "accident_time"], na_position="last").drop(columns="_sort_date")
        if batch_size:
            df = df.head(batch_size)
        df["batch_id"] = batch_id

        logger.info(f"{total_in_file:,} collisions in source file; {len(df):,} new collisions selected for {batch_id}.")

        if not df.empty:
            raw_dir = DATA_DIR / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_file = raw_dir / f"accidents_raw_{batch_id}.csv"
            df.to_csv(raw_file, index=False)
            logger.info(f"Saved raw landing file to: {raw_file}")
            df.to_sql("stg_accidents_raw", con=engine, if_exists="append", index=False, chunksize=10000)

        _write_audit(engine, batch_id, source_file, "SUCCESS", len(df), start_time)
        logger.info(f"Ingestion batch {batch_id} logged to audit ({len(df):,} records).")
        return batch_id

    except Exception as e:
        logger.exception(f"Accident ingestion failed for batch {batch_id}")
        _write_audit(engine, batch_id, source_file, "FAILED", 0, start_time, error=f"{e.__class__.__name__}: {e}")
        raise


if __name__ == "__main__":
    ingest_accidents()
