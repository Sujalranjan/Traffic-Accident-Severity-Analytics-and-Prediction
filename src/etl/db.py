"""
Database connection manager supporting PostgreSQL with transparent fallback to SQLite.
"""

import os
import sys
import uuid
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("db_manager")

# An empty DATABASE_URL skips PostgreSQL and uses SQLite directly.
POSTGRES_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/traffic_accidents")
SQLITE_PATH = os.getenv("SQLITE_FALLBACK_PATH", str(PROJECT_ROOT / "data" / "traffic_accidents.db"))
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
SQL_DIR = PROJECT_ROOT / "sql"

# Tables created by this project, in drop order (facts and marts before dimensions)
PROJECT_TABLES = [
    "agg_hourly_severity_mart", "agg_weather_road_risk_mart", "agg_location_hotspots_mart",
    "fact_accidents", "dim_date", "dim_location", "dim_road", "dim_weather", "dim_severity",
    "stg_accidents_raw", "stg_weather_raw", "quarantine_rejected_records", "ingestion_audit",
    "prediction_log",
]

_ENGINE = None


def get_engine():
    """
    Returns a cached SQLAlchemy engine. Tries PostgreSQL first;
    if connection fails, falls back to SQLite.
    """
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE

    if POSTGRES_URL:
        try:
            pg_engine = create_engine(POSTGRES_URL, pool_pre_ping=True, connect_args={"connect_timeout": 3})
            with pg_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Successfully connected to PostgreSQL warehouse.")
            _ENGINE = pg_engine
            return _ENGINE
        except (OperationalError, Exception) as e:
            logger.warning(f"PostgreSQL connection failed ({e}). Falling back to SQLite.")

    sqlite_file = Path(SQLITE_PATH)
    if not sqlite_file.is_absolute():
        sqlite_file = PROJECT_ROOT / sqlite_file
    sqlite_file.parent.mkdir(parents=True, exist_ok=True)
    sqlite_url = f"sqlite:///{sqlite_file.resolve()}"
    logger.info(f"Using local SQLite database at: {sqlite_url}")
    _ENGINE = create_engine(sqlite_url, connect_args={"check_same_thread": False})
    return _ENGINE


def is_sqlite(engine) -> bool:
    return "sqlite" in str(engine.url)


def _split_statements(content: str) -> list:
    return [s.strip() for s in content.split(";") if s.strip()]


def _adapt_for_sqlite(content: str) -> str:
    content = content.replace("SERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
    content = content.replace("TIMESTAMP WITH TIME ZONE", "TIMESTAMP")
    content = content.replace("NUMERIC(10, 6)", "REAL")
    content = content.replace("NUMERIC(5, 2)", "REAL")
    content = content.replace("NUMERIC(8, 2)", "REAL")
    content = content.replace("NUMERIC(9, 4)", "REAL")
    content = content.replace("BOOLEAN", "INTEGER")
    return content


def init_db(engine=None):
    """
    Executes DDL schemas to ensure tables and views exist.
    """
    if engine is None:
        engine = get_engine()

    ddl_files = ["01_staging_and_quarantine.sql", "02_star_schema.sql"]
    sqlite = is_sqlite(engine)

    with engine.begin() as conn:
        for file_name in ddl_files:
            file_path = SQL_DIR / file_name
            logger.info(f"Applying schema: {file_name}")
            content = file_path.read_text(encoding="utf-8")
            if sqlite:
                content = _adapt_for_sqlite(content)
            for stmt in _split_statements(content):
                conn.execute(text(stmt))

        seed_severity = [
            {"severity_key": 1, "severity_code": 1, "severity_name": "Fatal", "severity_description": "Accidents involving at least one fatality"},
            {"severity_key": 2, "severity_code": 2, "severity_name": "Serious", "severity_description": "Accidents involving severe/hospitalizing injuries"},
            {"severity_key": 3, "severity_code": 3, "severity_name": "Slight", "severity_description": "Accidents involving minor or slight injuries"},
        ]
        insert_sql = (
            "INSERT OR IGNORE INTO dim_severity (severity_key, severity_code, severity_name, severity_description) "
            "VALUES (:severity_key, :severity_code, :severity_name, :severity_description)"
            if sqlite else
            "INSERT INTO dim_severity (severity_key, severity_code, severity_name, severity_description) "
            "VALUES (:severity_key, :severity_code, :severity_name, :severity_description) ON CONFLICT (severity_key) DO NOTHING"
        )
        conn.execute(text(insert_sql), seed_severity)

    if not sqlite:
        apply_postgis(engine)

    logger.info("Database schemas successfully initialized.")


def apply_postgis(engine):
    """
    PostgreSQL only: enables PostGIS and keeps dim_location.geom in sync with lat/lon.
    Each statement runs in its own transaction so a server without PostGIS
    degrades to plain lat/lon columns instead of aborting the pipeline.
    """
    content = (SQL_DIR / "04_postgis.sql").read_text(encoding="utf-8")
    for stmt in _split_statements(content):
        try:
            with engine.begin() as conn:
                conn.execute(text(stmt))
        except Exception as ex:
            logger.warning(f"PostGIS statement skipped ({ex.__class__.__name__}): {stmt.splitlines()[0][:80]}")
            return


def run_sql_file(engine, file_name: str):
    """Runs a SQL script (e.g. the mart refresh) statement by statement in one transaction."""
    content = (SQL_DIR / file_name).read_text(encoding="utf-8")
    if is_sqlite(engine):
        content = _adapt_for_sqlite(content)
    with engine.begin() as conn:
        for stmt in _split_statements(content):
            conn.execute(text(stmt))


def insert_ignore_duplicates(engine, df: pd.DataFrame, table: str, conflict_cols: list):
    """
    Bulk-inserts a DataFrame, skipping rows that violate the table's unique key.
    Loads into a scratch table first so the insert is one set-based statement
    on both SQLite and PostgreSQL. The scratch table name is unique per call so
    concurrent runs never collide.
    """
    if df.empty:
        return
    tmp = f"tmp_load_{table}_{uuid.uuid4().hex[:8]}"
    cols = ", ".join(df.columns)
    try:
        df.to_sql(tmp, con=engine, if_exists="replace", index=False, chunksize=5000)
        if is_sqlite(engine):
            sql = f"INSERT OR IGNORE INTO {table} ({cols}) SELECT {cols} FROM {tmp}"
        else:
            # Cast each scratch column to the target type: pandas writes an all-null column as TEXT,
            # which PostgreSQL will not insert into a NUMERIC/DATE column without a cast.
            types = pd.read_sql(
                text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = :t"),
                con=engine, params={"t": table},
            ).set_index("column_name")["data_type"].to_dict()
            select = ", ".join(f"CAST({c} AS {types[c]})" if c in types else c for c in df.columns)
            sql = f"INSERT INTO {table} ({cols}) SELECT {select} FROM {tmp} ON CONFLICT ({', '.join(conflict_cols)}) DO NOTHING"
        with engine.begin() as conn:
            conn.execute(text(sql))
    finally:
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {tmp}"))


def reset_db(engine=None):
    """Drops every project table and view so the schema can be rebuilt from scratch."""
    if engine is None:
        engine = get_engine()
    cascade = "" if is_sqlite(engine) else " CASCADE"
    with engine.begin() as conn:
        conn.execute(text("DROP VIEW IF EXISTS v_model_training_dataset"))
        for table in PROJECT_TABLES:
            conn.execute(text(f"DROP TABLE IF EXISTS {table}{cascade}"))
    logger.info("All project tables dropped.")


if __name__ == "__main__":
    eng = get_engine()
    if "--reset" in sys.argv:
        reset_db(eng)
    init_db(eng)
