# Dataset Source Information & Access Instructions

## 1. Overview of Data Sources
This project integrates two free, public, legally accessible data sources to build a time- and location-aware traffic accident warehouse and a severity prediction pipeline.

| Source | Publisher | Licence | Used for |
| :--- | :--- | :--- | :--- |
| Road Safety Data – collisions, last 5 years | UK Department for Transport (DfT) | Open Government Licence v3.0 | Accident records (primary dataset) |
| Historical Weather API | Open-Meteo | CC BY 4.0 (data), no API key | Hourly weather enrichment |

---

## 2. Primary Dataset: UK DfT Road Safety Data (STATS19)
- **Landing page**: [data.gov.uk – Road Safety Data](https://www.data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data)
- **File used**: `dft-road-casualty-statistics-collision-last-5-years.csv`
- **Direct URL**: `https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-last-5-years.csv`
- **Snapshot used in this project**: 513,801 collisions, 2021–2025 (≈98 MB). Severity mix: 1.47% Fatal, 22.74% Serious, 75.79% Slight.
- **Format**: one row per collision, 44 columns. Categorical fields are numeric codes (decoded with the DfT *Road Safety Open Data Guide*, see `docs/data_dictionary.md` §4). Dates are `DD/MM/YYYY`, times `HH:MM` (UK local time). `-1` means "data missing or out of range".
- **Columns used** (staging name ← DfT name): `accident_index` ← `collision_index`, `accident_date` ← `date`, `accident_time` ← `time`, `latitude`, `longitude`, `accident_severity` ← `collision_severity`, `number_of_vehicles`, `number_of_casualties`, `speed_limit`, `road_type`, `light_conditions`, `weather_conditions`, `road_surface_conditions`, `urban_or_rural_area`, `local_authority` ← `local_authority_ons_district`. Older releases that use `accident_*` names are also accepted.

### How to obtain it
You do not need to download it by hand. `src/ingestion/accident_ingest.py` downloads the file into `datasets/` the first time it runs (configurable with `DFT_COLLISION_URL` / `DFT_COLLISION_FILE`). To use a copy you already have, place it at:

```text
datasets/dft-road-casualty-statistics-collision-last-5-years.csv
```

`datasets/` is gitignored, so the ~100 MB file is never committed.

---

## 3. Weather Enrichment: Open-Meteo Historical Weather API
- **Endpoint**: `https://archive-api.open-meteo.com/v1/archive` (no API key)
- **Request parameters**: `latitude`, `longitude` (weather grid cell centre), `start_date` / `end_date` (ISO `YYYY-MM-DD`), `hourly=temperature_2m,precipitation,wind_speed_10m,visibility,weather_code`, `timezone=Europe/London` (so hours line up with DfT local times).
- **Join method**: each accident is snapped to a grid cell of `WEATHER_GRID_DEG` degrees (default 1.0°) and to its local hour; one request per grid cell and calendar year returns the hourly series, and the ETL joins on (cell, hour).
- **Why 1.0°**: the free tier counts a year-long request as roughly 26 calls and allows about 10,000 calls a day. A 5-year backfill needs ~7,600 weighted calls at 1.0° (62 cells) but ~24,000 at 0.5° (196 cells). Smaller daily batches (`ACCIDENT_BATCH_SIZE`) fit a finer grid.
- **Failures**: failed or rate-limited requests are written to `ingestion_audit` (status `PARTIAL`/`FAILED`); no substitute values are invented. The next run fetches only the hours that are still missing.

---

## 4. Raw Landing Zone & Audit
- Accident batches: `data/raw/accidents_raw_<batch_id>.csv` (exactly the rows staged in that batch).
- Weather batches: `data/raw/weather_raw_<batch_id>.jsonl.gz` (complete API responses, one JSON line per request).
- Every run writes a row to `ingestion_audit` with source, file/endpoint, status, row counts, timestamps and errors.
- Raw files are never modified; `data/raw/` is gitignored.
