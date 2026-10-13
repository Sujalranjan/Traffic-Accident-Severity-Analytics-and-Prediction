# Data Dictionary & Data Validation Rules

## 1. Relational & Dimensional Schema Dictionary

All categorical values in the warehouse are **decoded DfT labels** (for example `Single carriageway`), not the numeric codes published in the source CSV. The decoding tables are in section 4 and in `src/etl/standardize.py`.

### Table: `dim_date`
Stores calendar and time-of-day dimensional attributes for time-aware analytical slices.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `date_key` | INT | PRIMARY KEY | Unique integer key formatted as `YYYYMMDDHH` |
| `full_date` | DATE | NOT NULL | Calendar date in ISO format (`YYYY-MM-DD`), parsed from DfT `DD/MM/YYYY` |
| `year` | INT | NOT NULL | Year integer (e.g., 2024) |
| `quarter` | INT | NOT NULL | Calendar quarter (1-4) |
| `month` | INT | NOT NULL | Month integer (1-12) |
| `month_name` | VARCHAR(20) | NOT NULL | Name of month (January - December) |
| `day_of_month` | INT | NOT NULL | Day index within month (1-31) |
| `day_of_week` | INT | NOT NULL | Weekday index (0=Monday, 6=Sunday) |
| `day_name` | VARCHAR(20) | NOT NULL | Full day name (Monday - Sunday) |
| `is_weekend` | BOOLEAN | NOT NULL | True on Saturday or Sunday |
| `hour` | INT | NOT NULL | Hour of day (0-23), UK local time as recorded by police |
| `time_of_day` | VARCHAR(30) | NOT NULL | Bucket: `Morning Rush` (06-09), `Midday` (10-15), `Evening Rush` (16-19), `Night` |

---

### Table: `dim_location`
Geospatial and jurisdictional attributes for accident occurrence sites.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `location_key` | INT | PRIMARY KEY | Surrogate integer key |
| `latitude` | NUMERIC(10,6) | NOT NULL | WGS84 latitude |
| `longitude` | NUMERIC(10,6) | NOT NULL | WGS84 longitude |
| `urban_or_rural` | VARCHAR(32) | UNIQUE with lat/lon/authority | `Urban`, `Rural`, `Unallocated` |
| `local_authority` | VARCHAR(128) | UNIQUE with lat/lon/urban | ONS local authority district code (e.g. `E09000033`), from `local_authority_ons_district` |
| `region` | VARCHAR(64) | NULL | Nation derived from the ONS code prefix: England, Wales, Scotland |
| `geom` | geometry(Point, 4326) | PostgreSQL only | PostGIS point built from lat/lon (GIST-indexed), see `sql/04_postgis.sql` |

---

### Table: `dim_road`
Road network features, speed regulation, and physical conditions. Unique on all four attributes.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `road_key` | INT | PRIMARY KEY | Surrogate integer key |
| `road_type` | VARCHAR(64) | NOT NULL | Decoded DfT road type (section 4) |
| `speed_limit` | INT | NOT NULL | Posted speed limit in MPH (10-100; DfT `-1` is quarantined) |
| `light_conditions` | VARCHAR(128) | NOT NULL | Decoded DfT light conditions |
| `road_surface_conditions` | VARCHAR(128) | NOT NULL | Decoded DfT road surface conditions |

---

### Table: `dim_weather`
Grain: one Open-Meteo hourly observation for a weather grid cell (`WEATHER_GRID_DEG`, default 1.0°), combined with the police-reported DfT weather condition. Accidents with no matched observation share one `Unavailable` row per reported condition.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `weather_key` | INT | PRIMARY KEY | Surrogate integer key |
| `weather_nk` | VARCHAR(160) | UNIQUE, NOT NULL | Natural key `grid_lat|grid_lon|hour|condition` (or `unavailable|condition`) |
| `grid_latitude` / `grid_longitude` | NUMERIC(9,4) | NULL | Centre of the weather grid cell |
| `observed_hour` | VARCHAR(16) | NULL | Local UK hour of the observation (`YYYY-MM-DDTHH:00`) |
| `weather_condition` | VARCHAR(128) | NOT NULL | Police-reported DfT weather condition (decoded) |
| `weather_description` | VARCHAR(128) | NULL | Open-Meteo WMO weather code description (e.g. `Slight rain`) |
| `weather_code` | INT | NULL | WMO weather code |
| `temperature_c` | NUMERIC(5,2) | NULL | 2 m air temperature (°C) |
| `precipitation_mm` | NUMERIC(5,2) | NULL | Hourly precipitation (mm) |
| `visibility_m` | NUMERIC(8,2) | NULL | Visibility (m); the ERA5-based archive often returns no value |
| `wind_speed_kmh` | NUMERIC(5,2) | NULL | 10 m wind speed (km/h) |
| `weather_risk_level` | VARCHAR(32) | NULL | `Low`, `Moderate`, `Severe` (from measured values, else from the reported condition) |
| `weather_source` | VARCHAR(32) | NULL | `Open-Meteo` or `Unavailable` |

---

### Table: `dim_severity`
Accident severity categories mapped to official DfT definitions.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `severity_key` | INT | PRIMARY KEY | Severity integer key (1, 2, 3) |
| `severity_code` | INT | NOT NULL | 1: Fatal, 2: Serious, 3: Slight |
| `severity_name` | VARCHAR(32) | NOT NULL | 'Fatal', 'Serious', 'Slight' |
| `severity_description` | TEXT | NULL | Formal definition of injury outcome |

---

### Table: `fact_accidents`
Grain: one row per DfT collision.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `accident_key` | INT | PRIMARY KEY | Surrogate fact table primary key |
| `accident_index` | VARCHAR(64) | UNIQUE, NOT NULL | DfT `collision_index` |
| `date_key` | INT | REFERENCES dim_date | Foreign key to `dim_date` |
| `location_key` | INT | REFERENCES dim_location | Foreign key to `dim_location` |
| `road_key` | INT | REFERENCES dim_road | Foreign key to `dim_road` |
| `weather_key` | INT | REFERENCES dim_weather | Foreign key to `dim_weather` |
| `severity_key` | INT | REFERENCES dim_severity | Foreign key to `dim_severity` |
| `number_of_vehicles` | INT | DEFAULT 1 | Vehicles involved |
| `number_of_casualties` | INT | DEFAULT 1 | Casualties (known only after the event) |
| `created_at` | TIMESTAMP | DEFAULT NOW() | Load timestamp |

---

## 2. Staging, Audit, Quarantine & Monitoring Tables

### Table: `stg_accidents_raw`
The selected DfT columns exactly as read from the CSV (codes not yet decoded), plus `batch_id` and `ingested_at`. Rows are deleted per batch after the ETL has loaded or quarantined them.

### Table: `stg_weather_raw`
Open-Meteo hourly readings for the (grid cell, local hour) pairs that match staged accidents.

| Column | Type | Description |
| :--- | :--- | :--- |
| `weather_timestamp` | VARCHAR(16) | Local UK hour `YYYY-MM-DDTHH:00` (API called with `timezone=Europe/London`) |
| `latitude` / `longitude` | NUMERIC | Weather grid cell centre |
| `temperature_2m`, `precipitation`, `wind_speed_10m`, `visibility`, `weather_code` | NUMERIC / INT | Hourly API values |
| `weather_description` | VARCHAR(128) | WMO code description |
| `batch_id` | VARCHAR(64) | Weather ingestion batch |

The complete API responses are kept as the raw landing copy in `data/raw/weather_raw_<batch>.jsonl.gz`.

### Table: `ingestion_audit`
| Column | Type | Description |
| :--- | :--- | :--- |
| `batch_id` | VARCHAR(64) | Unique execution identifier |
| `source_name` | VARCHAR(128) | `UK DfT Road Safety Data (STATS19 collisions)` or `Open-Meteo Historical Weather API` |
| `file_or_endpoint` | VARCHAR(256) | Source file path or API URL (and raw landing file) |
| `status` | VARCHAR(32) | `SUCCESS`, `PARTIAL` (some weather requests failed), or `FAILED` |
| `records_ingested` | INT | Accident rows staged / weather readings staged |
| `records_rejected` | INT | Accidents quarantined by validation / failed weather requests |
| `started_at` / `completed_at` | TIMESTAMP | Run timestamps |
| `error_message` | TEXT | Exception or failed-request summary |

### Table: `quarantine_rejected_records`
| Column | Type | Description |
| :--- | :--- | :--- |
| `rejection_id` | INT | Auto-incrementing primary key |
| `batch_id` | VARCHAR(64) | Originating batch ID |
| `source_entity` | VARCHAR(64) | Staging table name |
| `record_identifier` | VARCHAR(128) | DfT collision index of the offending row |
| `rejection_reason` | VARCHAR(256) | Rule code and explanation (section 3) |
| `raw_payload` | TEXT | Complete JSON snapshot of the rejected staging row |
| `rejected_at` | TIMESTAMP | Timestamp of quarantine event |

### Table: `prediction_log`
One row per prediction served by the FastAPI service.

| Column | Type | Description |
| :--- | :--- | :--- |
| `prediction_id` | INT | Auto-incrementing primary key |
| `accident_index` | VARCHAR(64) | Optional DfT collision index sent by the client; used to measure live false-negative rates once the recorded severity is loaded |
| `request_payload` | TEXT | Full JSON request |
| `predicted_severity_code` | INT | 1 Fatal, 2 Serious, 3 Slight |
| `p_fatal`, `p_serious`, `p_slight` | NUMERIC(9,4) | Class probabilities |
| `model_version` | VARCHAR(64) | Registered MLflow model version, e.g. `accident-severity-classifier/v1` |
| `latency_ms` | NUMERIC(9,4) | Inference latency |
| `predicted_at` | TIMESTAMP | Prediction timestamp |

---

## 3. Data Quality Gate & Validation Rules

Rules run in this order inside `src/etl/validation.py`; a record is quarantined with the first rule it fails. Nothing is silently fixed or imputed.

| Rule ID | Rule Name | Target Column | Validation Condition | Quarantine Reason |
| :--- | :--- | :--- | :--- | :--- |
| **VR-01** | Geographic Coordinates | `latitude`, `longitude` | Present, `49.0 <= latitude <= 61.5` AND `-8.5 <= longitude <= 2.5` | `INVALID_COORDINATES` |
| **VR-03** | Date/Time Integrity | `accident_date`, `accident_time` | Date parses as `DD/MM/YYYY` (or ISO) and time as `HH:MM` | `INVALID_DATETIME` |
| **VR-02** | Valid Speed Limit | `speed_limit` | Present and `10 <= speed_limit <= 100` (DfT `-1` = missing) | `INVALID_SPEED_LIMIT` |
| **VR-04** | Valid Severity Domain | `accident_severity` | Value in `{1, 2, 3}` | `INVALID_SEVERITY` |
| **VR-05** | Primary Key Uniqueness | `accident_index` | Present, not repeated in the batch, not already in `fact_accidents` | `DUPLICATE_ACCIDENT_INDEX` |

Ingestion also skips collisions that were already loaded, staged or quarantined, so a scheduled re-run never duplicates data.

---

## 4. DfT Code Decoding

Source: DfT *Road Safety Open Data Guide*. Any other value, including `-1`, becomes `Data missing or out of range`.

| Column | Codes |
| :--- | :--- |
| `road_type` | 1 Roundabout · 2 One way street · 3 Dual carriageway · 6 Single carriageway · 7 Slip road · 9 Unknown · 12 One way street/Slip road |
| `light_conditions` | 1 Daylight · 4 Darkness - lights lit · 5 Darkness - lights unlit · 6 Darkness - no lighting · 7 Darkness - lighting unknown |
| `weather_conditions` | 1 Fine no high winds · 2 Raining no high winds · 3 Snowing no high winds · 4 Fine + high winds · 5 Raining + high winds · 6 Snowing + high winds · 7 Fog or mist · 8 Other · 9 Unknown |
| `road_surface_conditions` | 1 Dry · 2 Wet or damp · 3 Snow · 4 Frost or ice · 5 Flood over 3cm. deep · 6 Oil or diesel · 7 Mud · 9 Unknown (self reported) |
| `urban_or_rural_area` | 1 Urban · 2 Rural · 3 Unallocated |
| `collision_severity` | 1 Fatal · 2 Serious · 3 Slight |
