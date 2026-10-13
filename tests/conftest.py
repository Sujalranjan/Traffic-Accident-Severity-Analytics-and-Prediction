"""
Test configuration: every test session runs against a throwaway SQLite warehouse and
data directory, so the real database and data/ folder are never touched.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_TMP = Path(tempfile.mkdtemp(prefix="accident_tests_"))
# Must be set before any src module is imported
os.environ["DATABASE_URL"] = ""
os.environ["SQLITE_FALLBACK_PATH"] = str(_TMP / "test_warehouse.db")
os.environ["DATA_DIR"] = str(_TMP / "data")
os.environ["REPORTS_DIR"] = str(_TMP / "reports")
os.environ["LOG_PREDICTIONS"] = "true"
os.environ.pop("ACCIDENT_BATCH_SIZE", None)

DFT_HEADER = [
    "collision_index", "collision_year", "collision_ref_no", "location_easting_osgr", "location_northing_osgr",
    "longitude", "latitude", "police_force", "collision_severity", "number_of_vehicles", "number_of_casualties",
    "date", "day_of_week", "time", "local_authority_district", "local_authority_ons_district",
    "local_authority_highway", "local_authority_highway_current", "first_road_class", "first_road_number",
    "road_type", "speed_limit", "junction_detail_historic", "junction_detail", "junction_control",
    "second_road_class", "second_road_number", "pedestrian_crossing_human_control_historic",
    "pedestrian_crossing_physical_facilities_historic", "pedestrian_crossing", "light_conditions",
    "weather_conditions", "road_surface_conditions", "special_conditions_at_site", "carriageway_hazards_historic",
    "carriageway_hazards", "urban_or_rural_area", "did_police_officer_attend_scene_of_accident", "trunk_road_flag",
    "lsoa_of_accident_location", "enhanced_severity_collision", "collision_injury_based",
    "collision_adjusted_severity_serious", "collision_adjusted_severity_slight",
]

# (index, date, time, lat, lon, severity, speed, road_type, light, weather, surface, urban)
GOOD_ROWS = [
    ("2023010000001", "02/10/2023", "08:15", 51.5072, -0.1276, 3, 30, 6, 1, 1, 1, 1),
    ("2023010000002", "22/02/2023", "17:40", 53.4808, -2.2426, 2, 40, 3, 4, 2, 2, 1),
    ("2023010000003", "31/12/2023", "23:05", 55.8642, -4.2518, 1, 60, 6, 6, 7, 4, 2),
    ("2024010000004", "15/06/2024", "12:00", 52.4862, -1.8904, 3, 20, 1, 1, 5, 2, 1),
    ("2024010000005", "01/03/2024", "06:30", 53.8008, -1.5491, 3, 70, 3, 1, 3, 3, 2),
]
BAD_ROWS = [
    ("2024010000006", "31/02/2024", "09:00", 51.5, -0.1, 3, 30, 6, 1, 1, 1, 1),   # impossible date
    ("2024010000007", "10/04/2024", "09:00", 51.5, -0.1, 3, -1, 6, 1, 1, 1, 1),   # DfT missing speed
    ("2024010000008", "10/04/2024", "09:00", "", "", 3, 30, 6, 1, 1, 1, 1),       # no coordinates
    ("2024010000001", "10/04/2024", "25:61", 51.5, -0.1, 3, 30, 6, 1, 1, 1, 1),   # bad time
]


@pytest.fixture(scope="session")
def tmp_root() -> Path:
    return _TMP


@pytest.fixture(scope="session")
def dft_csv(tmp_root) -> Path:
    """A small file in the official DfT 44-column layout, with coded values."""
    path = tmp_root / "dft_sample.csv"
    lines = [",".join(DFT_HEADER)]
    for (idx, date, time, lat, lon, sev, speed, road, light, weather, surface, urban) in GOOD_ROWS + BAD_ROWS:
        row = {c: "-1" for c in DFT_HEADER}
        row.update({
            "collision_index": idx, "date": date, "time": time, "latitude": lat, "longitude": lon,
            "collision_severity": sev, "speed_limit": speed, "road_type": road, "light_conditions": light,
            "weather_conditions": weather, "road_surface_conditions": surface, "urban_or_rural_area": urban,
            "number_of_vehicles": 2, "number_of_casualties": 1, "local_authority_ons_district": "E09000033",
        })
        lines.append(",".join(str(row[c]) for c in DFT_HEADER))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
