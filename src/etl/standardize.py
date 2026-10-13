"""
Standardisation helpers for the UK DfT Road Safety (STATS19) collision data.

DfT publishes categorical fields as numeric codes and dates as DD/MM/YYYY.
These helpers decode the codes into readable labels (per the DfT
"Road Safety Open Data Guide") and parse dates with an explicit format so
day and month are never swapped.
"""

import numpy as np
import pandas as pd

MISSING_LABEL = "Data missing or out of range"

ROAD_TYPE = {
    1: "Roundabout",
    2: "One way street",
    3: "Dual carriageway",
    6: "Single carriageway",
    7: "Slip road",
    9: "Unknown",
    12: "One way street/Slip road",
}

LIGHT_CONDITIONS = {
    1: "Daylight",
    4: "Darkness - lights lit",
    5: "Darkness - lights unlit",
    6: "Darkness - no lighting",
    7: "Darkness - lighting unknown",
}

WEATHER_CONDITIONS = {
    1: "Fine no high winds",
    2: "Raining no high winds",
    3: "Snowing no high winds",
    4: "Fine + high winds",
    5: "Raining + high winds",
    6: "Snowing + high winds",
    7: "Fog or mist",
    8: "Other",
    9: "Unknown",
}

ROAD_SURFACE_CONDITIONS = {
    1: "Dry",
    2: "Wet or damp",
    3: "Snow",
    4: "Frost or ice",
    5: "Flood over 3cm. deep",
    6: "Oil or diesel",
    7: "Mud",
    9: "Unknown (self reported)",
}

URBAN_OR_RURAL = {
    1: "Urban",
    2: "Rural",
    3: "Unallocated",
}

SEVERITY = {1: "Fatal", 2: "Serious", 3: "Slight"}

# ONS district code prefix -> nation
ONS_REGION_PREFIX = {"E": "England", "W": "Wales", "S": "Scotland"}

DATE_FORMATS = ("%d/%m/%Y", "%Y-%m-%d")


def decode_column(values: pd.Series, mapping: dict) -> pd.Series:
    """
    Maps DfT numeric codes to labels. Values that are already text labels
    (e.g. from an older extract) are kept as they are; -1 and blanks become
    the DfT "missing" label.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    decoded = numeric.map(lambda v: mapping.get(int(v)) if pd.notna(v) else None)
    text = values.astype("string").str.strip()
    is_text_label = numeric.isna() & text.notna() & (text != "")
    decoded = decoded.where(~is_text_label, text)
    return decoded.fillna(MISSING_LABEL).astype(str)


def parse_accident_date(dates: pd.Series) -> pd.Series:
    """Parses DD/MM/YYYY (DfT) or ISO YYYY-MM-DD dates. Unparseable values become NaT."""
    text = dates.astype("string").str.strip()
    parsed = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    for fmt in DATE_FORMATS:
        missing = parsed.isna()
        if not missing.any():
            break
        parsed[missing] = pd.to_datetime(text[missing], format=fmt, errors="coerce")
    return parsed


def parse_accident_time(times: pd.Series) -> pd.Series:
    """Parses HH:MM times into a Timedelta. Unparseable values become NaT."""
    text = times.astype("string").str.strip().str.slice(0, 5)
    parsed = pd.to_datetime(text, format="%H:%M", errors="coerce")
    return parsed - parsed.dt.normalize()


def parse_accident_datetime(dates: pd.Series, times: pd.Series) -> pd.Series:
    """Combines date and time; NaT if either part is invalid."""
    return parse_accident_date(dates) + parse_accident_time(times)


def region_from_ons_code(codes: pd.Series) -> pd.Series:
    first = codes.astype("string").str.strip().str.slice(0, 1)
    return first.map(ONS_REGION_PREFIX).fillna("Unknown").astype(str)


def weather_grid(values: pd.Series, grid_deg: float) -> pd.Series:
    """Snaps a coordinate to the weather grid used for Open-Meteo lookups."""
    return (np.round(pd.to_numeric(values, errors="coerce") / grid_deg) * grid_deg).round(4)


def observation_hour(dt: pd.Series) -> pd.Series:
    """Local-time hour key matching Open-Meteo hourly timestamps (YYYY-MM-DDTHH:00)."""
    return dt.dt.floor("h").dt.strftime("%Y-%m-%dT%H:00")
