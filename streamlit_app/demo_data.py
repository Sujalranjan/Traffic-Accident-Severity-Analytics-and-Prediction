"""
Demo data generator for Streamlit Cloud deployment.
Produces realistic synthetic UK traffic accident data
when no live database is available.
"""

import numpy as np
import pandas as pd
import datetime

def generate_demo_dataframe(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    norm_p = lambda arr: np.array(arr, dtype=float) / np.sum(arr)

    cities = [
        {"name": "London",     "lat": 51.5074, "lon": -0.1278,  "authority": "Greater London Authority"},
        {"name": "Birmingham", "lat": 52.4862, "lon": -1.8904,  "authority": "Birmingham City Council"},
        {"name": "Manchester", "lat": 53.4808, "lon": -2.2426,  "authority": "Manchester City Council"},
        {"name": "Leeds",      "lat": 53.8008, "lon": -1.5491,  "authority": "Leeds City Council"},
        {"name": "Glasgow",    "lat": 55.8642, "lon": -4.2518,  "authority": "Glasgow City Council"},
    ]
    city_indices = np.random.choice(len(cities), size=n)
    city_choices = [cities[i] for i in city_indices]

    base = datetime.date(2024, 1, 1)
    dates = [base + datetime.timedelta(days=int(d)) for d in np.random.randint(0, 730, n)]
    
    hours_raw = [
        .015, .010, .008, .007, .010, .020, .050, .080, .090, .060,
        .050, .050, .060, .060, .060, .070, .090, .090, .070, .050,
        .040, .030, .020, .020
    ]
    hours = np.random.choice(range(24), n, p=norm_p(hours_raw))

    severity_names = np.random.choice(["Fatal", "Serious", "Slight"], n, p=norm_p([0.025, 0.185, 0.790]))
    severity_codes = {"Fatal": 1, "Serious": 2, "Slight": 3}

    road_types = np.random.choice(
        ["Single carriageway", "Dual carriageway", "Roundabout", "One way street", "Slip road"],
        n, p=norm_p([0.70, 0.18, 0.06, 0.04, 0.02])
    )
    speed_limits = np.random.choice([20, 30, 40, 50, 60, 70], n, p=norm_p([0.10, 0.60, 0.10, 0.05, 0.08, 0.07]))
    light_conds  = np.random.choice(
        ["Daylight", "Darkness - lights lit", "Darkness - lights unlit", "Darkness - no lighting"],
        n, p=norm_p([0.72, 0.20, 0.03, 0.05])
    )
    road_surfs   = np.random.choice(
        ["Dry", "Wet or damp", "Frost or ice", "Snow", "Flood over 3cm. deep"],
        n, p=norm_p([0.68, 0.26, 0.04, 0.015, 0.005])
    )
    weather_conds = np.random.choice(
        ["Fine no high winds", "Raining no high winds", "Raining + high winds", "Fog or mist", "Snowing no high winds"],
        n, p=norm_p([0.75, 0.16, 0.04, 0.03, 0.02])
    )
    weather_risk  = [
        "Severe" if "Snow" in w or "Fog" in w else ("Moderate" if "Rain" in w else "Low")
        for w in weather_conds
    ]
    urban_rural   = np.random.choice(["Urban", "Rural"], n, p=norm_p([0.65, 0.35]))
    vehicles      = np.random.choice([1, 2, 3, 4], n, p=norm_p([0.35, 0.55, 0.08, 0.02]))
    casualties    = np.random.choice([1, 2, 3, 4], n, p=norm_p([0.78, 0.16, 0.04, 0.02]))

    tod_map = lambda h: (
        "Morning Rush" if 6  <= h < 10 else
        "Midday"       if 10 <= h < 16 else
        "Evening Rush" if 16 <= h < 20 else "Night"
    )

    temp_map = {
        "Fine no high winds":    14.5,
        "Raining no high winds":  8.0,
        "Raining + high winds":   7.5,
        "Fog or mist":           10.0,
        "Snowing no high winds":  1.5,
    }
    precip_map = {
        "Fine no high winds":    0.0,
        "Raining no high winds": 3.5,
        "Raining + high winds":  6.0,
        "Fog or mist":           0.5,
        "Snowing no high winds": 1.0,
    }

    df = pd.DataFrame({
        "accident_index":        [f"DEMO_{i:05d}" for i in range(n)],
        "full_date":             [d.strftime("%Y-%m-%d") for d in dates],
        "year":                  [d.year for d in dates],
        "month":                 [d.month for d in dates],
        "month_name":            [d.strftime("%B") for d in dates],
        "day_name":              [d.strftime("%A") for d in dates],
        "day_of_week":           [d.weekday() for d in dates],
        "hour":                  hours,
        "time_of_day":           [tod_map(h) for h in hours],
        "is_weekend":            [d.weekday() >= 5 for d in dates],
        "latitude":              [c["lat"] + np.random.normal(0, 0.05) for c in city_choices],
        "longitude":             [c["lon"] + np.random.normal(0, 0.05) for c in city_choices],
        "urban_or_rural":        urban_rural,
        "local_authority":       [c["authority"] for c in city_choices],
        "road_type":             road_types,
        "speed_limit":           speed_limits,
        "light_conditions":      light_conds,
        "road_surface_conditions": road_surfs,
        "weather_condition":     weather_conds,
        "temperature_c":         [temp_map[w]   + np.random.normal(0, 1.5) for w in weather_conds],
        "precipitation_mm":      [max(0, precip_map[w] + np.random.normal(0, 0.5)) for w in weather_conds],
        "visibility_m":          [10000 if "Fine" in w else (4000 if "Rain" in w else 800) for w in weather_conds],
        "wind_speed_kmh":        [12.0 if "Fine" in w else (28.0 if "high winds" in w else 16.0) for w in weather_conds],
        "weather_risk_level":    weather_risk,
        "number_of_vehicles":    vehicles,
        "number_of_casualties":  casualties,
        "severity_name":         severity_names,
        "severity_code":         [severity_codes[s] for s in severity_names],
    })
    return df
