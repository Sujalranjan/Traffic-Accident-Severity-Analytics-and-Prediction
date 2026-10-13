"""
Data, Location, Class and Model-Performance Monitoring.

Compares the oldest 70% of the warehouse (reference, the period the model was trained on)
with the newest 30% (current):
  * feature drift      - PSI and two-sample KS on weather and speed features
  * location drift     - two-sample KS on latitude / longitude
  * class drift        - chi-square test on the severity mix
  * model performance  - champion model scored on the current window: macro-F1 and the
                         false-negative rates (FNR) for Fatal and Serious collisions
  * live predictions   - FNR of logged API predictions whose accident_index has a known outcome
and turns the results into a retraining decision.
"""

import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import os
import logging
import datetime
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score
from src.etl.db import PROJECT_ROOT, get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("drift_monitoring")

REPORTS_DIR = Path(os.getenv("REPORTS_DIR", str(PROJECT_ROOT / "reports")))
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(PROJECT_ROOT / "models")))

# Retraining thresholds
PSI_CRITICAL = 0.25
PSI_MODERATE = 0.10
LOCATION_KS = 0.15
SIGNIFICANCE = 0.01
CLASS_SHIFT_PP = 0.02        # absolute change in any class share
MACRO_F1_DROP = 0.10         # relative drop vs. the registered baseline
FNR_INCREASE = 0.10          # absolute increase in Fatal or Serious FNR vs. baseline
MIN_LIVE_LABELLED = 30       # labelled live predictions needed before live FNR counts

SEVERITY_CODE_TO_CLASS = {3: 0, 2: 1, 1: 2}   # model classes: 0 Slight, 1 Serious, 2 Fatal


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_bins: int = 10) -> float:
    """Calculates the Population Stability Index (PSI) between two distributions."""
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.unique(np.percentile(expected, quantiles))
    if len(bin_edges) < 2:
        return 0.0
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    exp_counts, _ = np.histogram(expected, bins=bin_edges)
    act_counts, _ = np.histogram(actual, bins=bin_edges)

    # Avoid zero division with smoothing
    exp_pct = np.where(exp_counts == 0, 1e-4, exp_counts) / len(expected)
    act_pct = np.where(act_counts == 0, 1e-4, act_counts) / len(actual)

    psi_value = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(np.round(psi_value, 4))


def fnr(y_true: np.ndarray, y_pred: np.ndarray, cls: int):
    positives = np.sum(y_true == cls)
    if positives == 0:
        return None
    return round(float(np.sum((y_true == cls) & (y_pred != cls)) / positives), 4)


def load_baseline() -> dict:
    try:
        with open(MODEL_DIR / "model_metadata.json", encoding="utf-8") as f:
            meta = json.load(f)
        return {"model_version": meta.get("model_version"), **meta.get("baseline_test_metrics", {})}
    except (OSError, ValueError):
        return {}


def class_drift(ref: pd.Series, cur: pd.Series) -> dict:
    labels = ["Fatal", "Serious", "Slight"]
    ref_counts = ref.value_counts().reindex(labels, fill_value=0)
    cur_counts = cur.value_counts().reindex(labels, fill_value=0)
    chi2, p_val, _, _ = stats.chi2_contingency(np.vstack([ref_counts.values, cur_counts.values]) + 1)
    ref_share = (ref_counts / ref_counts.sum()).round(4)
    cur_share = (cur_counts / cur_counts.sum()).round(4)
    max_shift = float((cur_share - ref_share).abs().max())
    return {
        "reference": ref_share.to_dict(),
        "current": cur_share.to_dict(),
        "chi2": round(float(chi2), 4),
        "p_value": round(float(p_val), 4),
        "max_share_change": round(max_shift, 4),
        "drift_detected": bool(p_val < SIGNIFICANCE and max_shift > CLASS_SHIFT_PP),
    }


def model_performance(cur_df: pd.DataFrame, baseline: dict) -> dict:
    """Scores the champion model on the current window (backtest against recorded severities)."""
    import joblib
    from src.ml.features import engineer_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES

    model_path = MODEL_DIR / "accident_severity_model.joblib"
    if not model_path.exists():
        return {"evaluated": False, "reason": "model artifact not found"}
    try:
        model = joblib.load(model_path)
        feats = engineer_features(cur_df)
        y_true = feats["severity_code"].map(SEVERITY_CODE_TO_CLASS).values
        y_pred = model.predict(feats[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    except Exception as e:
        logger.warning(f"Model performance check skipped: {e}")
        return {"evaluated": False, "reason": f"{e.__class__.__name__}: {e}"}

    result = {
        "evaluated": True,
        "records": int(len(y_true)),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "fatal_fnr": fnr(y_true, y_pred, 2),
        "serious_fnr": fnr(y_true, y_pred, 1),
        "baseline": baseline,
    }
    return result


def live_prediction_fnr(engine) -> dict:
    """FNR on API predictions whose accident_index now has a recorded severity in the warehouse."""
    try:
        logged = pd.read_sql(
            """
            SELECT p.predicted_severity_code, s.severity_code
            FROM prediction_log p
            JOIN fact_accidents f ON p.accident_index = f.accident_index
            JOIN dim_severity s ON f.severity_key = s.severity_key
            """,
            con=engine,
        )
        total_logged = int(pd.read_sql("SELECT COUNT(*) AS n FROM prediction_log", con=engine)["n"].iloc[0])
    except Exception as e:
        return {"labelled_predictions": 0, "reason": f"{e.__class__.__name__}"}

    out = {"logged_predictions": total_logged, "labelled_predictions": int(len(logged))}
    if len(logged) >= MIN_LIVE_LABELLED:
        y_true = logged["severity_code"].map(SEVERITY_CODE_TO_CLASS).values
        y_pred = logged["predicted_severity_code"].map(SEVERITY_CODE_TO_CLASS).values
        out.update({"fatal_fnr": fnr(y_true, y_pred, 2), "serious_fnr": fnr(y_true, y_pred, 1)})
    return out


def run_drift_analysis() -> dict:
    """Executes full monitoring analysis and writes reports/drift_report_<date>.json."""
    from src.ml.features import load_dataset_from_warehouse

    engine = get_engine()
    df = load_dataset_from_warehouse()
    if len(df) < 40:
        logger.warning("Insufficient data for drift analysis.")
        return {}

    df["_sort"] = pd.to_datetime(df["full_date"].astype(str).str.slice(0, 10), format="%Y-%m-%d") \
        + pd.to_timedelta(df["hour"], unit="h")
    df = df.sort_values("_sort").reset_index(drop=True)

    split_point = int(len(df) * 0.70)
    ref_df = df.iloc[:split_point]
    cur_df = df.iloc[split_point:]
    logger.info(f"Analyzing drift - Reference: {len(ref_df)} records, Current: {len(cur_df)} records")

    # 1. Feature drift (PSI & KS)
    feature_drift_results = {}
    for feat in ["temperature_c", "precipitation_mm", "wind_speed_kmh", "speed_limit"]:
        ref_vals = pd.to_numeric(ref_df[feat], errors="coerce").dropna().values.astype(float)
        cur_vals = pd.to_numeric(cur_df[feat], errors="coerce").dropna().values.astype(float)
        if len(ref_vals) < 10 or len(cur_vals) < 10:
            feature_drift_results[feat] = {"status": "INSUFFICIENT_DATA", "drift_detected": False,
                                           "non_null_reference": int(len(ref_vals)), "non_null_current": int(len(cur_vals))}
            continue
        ks_stat, p_val = stats.ks_2samp(ref_vals, cur_vals)
        psi = calculate_psi(ref_vals, cur_vals)
        status = "CRITICAL_DRIFT" if psi > PSI_CRITICAL else ("MODERATE_DRIFT" if psi > PSI_MODERATE else "STABLE")
        feature_drift_results[feat] = {
            "ks_statistic": round(float(ks_stat), 4),
            "p_value": round(float(p_val), 4),
            "psi": psi,
            "status": status,
            "drift_detected": bool(psi > PSI_CRITICAL or p_val < SIGNIFICANCE),
        }

    # 2. Location drift
    lat_ks, lat_p = stats.ks_2samp(ref_df["latitude"].astype(float), cur_df["latitude"].astype(float))
    lon_ks, lon_p = stats.ks_2samp(ref_df["longitude"].astype(float), cur_df["longitude"].astype(float))
    location_drift_detected = bool((lat_p < SIGNIFICANCE and lat_ks > LOCATION_KS) or (lon_p < SIGNIFICANCE and lon_ks > LOCATION_KS))

    # 3. Class drift
    class_result = class_drift(ref_df["severity_name"], cur_df["severity_name"])

    # 4. Model performance and false-negative rates
    baseline = load_baseline()
    performance = model_performance(cur_df, baseline)
    live = live_prediction_fnr(engine)

    # 5. Retraining decision
    reasons = []
    critical = [f for f, v in feature_drift_results.items() if v.get("status") == "CRITICAL_DRIFT"]
    if critical:
        reasons.append(f"Critical PSI (> {PSI_CRITICAL}) on: {', '.join(critical)}")
    if location_drift_detected:
        reasons.append(f"Significant spatial shift (KS > {LOCATION_KS}, p < {SIGNIFICANCE})")
    if class_result["drift_detected"]:
        reasons.append(f"Severity mix changed by {class_result['max_share_change']:.1%} (chi-square p < {SIGNIFICANCE})")
    if performance.get("evaluated") and baseline:
        base_f1 = baseline.get("macro_f1")
        if base_f1 and performance["macro_f1"] < base_f1 * (1 - MACRO_F1_DROP):
            reasons.append(f"Macro-F1 fell from {base_f1:.3f} to {performance['macro_f1']:.3f} (> {MACRO_F1_DROP:.0%} drop)")
        for cls in ["fatal", "serious"]:
            base, now = baseline.get(f"{cls}_fnr"), performance.get(f"{cls}_fnr")
            if base is not None and now is not None and now > base + FNR_INCREASE:
                reasons.append(f"{cls.title()} false-negative rate rose from {base:.3f} to {now:.3f}")
    for cls in ["fatal", "serious"]:
        live_fnr, base = live.get(f"{cls}_fnr"), baseline.get(f"{cls}_fnr")
        if live_fnr is not None and base is not None and live_fnr > base + FNR_INCREASE:
            reasons.append(f"Live {cls.title()} FNR {live_fnr:.3f} exceeds baseline {base:.3f}")

    retrain_needed = len(reasons) > 0

    report = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "reference_records": len(ref_df),
        "current_records": len(cur_df),
        "reference_period": [str(ref_df["_sort"].min().date()), str(ref_df["_sort"].max().date())],
        "current_period": [str(cur_df["_sort"].min().date()), str(cur_df["_sort"].max().date())],
        "feature_drift": feature_drift_results,
        "location_drift": {
            "latitude_ks": round(float(lat_ks), 4),
            "latitude_p_val": round(float(lat_p), 4),
            "longitude_ks": round(float(lon_ks), 4),
            "longitude_p_val": round(float(lon_p), 4),
            "drift_detected": location_drift_detected,
        },
        "class_distribution": class_result,
        "model_performance": performance,
        "live_predictions": live,
        "thresholds": {
            "psi_critical": PSI_CRITICAL, "location_ks": LOCATION_KS, "significance": SIGNIFICANCE,
            "class_shift": CLASS_SHIFT_PP, "macro_f1_relative_drop": MACRO_F1_DROP, "fnr_increase": FNR_INCREASE,
        },
        "retraining_decision": {
            "retrain_triggered": retrain_needed,
            "triggers": reasons if retrain_needed else ["All metrics within tolerance stability thresholds"],
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_file = REPORTS_DIR / f"drift_report_{datetime.date.today()}.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Drift evaluation report written to: {report_file}")
    logger.info(f"Retraining Trigger Status: {'TRIGGERED' if retrain_needed else 'OK (STABLE)'}")
    return report


if __name__ == "__main__":
    run_drift_analysis()
