"""
Interactive Real-Time Severity Prediction Page.
Scores user-entered collision conditions with the registered champion model pipeline.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Live Prediction | MLOps", layout="wide")

st.title("Accident Severity Prediction & Inference")
st.markdown("Simulate crash conditions and compute instant severity classification & risk scores using the operationalized model.")

MODEL_PATH = Path("models/accident_severity_model.joblib")
REPORT_PATH = Path("models/test_evaluation_report.json")


@st.cache_resource
def load_production_pipeline():
    """Loads the registered champion pipeline; returns (pipeline, error message)."""
    if not MODEL_PATH.exists():
        return None, f"Model artifact not found at {MODEL_PATH}. Train it with `python src/ml/train.py`."
    try:
        return joblib.load(MODEL_PATH), None
    except Exception as e:
        return None, f"Model artifact could not be loaded ({e.__class__.__name__}: {e})."


@st.cache_data
def load_model_metrics():
    if REPORT_PATH.exists():
        import json
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


model_pipeline, model_error = load_production_pipeline()
model_metrics = load_model_metrics()

# Metrics Banner
m1, m2, m3, m4 = st.columns(4)
overall_acc = model_metrics.get("overall_accuracy", model_metrics.get("accuracy", 0.0))
macro_f1 = model_metrics.get("macro avg", {}).get("f1-score", 0.0)

with m1:
    st.metric("Overall Model Accuracy", f"{overall_acc * 100:.1f}%", help="Percentage of correctly classified test incidents across all severity classes.")
with m2:
    st.metric("Balanced Macro-F1", f"{macro_f1:.2f}", help="Unweighted harmonic mean of precision and recall giving equal evaluation weight to fatal and serious categories.")
with m3:
    fatal_recall = 1 - model_metrics["fatal_fnr"] if "fatal_fnr" in model_metrics else model_metrics.get("Fatal", {}).get("recall", 0.0)
    st.metric("Fatal Recall (test)", f"{fatal_recall * 100:.1f}%", help="Share of fatal collisions in the hold-out test period that the model flags as Fatal (1 - false-negative rate). Imbalance handling: SMOTE + class weights.")
with m4:
    champion_name = model_metrics.get("champion_model", "Unknown")
    st.metric("Production Champion", champion_name, help="Best validation macro-F1 model, registered in the MLflow Model Registry with the 'champion' alias.")

st.divider()

if model_pipeline is None:
    st.error(model_error)
else:
    col_input, col_pred = st.columns([1.1, 1.2])
    
    with col_input:
        st.subheader("Scenario Input Parameters")
        
        c1, c2 = st.columns(2)
        with c1:
            hour = st.slider("Hour of Day", 0, 23, 18)
            day_name = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=4)
            urban_rural = st.selectbox("Area Type", ["Urban", "Rural"], index=0)
            road_type = st.selectbox("Road Classification", [
                "Single carriageway", "Dual carriageway", "Roundabout", "One way street", "Slip road",
                "One way street/Slip road", "Unknown"
            ], index=0)
            speed_limit = st.select_slider("Speed Limit (MPH)", options=[20, 30, 40, 50, 60, 70], value=30)
            
        with c2:
            weather_condition = st.selectbox("Weather Condition", [
                "Fine no high winds", "Raining no high winds", "Snowing no high winds", "Fine + high winds",
                "Raining + high winds", "Snowing + high winds", "Fog or mist", "Other", "Unknown"
            ], index=1)
            road_surface = st.selectbox("Road Surface Condition", [
                "Dry", "Wet or damp", "Snow", "Frost or ice", "Flood over 3cm. deep", "Oil or diesel", "Mud"
            ], index=1)
            light_conditions = st.selectbox("Light Conditions", [
                "Daylight", "Darkness - lights lit", "Darkness - lights unlit", "Darkness - no lighting",
                "Darkness - lighting unknown"
            ], index=1)
            num_vehicles = st.number_input("Vehicles Involved", min_value=1, max_value=10, value=2)
            num_casualties = st.number_input("Casualties Involved", min_value=1, max_value=10, value=1)
            
        st.divider()
        with st.expander("Atmospheric & Weather Variables"):
            temp = st.slider("Temperature (°C)", -10.0, 35.0, 8.5)
            precip = st.slider("Precipitation (mm)", 0.0, 20.0, 2.5)
            vis = st.slider("Visibility (meters)", 100.0, 20000.0, 4500.0)
            wind = st.slider("Wind Speed (km/h)", 0.0, 80.0, 25.0)

    # Inference logic
    with col_pred:
        st.subheader("Real-Time Prediction Outcome")
        
        # Determine time of day
        if 6 <= hour < 10:
            time_of_day = "Morning Rush"
        elif 10 <= hour < 16:
            time_of_day = "Midday"
        elif 16 <= hour < 20:
            time_of_day = "Evening Rush"
        else:
            time_of_day = "Night"
            
        # Format payload
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        
        input_data = pd.DataFrame([{
            "hour": hour,
            "speed_limit": speed_limit,
            "temperature_c": temp,
            "precipitation_mm": precip,
            "visibility_m": vis,
            "wind_speed_kmh": wind,
            "number_of_vehicles": num_vehicles,
            "number_of_casualties": num_casualties,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_name": day_name,
            "time_of_day": time_of_day,
            "urban_or_rural": urban_rural,
            "road_type": road_type,
            "light_conditions": light_conditions,
            "road_surface_conditions": road_surface,
            "weather_condition": weather_condition
        }])
        
        pred_idx = int(model_pipeline.predict(input_data)[0])
        probas = model_pipeline.predict_proba(input_data)[0]
        
        class_labels = {0: "Slight", 1: "Serious", 2: "Fatal"}
        badge_colors = {"Fatal": "#EF4444", "Serious": "#F59E0B", "Slight": "#10B981"}
        
        predicted_label = class_labels[pred_idx]
        badge_color = badge_colors[predicted_label]
        
        # Composite risk index: 0.1*P_slight + 0.5*P_serious + 1.0*P_fatal
        composite_risk = float(0.1 * probas[0] + 0.5 * probas[1] + 1.0 * probas[2])
        
        # Metric card
        st.markdown(f"""
        <div style="background-color: #0F172A; border-radius: 12px; padding: 1.5rem; border-left: 8px solid {badge_color}; margin-bottom: 1rem;">
            <div style="color: #94A3B8; font-size: 0.9rem; text-transform: uppercase;">Predicted Severity Classification</div>
            <div style="color: {badge_color}; font-size: 2.2rem; font-weight: 800; margin: 0.2rem 0;">{predicted_label.upper()}</div>
            <div style="color: #CBD5E1; font-size: 0.95rem;">Composite Risk Index: <b>{composite_risk:.2f} / 1.00</b></div>
        </div>
        """, unsafe_allow_html=True)
        
        # Probability Bar Chart
        prob_df = pd.DataFrame({
            "Severity Class": ["Slight", "Serious", "Fatal"],
            "Probability (%)": [probas[0] * 100, probas[1] * 100, probas[2] * 100]
        })
        
        fig_prob = px.bar(
            prob_df,
            x="Severity Class",
            y="Probability (%)",
            color="Severity Class",
            color_discrete_map={"Fatal": "#EF4444", "Serious": "#F59E0B", "Slight": "#38BDF8"},
            text=prob_df["Probability (%)"].apply(lambda v: f"{v:.1f}%")
        )
        fig_prob.update_layout(yaxis=dict(range=[0, 100]), height=280, showlegend=False)
        st.plotly_chart(fig_prob, width="stretch")
        
        # Model Details
        model_type = type(model_pipeline.named_steps["classifier"]).__name__
        st.caption(f"Operational Model: `{model_type}` | Imbalance Handling: SMOTE + class weights")
