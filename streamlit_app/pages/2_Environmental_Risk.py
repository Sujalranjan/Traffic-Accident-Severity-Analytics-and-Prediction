"""
Environmental & Road Condition Impact Page.
Examines the intersection of weather variables, road surface, and speed limits on accident severity.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
from src.etl.db import get_engine

st.set_page_config(page_title="Environmental Risk | Traffic Analytics", layout="wide")

st.title("Environmental & Road Surface Analysis")
st.markdown("Examine how adverse weather conditions, lighting, and road surfaces correlate with accident severity.")


@st.cache_data(ttl=60)
def load_environmental_data():
    try:
        engine = get_engine()
        query = """
            SELECT 
                w.weather_condition,
                w.weather_risk_level,
                w.temperature_c,
                w.precipitation_mm,
                w.visibility_m,
                r.road_type,
                r.speed_limit,
                r.light_conditions,
                r.road_surface_conditions,
                s.severity_name,
                f.number_of_casualties
            FROM fact_accidents f
            JOIN dim_road r ON f.road_key = r.road_key
            JOIN dim_weather w ON f.weather_key = w.weather_key
            JOIN dim_severity s ON f.severity_key = s.severity_key
        """
        df = pd.read_sql(query, con=engine)
        if not df.empty:
            return df
    except Exception:
        pass
    from streamlit_app.demo_data import generate_demo_dataframe
    return generate_demo_dataframe(n=2000)


df = load_environmental_data()

if df.empty:
    st.warning("No data available.")
else:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Incident Volume by Weather Condition")
        wea_summary = df.groupby(["weather_condition", "severity_name"]).size().reset_index(name="Accidents")
        fig_wea = px.bar(
            wea_summary,
            x="weather_condition",
            y="Accidents",
            color="severity_name",
            color_discrete_map={"Fatal": "#EF4444", "Serious": "#F59E0B", "Slight": "#38BDF8"},
            labels={"weather_condition": "Atmospheric Condition"}
        )
        fig_wea.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_wea, width="stretch")
        
    with col2:
        st.subheader("Road Surface Conditions vs. Severity")
        surface_summary = df.groupby(["road_surface_conditions", "severity_name"]).size().reset_index(name="Accidents")
        fig_surf = px.bar(
            surface_summary,
            x="road_surface_conditions",
            y="Accidents",
            color="severity_name",
            color_discrete_map={"Fatal": "#EF4444", "Serious": "#F59E0B", "Slight": "#38BDF8"},
            labels={"road_surface_conditions": "Road Surface State"}
        )
        fig_surf.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_surf, width="stretch")
        
    st.divider()
    
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Speed Limit vs. High Severity Rate")
        speed_sev = df.groupby("speed_limit")["severity_name"].value_counts(normalize=True).unstack().fillna(0)
        if "Fatal" in speed_sev.columns:
            speed_sev["Fatal_Pct"] = speed_sev["Fatal"] * 100
        else:
            speed_sev["Fatal_Pct"] = 0
            
        if "Serious" in speed_sev.columns:
            speed_sev["Serious_Pct"] = speed_sev["Serious"] * 100
        else:
            speed_sev["Serious_Pct"] = 0
            
        speed_plot_df = speed_sev[["Fatal_Pct", "Serious_Pct"]].reset_index()
        fig_speed = px.line(
            speed_plot_df,
            x="speed_limit",
            y=["Fatal_Pct", "Serious_Pct"],
            markers=True,
            labels={"value": "Percentage (%)", "speed_limit": "Speed Limit (MPH)", "variable": "Risk Metric"},
            title="Fatal & Serious Injury Rates Across Speed Limits"
        )
        st.plotly_chart(fig_speed, width="stretch")
        
    with col4:
        st.subheader("Lighting Conditions Impact")
        light_summary = df.groupby(["light_conditions", "severity_name"]).size().reset_index(name="Accidents")
        fig_light = px.bar(
            light_summary,
            x="light_conditions",
            y="Accidents",
            color="severity_name",
            color_discrete_map={"Fatal": "#EF4444", "Serious": "#F59E0B", "Slight": "#38BDF8"},
            labels={"light_conditions": "Illumination"}
        )
        fig_light.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_light, width="stretch")
