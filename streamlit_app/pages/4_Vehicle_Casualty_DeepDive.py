"""
Vehicle and Casualty Deep-Dive Page.
Analyzes vehicle collision counts, casualty impact, and severity correlations.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
from src.etl.db import get_engine

st.set_page_config(page_title="Vehicle & Casualty Analysis | Traffic Analytics", layout="wide")

st.title("Vehicle & Casualty Analysis")
st.markdown("Investigate the relationship between multi-vehicle pileups, casualty rates, and final severity outcomes.")


@st.cache_data(ttl=60)
def load_vehicle_data():
    try:
        engine = get_engine()
        query = """
            SELECT 
                f.number_of_vehicles,
                f.number_of_casualties,
                r.road_type,
                r.speed_limit,
                s.severity_name,
                l.urban_or_rural
            FROM fact_accidents f
            JOIN dim_road r ON f.road_key = r.road_key
            JOIN dim_severity s ON f.severity_key = s.severity_key
            JOIN dim_location l ON f.location_key = l.location_key
        """
        df = pd.read_sql(query, con=engine)
        if not df.empty:
            return df
    except Exception:
        pass
    from streamlit_app.demo_data import generate_demo_dataframe
    return generate_demo_dataframe(n=2000)


df = load_vehicle_data()

if df.empty:
    st.warning("No data available.")
else:
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Vehicle Count vs. Severity Outcome")
        veh_summary = df.groupby(["number_of_vehicles", "severity_name"]).size().reset_index(name="Accidents")
        fig_veh = px.bar(
            veh_summary,
            x="number_of_vehicles",
            y="Accidents",
            color="severity_name",
            color_discrete_map={"Fatal": "#EF4444", "Serious": "#F59E0B", "Slight": "#38BDF8"},
            labels={"number_of_vehicles": "Vehicles Involved"}
        )
        st.plotly_chart(fig_veh, width="stretch")
        
    with c2:
        st.subheader("Casualty Count Distribution")
        cas_summary = df.groupby(["number_of_casualties", "severity_name"]).size().reset_index(name="Accidents")
        fig_cas = px.bar(
            cas_summary,
            x="number_of_casualties",
            y="Accidents",
            color="severity_name",
            color_discrete_map={"Fatal": "#EF4444", "Serious": "#F59E0B", "Slight": "#38BDF8"},
            labels={"number_of_casualties": "Total Casualties"}
        )
        st.plotly_chart(fig_cas, width="stretch")
        
    st.divider()
    
    st.subheader("Multi-Vehicle Incidents by Road Classification")
    road_veh = df.groupby(["road_type", "number_of_vehicles"]).size().reset_index(name="Accident Count")
    fig_road_veh = px.sunburst(
        road_veh,
        path=["road_type", "number_of_vehicles"],
        values="Accident Count",
        color="Accident Count",
        color_continuous_scale="Tealgrn"
    )
    st.plotly_chart(fig_road_veh, width="stretch")
