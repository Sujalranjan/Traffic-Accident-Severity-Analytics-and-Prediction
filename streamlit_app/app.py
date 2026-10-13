"""
Streamlit Application - Traffic Accident Severity Analytics & Prediction Platform
Main Portal & Executive KPI Dashboard
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.etl.db import get_engine

st.set_page_config(
    page_title="Traffic Accident Severity Analytics & MLOps",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        color: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border: 1px solid #334155;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def load_data():
    try:
        engine = get_engine()
        query = """
            SELECT 
                f.accident_index,
                d.full_date,
                d.year,
                d.month,
                d.day_name,
                d.hour,
                d.time_of_day,
                d.is_weekend,
                l.latitude,
                l.longitude,
                l.urban_or_rural,
                l.local_authority,
                r.road_type,
                r.speed_limit,
                r.light_conditions,
                r.road_surface_conditions,
                w.weather_condition,
                w.temperature_c,
                w.precipitation_mm,
                w.visibility_m,
                w.wind_speed_kmh,
                w.weather_risk_level,
                f.number_of_vehicles,
                f.number_of_casualties,
                s.severity_name,
                s.severity_code
            FROM fact_accidents f
            JOIN dim_date d ON f.date_key = d.date_key
            JOIN dim_location l ON f.location_key = l.location_key
            JOIN dim_road r ON f.road_key = r.road_key
            JOIN dim_weather w ON f.weather_key = w.weather_key
            JOIN dim_severity s ON f.severity_key = s.severity_key
        """
        df = pd.read_sql(query, con=engine)
        if not df.empty:
            return df, False  # (data, is_demo)
    except Exception:
        pass

    # Fallback: generate demo data for cloud deployment
    from streamlit_app.demo_data import generate_demo_dataframe
    return generate_demo_dataframe(n=2000), True


df, IS_DEMO = load_data()

st.markdown('<div class="main-title">Traffic Accident Severity Analytics Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Data Engineering, Geospatial Analytics, and Predictive MLOps</div>', unsafe_allow_html=True)

if IS_DEMO:
    st.info("Demonstration Mode: Viewing benchmark telemetry data. In the full production environment, the pipeline connects to the containerized PostgreSQL star schema and MLflow tracking service.")

# Sidebar Filters
st.sidebar.title("Filters")
if not df.empty:
    selected_urban = st.sidebar.multiselect("Area Type", options=sorted(df["urban_or_rural"].unique()), default=list(df["urban_or_rural"].unique()))
    selected_severities = st.sidebar.multiselect("Severity Level", options=sorted(df["severity_name"].unique()), default=list(df["severity_name"].unique()))
    
    # Filter dataset
    filtered_df = df[
        (df["urban_or_rural"].isin(selected_urban)) &
        (df["severity_name"].isin(selected_severities))
    ]
else:
    filtered_df = pd.DataFrame()

# KPI Row
if not filtered_df.empty:
    total_accidents = len(filtered_df)
    fatal_count = len(filtered_df[filtered_df["severity_name"] == "Fatal"])
    serious_count = len(filtered_df[filtered_df["severity_name"] == "Serious"])
    slight_count = len(filtered_df[filtered_df["severity_name"] == "Slight"])
    total_casualties = filtered_df["number_of_casualties"].sum()
    avg_vehicles = round(filtered_df["number_of_vehicles"].mean(), 2)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Total Incidents</div><div class="metric-val">{total_accidents:,}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Fatalities</div><div class="metric-val" style="color:#EF4444;">{fatal_count:,}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Serious Injuries</div><div class="metric-val" style="color:#F59E0B;">{serious_count:,}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Total Casualties</div><div class="metric-val" style="color:#10B981;">{total_casualties:,}</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card"><div class="metric-lbl">Avg Vehicles / Incident</div><div class="metric-val" style="color:#A855F7;">{avg_vehicles}</div></div>', unsafe_allow_html=True)

    st.write("")
    
    # Main Visuals: Severity Distribution & Area Breakdown
    c_left, c_right = st.columns([1.2, 1])
    
    with c_left:
        st.subheader("Severity Proportion & Class Distribution")
        sev_counts = filtered_df["severity_name"].value_counts().reset_index()
        sev_counts.columns = ["Severity", "Count"]
        color_map = {"Fatal": "#EF4444", "Serious": "#F59E0B", "Slight": "#38BDF8"}
        
        fig_donut = px.pie(
            sev_counts, 
            values="Count", 
            names="Severity", 
            hole=0.55,
            color="Severity",
            color_discrete_map=color_map
        )
        fig_donut.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#0F172A', width=2)))
        fig_donut.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with c_right:
        st.subheader("Severity Distribution by Area Type")
        urban_sev = filtered_df.groupby(["urban_or_rural", "severity_name"]).size().reset_index(name="Accidents")
        fig_bar = px.bar(
            urban_sev,
            x="urban_or_rural",
            y="Accidents",
            color="severity_name",
            barmode="group",
            color_discrete_map=color_map,
            labels={"urban_or_rural": "Area Type", "Accidents": "Recorded Accidents"}
        )
        fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    # Architectural Overview Section
    st.divider()
    st.subheader("System Architecture & Data Pipeline")
    st.markdown("""
    This platform operates a dual-tier architecture:
    1. **Data Engineering Layer**:
       - **Automated Ingestion**: Incremental loads of UK DfT Road Safety (STATS19) collisions, joined to hourly weather from the **Open-Meteo API** by grid cell and local hour.
       - **Quarantine & Validation**: Schema checks quarantine invalid coordinates or corrupted fields with audit logging prior to loading.
       - **Dimensional Star Schema**: Relational warehouse in **PostgreSQL / PostGIS** with dimensional models (`dim_date`, `dim_location`, `dim_road`, `dim_weather`, `dim_severity`) and aggregate marts.
       - **Orchestration**: Managed end-to-end via **Apache Airflow DAGs**.
    2. **MLOps Predictive Layer**:
       - **Class Imbalance Handling**: SMOTE plus class weights for rare fatal collisions (about 1.5% of records).
       - **Experiment Tracking & Registry**: **MLflow** tracks parameters, macro-F1, recall and confusion matrices; the champion is registered with a `champion` alias.
       - **Model Serving**: REST API endpoints implemented with **FastAPI**, with every prediction logged for monitoring.
       - **Monitoring**: PSI / KS drift, class-mix shift and Fatal/Serious false-negative rates, with automatic retraining triggers.
    """)
    
    st.caption("Use the sidebar pages to inspect Temporal Dynamics, Environmental Risk, Geospatial Hotspots, Vehicle/Casualty Analysis, Live Inference, and Pipeline Audit/Drift Monitoring.")
else:
    st.warning("No data found in warehouse. Please run the ETL ingestion pipeline first (`python src/etl/transform.py`).")

