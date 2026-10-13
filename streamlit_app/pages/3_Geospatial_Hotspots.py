"""
Geospatial Hotspots & Spatial Density Page.
Renders interactive PyDeck map layers and spatial clustering for accident high-density zones.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px
from src.etl.db import get_engine

st.set_page_config(page_title="Geospatial Hotspots | Traffic Analytics", layout="wide")

st.title("Geospatial Hotspots & Regional Density Analysis")
st.markdown("Locate high-frequency accident zones and explore spatial severity distributions across regional clusters.")


@st.cache_data(ttl=60)
def load_geospatial_data():
    try:
        engine = get_engine()
        query = """
            SELECT 
                l.latitude,
                l.longitude,
                l.urban_or_rural,
                l.local_authority,
                s.severity_name,
                f.number_of_casualties,
                f.number_of_vehicles
            FROM fact_accidents f
            JOIN dim_location l ON f.location_key = l.location_key
            JOIN dim_severity s ON f.severity_key = s.severity_key
            WHERE l.latitude IS NOT NULL AND l.longitude IS NOT NULL
        """
        df = pd.read_sql(query, con=engine)
        if not df.empty:
            return df
    except Exception:
        pass
    from streamlit_app.demo_data import generate_demo_dataframe
    return generate_demo_dataframe(n=2000)


df = load_geospatial_data()

if df.empty:
    st.warning("No geospatial coordinates found.")
else:
    # Color mapping for map points
    color_map = {
        "Fatal": [239, 68, 68, 200],      # Red
        "Serious": [245, 158, 11, 180],   # Orange
        "Slight": [56, 189, 248, 140]     # Blue
    }
    df["color"] = df["severity_name"].map(color_map)
    
    # Sidebar map filters
    st.sidebar.subheader("Map Controls")
    selected_sev = st.sidebar.multiselect("Filter Severity", options=["Fatal", "Serious", "Slight"], default=["Fatal", "Serious", "Slight"])
    plot_df = df[df["severity_name"].isin(selected_sev)]
    # The full DfT dataset has hundreds of thousands of points; draw a sample on the map
    MAX_MAP_POINTS = 60000
    map_df = plot_df.sample(MAX_MAP_POINTS, random_state=42) if len(plot_df) > MAX_MAP_POINTS else plot_df

    # PyDeck Map
    st.subheader(f"Spatial Density Map ({len(map_df):,} of {len(plot_df):,} Points)")
    
    mid_lat = plot_df["latitude"].mean()
    mid_lon = plot_df["longitude"].mean()
    
    view_state = pdk.ViewState(
        latitude=mid_lat if not pd.isna(mid_lat) else 52.5,
        longitude=mid_lon if not pd.isna(mid_lon) else -1.5,
        zoom=6,
        pitch=45
    )
    
    # Hexagon Layer for 3D Density
    hex_layer = pdk.Layer(
        "HexagonLayer",
        data=map_df,
        get_position=["longitude", "latitude"],
        radius=5000,
        elevation_scale=50,
        elevation_range=[0, 1000],
        pickable=True,
        extruded=True,
    )
    
    # Scatterplot Layer for individual points
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["longitude", "latitude"],
        get_color="color",
        get_radius=1500,
        pickable=True,
        opacity=0.7
    )
    
    r = pdk.Deck(
        layers=[hex_layer, scatter_layer],
        initial_view_state=view_state,
        tooltip={"text": "Severity: {severity_name}\nCasualties: {number_of_casualties}"}
    )
    st.pydeck_chart(r)
    
    st.divider()
    
    # Hotspot Authorities Ranking
    st.subheader("Top Local Authorities by Incident Volume")
    auth_summary = plot_df.groupby("local_authority").agg(
        Total_Accidents=("latitude", "count"),
        Fatalities=("severity_name", lambda x: (x == "Fatal").sum()),
        Serious=("severity_name", lambda x: (x == "Serious").sum()),
        Total_Casualties=("number_of_casualties", "sum")
    ).reset_index().sort_values(by="Total_Accidents", ascending=False)
    
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.dataframe(auth_summary, width="stretch", height=350)
    with col2:
        fig_top = px.bar(
            auth_summary.head(8),
            x="Total_Accidents",
            y="local_authority",
            orientation="h",
            color="Total_Accidents",
            color_continuous_scale="Viridis",
            title="Top Local Authorities (ONS code)"
        )
        fig_top.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_top, width="stretch")
