"""
Temporal Dynamics & Trend Analysis Page.
Examines accident volume and severity across hours, days of the week, and seasonal intervals.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import plotly.express as px
from src.etl.db import get_engine

st.set_page_config(page_title="Temporal Dynamics | Traffic Analytics", layout="wide")

st.title("Temporal Dynamics & Accident Patterns")
st.markdown("Analyze how accident incidence fluctuates across hours of the day, weekdays vs. weekends, and rush hour peaks.")


@st.cache_data(ttl=60)
def load_temporal_data():
    try:
        engine = get_engine()
        query = """
            SELECT 
                d.day_name,
                d.day_of_week,
                d.hour,
                d.time_of_day,
                d.is_weekend,
                d.month_name,
                s.severity_name,
                f.number_of_casualties
            FROM fact_accidents f
            JOIN dim_date d ON f.date_key = d.date_key
            JOIN dim_severity s ON f.severity_key = s.severity_key
        """
        df = pd.read_sql(query, con=engine)
        if not df.empty:
            return df
    except Exception:
        pass
    from streamlit_app.demo_data import generate_demo_dataframe
    return generate_demo_dataframe(n=2000)


df = load_temporal_data()

if df.empty:
    st.warning("No data available.")
else:
    # 1. 2D Heatmap: Day of Week vs Hour of Day
    st.subheader("Incident Density by Hour and Day of Week")
    
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot_df = df.pivot_table(index="day_name", columns="hour", values="number_of_casualties", aggfunc="count", fill_value=0)
    pivot_df = pivot_df.reindex(day_order)
    
    fig_heatmap = px.imshow(
        pivot_df,
        labels=dict(x="Hour of Day (00:00 - 23:00)", y="Day of Week", color="Accidents"),
        x=[f"{h:02d}:00" for h in range(24)],
        y=day_order,
        color_continuous_scale="Plasma",
        aspect="auto"
    )
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, width="stretch")
    
    # 2. Rush Hour vs Non-Rush Hour Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Incidents by Time-of-Day Window")
        tod_counts = df.groupby(["time_of_day", "severity_name"]).size().reset_index(name="Accident Count")
        tod_order = ["Morning Rush", "Midday", "Evening Rush", "Night"]
        fig_tod = px.bar(
            tod_counts,
            x="time_of_day",
            y="Accident Count",
            color="severity_name",
            category_orders={"time_of_day": tod_order},
            color_discrete_map={"Fatal": "#EF4444", "Serious": "#F59E0B", "Slight": "#38BDF8"},
            barmode="stack"
        )
        st.plotly_chart(fig_tod, width="stretch")
        
    with col2:
        st.subheader("Casualty Comparison: Weekday vs. Weekend")
        df["Weekend_Label"] = df["is_weekend"].map(
            lambda x: "Weekend (Sat-Sun)" if x in [True, 1, "true", "True", "1"] else "Weekday (Mon-Fri)"
        )
        wk_summary = df.groupby(["Weekend_Label", "severity_name"])["number_of_casualties"].sum().reset_index()
        fig_wk = px.bar(
            wk_summary,
            x="Weekend_Label",
            y="number_of_casualties",
            color="severity_name",
            color_discrete_map={"Fatal": "#EF4444", "Serious": "#F59E0B", "Slight": "#38BDF8"},
            barmode="group",
            labels={"number_of_casualties": "Total Casualties Recorded"}
        )
        st.plotly_chart(fig_wk, width="stretch")
