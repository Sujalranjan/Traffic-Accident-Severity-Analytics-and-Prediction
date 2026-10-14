"""
Pipeline Audit & Drift Monitoring Page.
Displays ingestion run history, data quality rejection quarantine logs,
and real-time statistical drift analysis with retraining status.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import json
import streamlit as st
import pandas as pd
from src.etl.db import get_engine
from src.monitoring.drift_detector import run_drift_analysis

st.set_page_config(page_title="Pipeline Audit & Drift | MLOps", layout="wide")

st.title("Data Engineering Audit & Drift Monitoring")
st.markdown("Inspect ingestion pipeline audit traces, data quality quarantine records, and statistical drift indicators.")

engine = get_engine()

tab1, tab2, tab3, tab4 = st.tabs([
    "Ingestion Audit Trail", 
    "Quarantine Rejection Log", 
    "Statistical Drift Monitoring",
    "Airflow Orchestration"
])

with tab1:
    st.subheader("Ingestion Run History & Health")
    try:
        audit_df = pd.read_sql("SELECT * FROM ingestion_audit ORDER BY started_at DESC", con=engine)
        if not audit_df.empty:
            st.dataframe(audit_df, width="stretch", height=300)
        else:
            st.info("No audit logs recorded yet.")
    except Exception:
        st.info("The warehouse database is not reachable from this deployment, so no ingestion audit is available. "
                "Run the pipeline locally or with Docker Compose to populate `ingestion_audit`.")

with tab2:
    st.subheader("Quarantined Faulty Records (Error Log)")
    try:
        quarantine_df = pd.read_sql("SELECT * FROM quarantine_rejected_records ORDER BY rejected_at DESC", con=engine)
        if not quarantine_df.empty:
            c1, c2 = st.columns([1, 2])
            with c1:
                st.metric("Total Quarantined Records", len(quarantine_df))
                reason_counts = quarantine_df["rejection_reason"].value_counts().reset_index()
                reason_counts.columns = ["Reason", "Count"]
                st.dataframe(reason_counts, width="stretch")
            with c2:
                st.dataframe(quarantine_df[["record_identifier", "rejection_reason", "rejected_at"]], width="stretch", height=260)
        else:
            st.success("Zero quarantined records found in database.")
    except Exception:
        st.info("The warehouse database is not reachable from this deployment, so no quarantine log is available.")

with tab3:
    st.subheader("Statistical Drift Detection & Retraining Triggers")
    st.markdown("Evaluates distribution shifts between baseline reference data and recent production records.")
    
    if st.button("Run Drift Analysis", key="run_drift"):
        with st.spinner("Calculating PSI and Kolmogorov-Smirnov statistics..."):
            try:
                report = run_drift_analysis()
                st.session_state["drift_report"] = report
                st.success("Drift analysis updated successfully.")
            except Exception:
                st.info("Demonstration Mode: Displaying baseline reference drift report.")
            
    # Load latest report or run if not present
    report = st.session_state.get("drift_report")
    if not report:
        # In Docker this points at the drift reports written by Airflow
        reports_dir = Path(os.getenv("REPORTS_DIR", "reports"))
        report_files = sorted(reports_dir.glob("drift_report_*.json"), reverse=True)
        if report_files:
            with open(report_files[0], "r", encoding="utf-8") as f:
                report = json.load(f)
                
    if report:
        retrain_decision = report.get("retraining_decision", {})
        is_triggered = retrain_decision.get("retrain_triggered", False)
        
        status_color = "#EF4444" if is_triggered else "#10B981"
        status_text = "RETRAINING TRIGGERED" if is_triggered else "STABLE (NO RETRAINING REQUIRED)"
        
        st.markdown(f"""
        <div style="background-color: #0F172A; border-radius: 10px; padding: 1.2rem; border-left: 6px solid {status_color}; margin-bottom: 1.5rem;">
            <span style="color: #94A3B8; font-size: 0.85rem; text-transform: uppercase;">Automated Retraining Status</span>
            <div style="color: {status_color}; font-size: 1.6rem; font-weight: 700;">{status_text}</div>
            <div style="color: #CBD5E1; font-size: 0.9rem; margin-top: 0.3rem;">Reason: {", ".join(retrain_decision.get("triggers", []))}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature Drift Table
        st.markdown("#### Feature Drift (PSI & KS-Test)")
        feat_drift = report.get("feature_drift", {})
        drift_rows = []
        for feat, metrics in feat_drift.items():
            drift_rows.append({
                "Feature": feat,
                "PSI": metrics.get("psi"),
                "KS-Statistic": metrics.get("ks_statistic"),
                "P-Value": metrics.get("p_value"),
                "Status": metrics.get("status"),
                "Drift Detected": metrics.get("drift_detected")
            })
        st.dataframe(pd.DataFrame(drift_rows), width="stretch")
        
        # Location & Class Shifts
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Location Drift")
            loc_data = report.get("location_drift", {})
            st.json(loc_data)
        with c2:
            st.markdown("#### Severity Class Shift (Ref vs. Current)")
            class_data = report.get("class_distribution", {})
            st.json(class_data)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("#### Model Performance & False-Negative Rates (current window)")
            st.json(report.get("model_performance", {"evaluated": False}))
        with c4:
            st.markdown("#### Live Predictions (prediction_log)")
            st.json(report.get("live_predictions", {}))
    else:
        st.info("Click 'Run On-Demand Drift Analysis' to compute drift indicators.")

with tab4:
    st.subheader("🔄 Apache Airflow Orchestrated DAG Workflow")
    st.markdown("""
    The automated end-to-end pipeline is governed by the DAG: **`accident_severity_pipeline`** located at `dags/accident_pipeline_dag.py`.
    """)
    
    # Airflow DAG Graph Representation
    st.markdown("#### Airflow Directed Acyclic Graph (DAG) Topology")
    dag_flow_html = """
    <div style="background-color: #0F172A; border-radius: 12px; padding: 1.5rem; border: 1px solid #334155; margin-bottom: 1.5rem;">
        <div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 0.8rem;">
            <div style="background: #1E293B; border: 1px solid #38BDF8; border-radius: 8px; padding: 0.8rem 1rem; color: #38BDF8; font-weight: 600; text-align: center;">
                1. check_db_readiness<br><span style="font-size: 0.75rem; color: #94A3B8;">DB Connectivity</span>
            </div>
            <div style="color: #64748B; font-size: 1.5rem;">➔</div>
            <div style="background: #1E293B; border: 1px solid #10B981; border-radius: 8px; padding: 0.8rem 1rem; color: #10B981; font-weight: 600; text-align: center;">
                2. ingest_raw_accidents<br><span style="font-size: 0.75rem; color: #94A3B8;">Raw Landing Zone</span>
            </div>
            <div style="color: #64748B; font-size: 1.5rem;">➔</div>
            <div style="background: #1E293B; border: 1px solid #F59E0B; border-radius: 8px; padding: 0.8rem 1rem; color: #F59E0B; font-weight: 600; text-align: center;">
                3. ingest_weather_api<br><span style="font-size: 0.75rem; color: #94A3B8;">Open-Meteo REST</span>
            </div>
            <div style="color: #64748B; font-size: 1.5rem;">➔</div>
            <div style="background: #1E293B; border: 1px solid #EF4444; border-radius: 8px; padding: 0.8rem 1rem; color: #EF4444; font-weight: 600; text-align: center;">
                4. validate_quarantine<br><span style="font-size: 0.75rem; color: #94A3B8;">Quality Gate</span>
            </div>
            <div style="color: #64748B; font-size: 1.5rem;">➔</div>
            <div style="background: #1E293B; border: 1px solid #A855F7; border-radius: 8px; padding: 0.8rem 1rem; color: #A855F7; font-weight: 600; text-align: center;">
                5. transform_star_schema<br><span style="font-size: 0.75rem; color: #94A3B8;">Warehouse ETL</span>
            </div>
            <div style="color: #64748B; font-size: 1.5rem;">➔</div>
            <div style="background: #1E293B; border: 1px solid #38BDF8; border-radius: 8px; padding: 0.8rem 1rem; color: #38BDF8; font-weight: 600; text-align: center;">
                6. refresh_data_marts<br><span style="font-size: 0.75rem; color: #94A3B8;">Analytical Marts</span>
            </div>
            <div style="color: #64748B; font-size: 1.5rem;">➔</div>
            <div style="background: #1E293B; border: 1px solid #F97316; border-radius: 8px; padding: 0.8rem 1rem; color: #F97316; font-weight: 600; text-align: center;">
                7. drift_and_retrain_check<br><span style="font-size: 0.75rem; color: #94A3B8;">MLOps Monitoring</span>
            </div>
            <div style="color: #64748B; font-size: 1.5rem;">➔</div>
            <div style="background: #1E293B; border: 1px dashed #F97316; border-radius: 8px; padding: 0.8rem 1rem; color: #F97316; font-weight: 600; text-align: center;">
                8. retrain_and_register_model<br><span style="font-size: 0.75rem; color: #94A3B8;">Only if a trigger fires</span>
            </div>
        </div>
    </div>
    """
    st.markdown(dag_flow_html, unsafe_allow_html=True)
    
    col_dag1, col_dag2 = st.columns(2)
    with col_dag1:
        st.markdown("#### DAG Scheduling & Execution Attributes")
        st.json({
            "dag_id": "accident_severity_pipeline",
            "schedule_interval": "@daily",
            "retries": 2,
            "retry_delay": "2 minutes",
            "catchup": False,
            "orchestrator": "Apache Airflow (LocalExecutor)",
            "state_passing": "Airflow XComs (Batch ID Propagation)"
        })
        
    with col_dag2:
        st.markdown("#### How to Open Apache Airflow Web UI")
        st.code("""# 1. Start Airflow with Docker Compose:
docker compose up airflow-webserver airflow-scheduler postgres -d

# 2. Access the Airflow Dashboard in your browser:
http://localhost:8080

# 3. Default Login Credentials:
Username: admin
Password: admin""", language="bash")
