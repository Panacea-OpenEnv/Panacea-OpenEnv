import streamlit as st
import sqlite3
import pandas as pd
import time
import os

# Streamlit App Configuration
st.set_page_config(page_title="Project Panacea | Telemetry", layout="wide", page_icon="🛡️")

# Database Path
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../telemetry.db'))

def fetch_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            query = "SELECT * FROM audit_logs ORDER BY timestamp DESC"
            df = pd.read_sql(query, conn)
            # Convert timestamp for charting
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Layout
st.title("🛡️ Project Panacea | Executive Telemetry Dashboard")
st.markdown("Real-time audit log of the Multi-Agent RL Oversight Environment.")

# Autorefresh hook
# Streamlit typically uses st_autorefresh or manual rerun. For native lightweight, we just provide a refresh button.
if st.button("Refresh Telemetry"):
    st.experimental_rerun()

df = fetch_data()

if df.empty:
    st.warning("No audit logs found. Please run the environment to populate `telemetry.db`.")
else:
    # High Level Metrics
    total_reviews = len(df[df['event_type'] == 'Oversight_Review'])
    total_rejections = len(df[(df['event_type'] == 'Oversight_Review') & (df['decision'] == 'REJECTED')])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Verifications", total_reviews)
    col2.metric("Deceptive Claims Blocked", total_rejections)
    
    # Basic accuracy math if available
    detection_rate = (total_rejections / total_reviews * 100) if total_reviews > 0 else 0
    col3.metric("Deception Catch Rate", f"{detection_rate:.1f}%")
    
    drift_recoveries = len(df[df['event_type'] == 'Schema_Recovery'])
    col4.metric("Schema Drifts Recovered", drift_recoveries)

    st.divider()

    # Live Audit Feed
    st.subheader("Live Verification Ledger")
    
    # Filter for display
    display_df = df[['timestamp', 'agent_id', 'patient_id', 'decision', 'reasoning', 'query_executed']].head(20)
    
    # Apply some styling based on decision
    def color_decision(val):
        color = '#ff4b4b' if val == 'REJECTED' else '#00cc66' if val == 'APPROVED' else 'white'
        return f'color: {color}; font-weight: bold'

    st.dataframe(
        display_df.style.applymap(color_decision, subset=['decision']),
        use_container_width=True,
        hide_index=True
    )

    # Analytics
    st.subheader("Department Trust Penalties Over Time")
    if not df.empty and 'agent_id' in df.columns:
        # Group rejections by department (assuming agent_id == department)
        rejections = df[df['decision'] == 'REJECTED']
        if not rejections.empty:
            dept_counts = rejections['agent_id'].value_counts()
            st.bar_chart(dept_counts, color="#ff4b4b")
        else:
            st.info("No rejections recorded yet.")
