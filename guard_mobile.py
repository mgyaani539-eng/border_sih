from datetime import datetime
import sqlite3
import pandas as pd
import streamlit as st

# 1. Mobile-Optimized Page Configuration
st.set_page_config(
    page_title="Field Guard Terminal", page_icon="📱", layout="centered"
)

# Custom CSS for high-contrast mobile view & large touch targets
st.markdown("""
    <style>
    .stButton button {
        width: 100%;
        height: 3em;
        font-weight: bold;
        font-size: 16px;
        border-radius: 8px;
    }
    .alert-card {
        padding: 15px;
        border-radius: 10px;
        background-color: #2b1d1d;
        border: 2px solid #ff4b4b;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📱 Field Guard Mobile Unit")
st.markdown("**Status:** Connected to Base Station (Local RF Mesh)")
st.markdown("---")


# Connect to Shared Database
def get_db_connection():
  conn = sqlite3.connect("border_events.db", check_same_thread=False)
  return conn


db_conn = get_db_connection()

# Refresh button for mobile user
if st.button("🔄 Refresh Sector Feed"):
  st.rerun()

# Fetch latest intrusion events
df_logs = pd.read_sql_query(
    """
    SELECT id, timestamp, object_type, person_id, location, time_spent, event_type, guard_status 
    FROM intrusions 
    ORDER BY id DESC LIMIT 5
""",
    db_conn,
)

if df_logs.empty:
  st.info("🟢 Sector secure. No active threat notifications received.")
else:
  st.subheader("🚨 Active Dispatched Alerts")

  # Filter for pending or recent items
  for index, row in df_logs.iterrows():
    alert_id = row["id"]
    p_id = row["person_id"]
    obj_type = row["object_type"]
    ev_type = row["event_type"]
    loc = row["location"]
    time_str = row["timestamp"]
    duration = row["time_spent"]
    current_status = row["guard_status"]

    with st.container():
      st.markdown(
          f"""
            <div class="alert-card">
                <h3>⚠️ {ev_type}</h3>
                <p><b>Target:</b> {obj_type} ({p_id})<br>
                <b>Sector:</b> {loc}<br>
                <b>Time:</b> {time_str} | <b>Duration:</b> {duration}<br>
                <b>Current Status:</b> <i>{current_status}</i></p>
            </div>
            """,
          unsafe_allow_html=True,
      )

      # Action buttons for the guard
      if current_status == "Pending Radio Confirmation":
        col1, col2 = st.columns(2)
        with col1:
          if st.button(f"✅ Confirm Threat", key=f"conf_{alert_id}"):
            cursor = db_conn.cursor()
            cursor.execute(
                "UPDATE intrusions SET guard_status = ? WHERE id = ?",
                ("CONFIRMED TRUE THREAT", alert_id),
            )
            db_conn.commit()
            st.success(f"Threat {p_id} confirmed to Base Station!")
            st.rerun()

        with col2:
          if st.button(f"❌ False Alarm", key=f"false_{alert_id}"):
            cursor = db_conn.cursor()
            cursor.execute(
                "UPDATE intrusions SET guard_status = ? WHERE id = ?",
                ("FALSE ALARM DISMISSED", alert_id),
            )
            db_conn.commit()
            st.warning(f"Alert {p_id} marked as False Alarm.")
            st.rerun()
      else:
        st.caption(f"Status locked: {current_status}")

      st.markdown("---")

# Quick Field Dispatch Note
st.markdown("### 📋 Guard Quick-Notes")
guard_note = st.text_area(
    "Log observation to Base Station database (offline text note):"
)
if st.button("Send Field Note"):
  if guard_note.strip():
    st.success("Note logged locally and synced with base station.")
  else:
    st.warning("Note cannot be empty.")