from datetime import datetime, timedelta
import sqlite3
import cv2
import numpy as np
import pandas as pd 
import uuid
import streamlit as st
from ultralytics import YOLO
from preprocess import preprocess_frame

# Load the trained/pretrained model
model = YOLO("yolov8s.pt")

# Export to ONNX format with FP16 precision for edge hardware
model.export(format="onnx", half=True, dynamic=True)
print("Successfully exported model to YOLOv8s.onnx for edge deployment!")

# 1. Page Configuration
st.set_page_config(
    page_title="Base Station - Border Surveillance",
    page_icon="🛡️",
    layout="wide",
)

st.title(
    "🛡️ Base Station Command Center (Edge-AI & Tactical Radio Integration)"
)
st.write(
    "Offline Architecture: Base Station Streamlit Dashboard ➡️ Outpost Guard RF"
    " Dispatch ➡️ Radio Confirmation"
)


# Initialize Local SQLite Database
def init_db():
  conn = sqlite3.connect("border_events.db", check_same_thread=False)
  cursor = conn.cursor()
  cursor.execute("DROP TABLE IF EXISTS intrusions")
  cursor.execute("""
        CREATE TABLE intrusions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            camera_id TEXT,
            object_type TEXT,
            person_id TEXT,
            location TEXT,
            time_spent TEXT,
            event_type TEXT,
            guard_status TEXT
        )
    """)
  conn.commit()
  return conn


db_conn = init_db()


# Safe Model Loader
@st.cache_resource
def load_model():
  return YOLO("yolov8s.pt")


with st.spinner("Initializing AI Model & ByteTrack Engine..."):
  model = load_model()

# 2. Sidebar Controls & Configurations
st.sidebar.header("Base Station Control Panel")
source_option = st.sidebar.selectbox(
    "Select Video Source", ["Sample Video File", "Live Webcam (0)"]
)

confidence_threshold = st.sidebar.slider(
    "AI Confidence Threshold", 0.1, 1.0, 0.35, 0.05
)

st.sidebar.markdown("---")
st.sidebar.subheader("📡 Offline Tactical Transmission")
st.sidebar.info(
    "Zero-Internet Mode: Alerts broadcast to field guards via local LoRa/RF"
    " mesh and voice radio confirmation."
)
radio_frequency = st.sidebar.text_input(
    "Tactical Radio Channel", value="142.5 MHz (VHF)"
)

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ Boundary Configuration")
fence_mode = st.sidebar.radio(
    "Select Intrusion Zone Type", ["Vector Line Crossing", "Polygon Zone"]
)

if fence_mode == "Vector Line Crossing":
  line_pos_pct = st.sidebar.slider("Line Position (Height %)", 10, 90, 60)
else:
  st.sidebar.subheader("Polygon Coordinates (%)")
  z_x1 = st.sidebar.slider("Top-Left X (%)", 0, 100, 20)
  z_y1 = st.sidebar.slider("Top-Left Y (%)", 0, 100, 30)
  z_x2 = st.sidebar.slider("Top-Right X (%)", 0, 100, 80)
  z_y2 = st.sidebar.slider("Top-Right Y (%)", 0, 100, 30)
  z_x3 = st.sidebar.slider("Bottom-Right X (%)", 0, 100, 80)
  z_y3 = st.sidebar.slider("Bottom-Right Y (%)", 0, 100, 70)
  z_x4 = st.sidebar.slider("Bottom-Left X (%)", 0, 100, 20)
  z_y4 = st.sidebar.slider("Bottom-Left Y (%)", 0, 100, 70)

with st.sidebar.form("threshold_form"):
    st.subheader("Advanced Surveillance Thresholds")
    
    loitering_threshold_sec = st.slider("Loitering Threshold (Seconds)", 5, 30, 10)
    erratic_velocity_spike = st.slider("Erratic Movement Sensitivity", 10, 100, 50)
    max_heading_deviation = st.slider("Max Heading Deviation (Degrees)", 10, 90, 45)
    cooldown_period_sec = st.slider("Alert Cooldown Period (Seconds)", 5, 60, 15)
    
    submitted = st.form_submit_button("Apply Thresholds")

if submitted:
    st.session_state.loitering_threshold = loitering_threshold_sec
    st.session_state.erratic_sensitivity = erratic_velocity_spike
    st.session_state.max_heading_deviation = max_heading_deviation
    st.session_state.alert_cooldown = cooldown_period_sec
    st.success("Settings applied!")

st.sidebar.subheader("Environmental Filters")
enable_advanced_filters = st.sidebar.checkbox(
    "⚡ Enable 4-Step Haze/Low-Light AI Pipeline", value=True
)
enable_gamma = st.sidebar.checkbox("Apply Gamma Correction (Night Vision)", value=False)

run_system = st.sidebar.checkbox("Start Base Station Feed", value=False)

# Multi-Tab Layout Structure
tab1, tab2 = st.tabs([
    "🛡️ Base Station Monitoring & Field Confirmation",
    "📊 Sector Analytics & Logs",
])

SECTOR_NAME = "Sector-04 (Rajasthan Post)"
SECTOR_LAT = 28.6139
SECTOR_LON = 77.2090
EXPECTED_HEADING_DEG = 90.0

# Session State Tracking Dictionaries
if "intrusion_logs" not in st.session_state:
  st.session_state.intrusion_logs = []
if "active_threats" not in st.session_state:
  st.session_state.active_threats = set()
if "track_history" not in st.session_state:
  st.session_state.track_history = {}

# --- TAB 1: BASE STATION MONITORING & FIELD CONFIRMATION ---
with tab1:
  col1, col2 = st.columns([2, 1])

  with col1:
    st.subheader("Live Sector Feed & Predictive Vectors")
    video_placeholder = st.empty()

  with col2:
    st.subheader("🚨 Base Station Alert & Field Guard Status")
    alert_placeholder = st.empty()

    st.markdown("---")
    st.subheader("📻 Field Guard Radio Confirmation Panel")
    guard_action_placeholder = st.empty()

# --- TAB 2: ANALYTICS DASHBOARD ---
with tab2:
  st.subheader("📊 Historical Threat & Surveillance Analytics")
  analytics_metrics_placeholder = st.empty()
  analytics_table_placeholder = st.empty()

# 3. Video Handling & Inference Loop
cap = None

if run_system:
  if source_option == "Sample Video File":
    video_path = "videos/sample_border.mp4"
    cap = cv2.VideoCapture(video_path)
  else:
    cap = cv2.VideoCapture(0)

  if cap is not None and not cap.isOpened():
    st.error(
        "Error: Could not open video source. Check file path or webcam connection."
    )
  else:
    while run_system and cap.isOpened():
      ret, frame = cap.read()

      if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

      frame = cv2.resize(frame, (512, 288))
      height, width, _ = frame.shape
      current_time_dt = datetime.now()
      current_time_str = current_time_dt.strftime("%Y-%m-%d %H:%M:%S")

      # --- ENVIRONMENTAL & ADVANCED PREPROCESSING ---
      if enable_advanced_filters:
          frame = preprocess_frame(frame)

      # --- BYTETRACK MULTI-OBJECT TRACKING ---
      results = model.track(
          frame,
          persist=True,
          tracker="bytetrack.yaml",
          conf=confidence_threshold,
          iou=0.5,
          classes=[0, 2, 3, 5, 7],
          verbose=False,
      )
      boxes = results[0].boxes
      annotated_frame = results[0].plot()
      
      current_frame_threats = set()
      active_track_ids = set()

      # --- BOUNDARY SETUP ---
      if fence_mode == "Vector Line Crossing":
        line_y = int(height * (line_pos_pct / 100.0))
        cv2.line(annotated_frame, (0, line_y), (width, line_y), (0, 0, 255), 3)
        cv2.putText(
            annotated_frame,
            "VECTOR FENCE LINE",
            (20, line_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            2,
        )
      else:
        pts = np.array(
            [
                [int(width * z_x1 / 100), int(height * z_y1 / 100)],
                [int(width * z_x2 / 100), int(height * z_y2 / 100)],
                [int(width * z_x3 / 100), int(height * z_y3 / 100)],
                [int(width * z_x4 / 100), int(height * z_y4 / 100)],
            ],
            np.int32,
        )
        cv2.polylines(annotated_frame, [pts], True, (0, 0, 255), 3)
        overlay = annotated_frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 255))
        cv2.addWeighted(
            overlay, 0.15, annotated_frame, 0.85, 0, annotated_frame
        )
        cv2.putText(
            annotated_frame,
            "RESTRICTED POLYGON ZONE",
            (pts[0][0], max(20, pts[0][1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            2,
        )

      for i, box in enumerate(boxes):
        if box.id is None:
          continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        track_id = int(box.id[0])
        track_str = f"ID-{track_id}"
        active_track_ids.add(track_str)

        anchor_x = int((x1 + x2) / 2)
        anchor_y = y2

        if track_str not in st.session_state.track_history:
          st.session_state.track_history[track_str] = {
              "entry_time": current_time_dt,
              "positions": [],
              "last_alert": datetime.min,
              "guard_status": "Pending Radio Confirmation",
          }

        track_data = st.session_state.track_history[track_str]
        track_data["positions"].append((anchor_x, anchor_y))
        if len(track_data["positions"]) > 15:
          track_data["positions"].pop(0)

        # --- SMOOTHED MOTION VECTOR CALCULATION ---
        heading_anomaly = False
        calculated_angle = EXPECTED_HEADING_DEG

        if len(track_data["positions"]) >= 5:
          pos_arr = np.array(track_data["positions"])
          diffs = np.diff(pos_arr, axis=0)
          avg_dx = np.mean(diffs[:, 0])
          avg_dy = np.mean(diffs[:, 1])

          if abs(avg_dx) > 0.5 or abs(avg_dy) > 0.5:
            calculated_angle = np.degrees(np.arctan2(avg_dy, avg_dx))
            if calculated_angle < 0:
              calculated_angle += 360

            arrow_end_x = int(anchor_x + avg_dx * 8)
            arrow_end_y = int(anchor_y + avg_dy * 8)
            cv2.arrowedLine(
                annotated_frame,
                (anchor_x, anchor_y),
                (arrow_end_x, arrow_end_y),
                (0, 255, 255),
                2,
                tipLength=0.3,
            )

            angle_diff = abs(calculated_angle - EXPECTED_HEADING_DEG)
            if angle_diff > 180:
              angle_diff = 360 - angle_diff

            if angle_diff > max_heading_deviation:
              heading_anomaly = True

        in_violation = False
        if fence_mode == "Vector Line Crossing":
          if y1 <= line_y <= y2:
            in_violation = True
        else:
          in_violation = (
              cv2.pointPolygonTest(pts, (anchor_x, anchor_y), False) >= 0
          )

        last_pos = (
            track_data["positions"][-2]
            if len(track_data["positions"]) >= 2
            else (anchor_x, anchor_y)
        )
        velocity_delta = np.sqrt(
            (anchor_x - last_pos[0]) ** 2 + (anchor_y - last_pos[1]) ** 2
        )

        event_type = None
        if in_violation:
          current_frame_threats.add(track_str)
          time_spent_sec = (
              current_time_dt - track_data["entry_time"]
          ).total_seconds()

          if time_spent_sec >= loitering_threshold_sec:
            event_type = "LOITERING THREAT"
          elif velocity_delta >= erratic_velocity_spike:
            event_type = "ERRATIC MOVEMENT BREACH"
          else:
            event_type = "PERIMETER BREACH"
        elif heading_anomaly:
          current_frame_threats.add(track_str)
          event_type = "TRAJECTORY ANOMALY (OUT-OF-ZONE)"

        if event_type:
          time_since_last_alert = (
              current_time_dt - track_data["last_alert"]
          ).total_seconds()
          if time_since_last_alert >= cooldown_period_sec:
            track_data["last_alert"] = current_time_dt

            total_time_spent_sec = int(
                (current_time_dt - track_data["entry_time"]).total_seconds()
            )
            time_spent_str = (
                f"{total_time_spent_sec}s"
                if total_time_spent_sec < 60
                else f"{total_time_spent_sec // 60}m {total_time_spent_sec % 60}s"
            )

            alert_msg = f"[{current_time_str}] 🚨 Dispatched RF Alert -> {event_type}: {class_name.upper()} ({track_str}) | GPS: {SECTOR_LAT}, {SECTOR_LON}"
            st.session_state.intrusion_logs.insert(0, alert_msg)
            if len(st.session_state.intrusion_logs) > 8:
              st.session_state.intrusion_logs.pop()

            cursor = db_conn.cursor()
            cursor.execute(
                """
                        INSERT INTO intrusions (timestamp, camera_id, object_type, person_id, location, time_spent, event_type, guard_status)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                (
                    current_time_str,
                    "Cam-01",
                    class_name.upper(),
                    track_str,
                    SECTOR_NAME,
                    time_spent_str,
                    event_type,
                    track_data["guard_status"],
                ),
            )
            db_conn.commit()

      st.session_state.active_threats = current_frame_threats

      frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
      video_placeholder.image(
          frame_rgb,
          channels="RGB",
          use_container_width=True,
          caption="Base Station Live Command Feed",
      )

      with alert_placeholder.container():
        total_objects = len(boxes)
        detected_types = [model.names[int(b.cls[0])] for b in boxes]
        st.metric(
            label="Active Tracked Targets in Sector", value=total_objects
        )
        if st.session_state.active_threats:
          st.error(
              f"⚠️ ACTIVE THREATS DISPATCHED: {len(st.session_state.active_threats)}"
          )
          for log in st.session_state.intrusion_logs[:3]:
            st.warning(log)
        else:
          st.success("Sector Secure. Radio link stable.")

      with guard_action_placeholder.container():
          if st.session_state.active_threats:
              st.write(
                  "**Field Guard Contact Options (via Radio Channel"
                  f" {radio_frequency}):**"
              )
              
              for idx, t_id in enumerate(list(st.session_state.active_threats)):
                  if t_id in st.session_state.track_history:
                      t_data = st.session_state.track_history[t_id]
                      
                      col1, col2, col3 = st.columns([2, 1, 1])
                      col1.write(f"**{t_id}** Status: *{t_data['guard_status']}*")
                      
                      # FIX: Generate a completely unique runtime key using uuid4 or distinct namespace
                      unique_suffix = uuid.uuid4().hex[:8]
                      btn_key_true = f"true_{t_id}_{idx}_{unique_suffix}"
                      btn_key_false = f"false_{t_id}_{idx}_{unique_suffix}"
                      
                      if col2.button(f"Confirm True Threat ({t_id})", key=btn_key_true):
                          t_data["guard_status"] = "CONFIRMED TRUE THREAT"
                          
                      if col3.button(f"Mark False Alarm ({t_id})", key=btn_key_false):
                          t_data["guard_status"] = "FALSE ALARM DISMISSED"
          else:
              st.info("No active field alerts requiring radio confirmation at this moment.")
      # Fetch logs for Analytics Tab
      df_logs = pd.read_sql_query(
          """
                SELECT timestamp, object_type, person_id, event_type, time_spent, guard_status, location 
                FROM intrusions 
                ORDER BY id DESC
            """,
          db_conn,
      )

      with analytics_metrics_placeholder.container():
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Dispatched Alerts", len(df_logs))
        m2.metric(
            "Confirmed Threats",
            len(
                df_logs[
                    df_logs["guard_status"] == "CONFIRMED TRUE THREAT"
                ]
            )
            if not df_logs.empty
            else 0,
        )
        m3.metric("System Mode", "OFFLINE SECURE (LoRa/RF)")

      with analytics_table_placeholder.container():
        if not df_logs.empty:
          st.dataframe(df_logs, use_container_width=True)
        else:
          st.info("No intrusion logs recorded yet.")

  if cap:
    cap.release()
else:
  video_placeholder.info(
      "System paused. Toggle 'Start Base Station Feed' in the sidebar to"
      " initialize."
  )
  with alert_placeholder.container():
    st.info("Awaiting system activation...")

  with guard_action_placeholder.container():
    st.info("Radio monitoring standby...")

  df_logs = pd.read_sql_query(
      """
        SELECT timestamp, object_type, person_id, event_type, time_spent, guard_status, location 
        FROM intrusions 
        ORDER BY id DESC
    """,
      db_conn,
  )
  with analytics_metrics_placeholder.container():
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Dispatched Alerts", len(df_logs))
    m2.metric(
        "Confirmed Threats",
        len(df_logs[df_logs["guard_status"] == "CONFIRMED TRUE THREAT"])
        if not df_logs.empty
        else 0,
    )
    m3.metric("System Mode", "STANDBY")

  with analytics_table_placeholder.container():
    if not df_logs.empty:
      st.dataframe(df_logs, use_container_width=True)
    else:
      st.info("No historical logs available.")