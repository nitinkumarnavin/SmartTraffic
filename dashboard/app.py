import streamlit as st
import pandas as pd
import sys
import os

# Add project root path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

# Import APIs
from audio_model.siren_detector import detect_siren
from accident_detection.accident_api import detect_accident

st.set_page_config(
    page_title="Smart Traffic Management System",
    layout="wide"
)

st.title("🚦 Smart Traffic Management System")

st.write("Integrated Dashboard")

# =========================
# Sidebar
# =========================

st.sidebar.title("Control Panel")

traffic_level = st.sidebar.selectbox(
    "Traffic Level",
    ["Low", "Medium", "High"]
)

st.write("Current Traffic Level:", traffic_level)

# =========================
# Signal Logic
# =========================

if traffic_level == "Low":
    st.success("🟢 Signal Status: GREEN")
    st.info("⏱ Green Signal Time: 30 Seconds")

elif traffic_level == "Medium":
    st.warning("🟡 Signal Status: YELLOW")
    st.info("⏱ Green Signal Time: 60 Seconds")

else:
    st.error("🔴 Signal Status: RED")
    st.info("⏱ Green Signal Time: 90 Seconds")

# =========================
# Metrics
# =========================

col1, col2, col3, col4 = st.columns(4)

if traffic_level == "Low":
    vehicles = 120
elif traffic_level == "Medium":
    vehicles = 260
else:
    vehicles = 410

col1.metric("Total Vehicles", vehicles)

col2.metric("Emergency Vehicles", 0)

if traffic_level == "Low":
    accidents = 0
elif traffic_level == "Medium":
    accidents = 1
else:
    accidents = 3

col3.metric("Accidents", accidents)

col4.metric("Signals Active", 4)

# =========================
# Traffic Density
# =========================

st.header("📊 Traffic Density")

traffic_data = pd.DataFrame({
    "Road": ["Road A", "Road B", "Road C", "Road D"],
    "Vehicles": [120, 80, 150, 60]
})

st.bar_chart(
    traffic_data.set_index("Road")
)

# =========================
# Emergency Vehicle Detection
# =========================

st.header("🚑 Emergency Vehicle Detection")

if st.button("Run Siren Detection"):

    try:

        result = detect_siren(
            "dataset/ambulance/u_xg7ssi08yr-ambulance-siren-363656.mp3"
        )

        st.error(
            f"🚨 Detected: {result}"
        )

    except Exception as e:

        st.error(
            f"Audio Detection Error: {e}"
        )

else:

    st.success(
        "No Emergency Vehicle Detected"
    )

# =========================
# Accident Detection
# =========================

st.header("⚠️ Accident Detection")

if st.button("Run Accident Detection"):

    try:

        result = detect_accident(
            "accident_detection/videos/test.mp4"
        )

        st.warning(result)

    except Exception as e:

        st.error(
            f"Accident Detection Error: {e}"
        )

else:

    st.success(
        "No Accident Detected"
    )

# =========================
# Signal Status
# =========================

st.header("🚦 Signal Status")

st.info(
    "All Traffic Signals Working Normally"
)
st.markdown("---")
st.subheader("🌐 Network Simulation Status")

col1, col2 = st.columns(2)

with col1:
    st.metric("Packet Delivery Rate", "98%")
    st.metric("Average Delay", "12 ms")

with col2:
    st.metric("Network Nodes", "25")
    st.metric("Emergency Route", "Active")

st.success("Network Status: Connected")