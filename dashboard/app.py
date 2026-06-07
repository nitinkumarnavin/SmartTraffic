import streamlit as st
import pandas as pd

st.title("🚦 Smart Traffic Management System")

st.write("Member 3 Dashboard")
st.sidebar.title("Control Panel")
traffic_level = st.sidebar.selectbox(
    "Traffic Level",
    ["Low", "Medium", "High"]
)

st.write("Current Traffic Level:", traffic_level)
emergency_count = 0
if emergency_count == 1:
    st.success("🚑 Emergency Priority Activated")
    st.info("🚑 Emergency Green Corridor Enabled")

elif traffic_level == "Low":
    st.success("🟢 Signal Status: GREEN")
    st.info("⏱ Green Signal Time: 30 Seconds")

elif traffic_level == "Medium":
    st.warning("🟡 Signal Status: YELLOW")
    st.info("⏱ Green Signal Time: 60 Seconds")

else:
    st.error("🔴 Signal Status: RED")
    st.info("⏱ Green Signal Time: 90 Seconds") 
if st.button("⚠️ Accident Detected"):
    st.warning("⚠️ Accident Reported on Road")


col1, col2, col3, col4 = st.columns(4)

if traffic_level == "Low":
    vehicles = 120
elif traffic_level == "Medium":
    vehicles = 260
else:
    vehicles = 410

col1.metric("Total Vehicles", vehicles)
emergency_count = 0

if st.button("🚑 Emergency Vehicle Detected"):
    emergency_count = 1
    st.error("🚨 Emergency Vehicle Approaching")

col2.metric("Emergency Vehicles", emergency_count)
if traffic_level == "Low":
    accidents = 0
elif traffic_level == "Medium":
    accidents = 1
else:
    accidents = 3

col3.metric("Accidents", accidents)
col4.metric("Signals Active", "4")

# Traffic Density
st.header("Traffic Density")

traffic_data = pd.DataFrame({
    "Road": ["Road A", "Road B", "Road C", "Road D"],
    "Vehicles": [120, 80, 150, 60]
})

st.bar_chart(traffic_data.set_index("Road"))

# Emergency Vehicle
st.header("🚑 Emergency Vehicle Detection")

st.success("No Emergency Vehicle Detected")

# Accident Detection
st.header("⚠️ Accident Detection")

st.success("No Accident Detected")

# Signal Status
st.header("🚦 Signal Status")

st.info("All Traffic Signals Working Normally")