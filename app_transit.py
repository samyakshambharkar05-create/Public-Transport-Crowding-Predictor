
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Page Configuration & Layout
st.set_page_config(
    page_title="TransitPulse | Public Transport Crowding Predictor", 
    page_icon="🚌", 
    layout="wide"
)

# 2. Load Backend Machine Learning Artifacts Safely
routes_list = [f"Route-{i}" for i in range(101, 111)]
days_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weather_list = ["Clear", "Rainy", "Overcast"]

with open('transit_regressor.pkl', 'rb') as f:
    reg_model = pickle.load(f)
with open('transit_classifier.pkl', 'rb') as f:
    clf_model = pickle.load(f)
with open('transit_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

try:
    dropdown_routes = encoders['route'].classes_
    dropdown_days = encoders['day'].classes_
    dropdown_weather = encoders['weather'].classes_
except:
    dropdown_routes = routes_list
    dropdown_days = days_list
    dropdown_weather = weather_list

# 3. Main Dashboard Header
st.title("🚌 TransitPulse AI")
st.subheader("Predictive Congestion Optimization Dashboard")
st.write("Leveraging dual-engine Machine Learning (Regression + Random Forest Classification) to optimize urban transit schedules.")
st.markdown("---")

# 4. Interactive Input Sidebar
st.sidebar.header("📋 Journey Parameters")
st.sidebar.write("Configure details to simulate expected rider metrics:")

route_selection = st.sidebar.selectbox("🎯 Target Route Number", dropdown_routes)
day_selection = st.sidebar.selectbox("📅 Day of the Week", dropdown_days)
time_selection = st.sidebar.slider("⏰ Departure Hour (24hr clock)", 6, 22, 9)
weather_selection = st.sidebar.selectbox("🌦️ Current Weather Condition", dropdown_weather)

st.sidebar.markdown("---")
predict_clicked = st.sidebar.button("⚡ Run Predictive Inference", type="primary", use_container_width=True)

# 5. Content Layout (Main Screen Container)
if predict_clicked:
    # Feature Vector Preprocessing Under-The-Hood
    try:
        encoded_route = encoders['route'].transform([route_selection])[0]
        encoded_day = encoders['day'].transform([day_selection])[0]
        encoded_weather = encoders['weather'].transform([weather_selection])[0]
    except:
        encoded_route = list(dropdown_routes).index(route_selection)
        encoded_day = list(dropdown_days).index(day_selection)
        encoded_weather = list(dropdown_weather).index(weather_selection)
    
    features = np.array([[encoded_route, encoded_day, time_selection, encoded_weather]])
    
    # Model Executions
    predicted_count = int(reg_model.predict(features)[0])
    predicted_class = clf_model.predict(features)[0]
    
    # Metric Display Column Block
    st.subheader("📊 Live Predictive Analytics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Predicted Passenger Load", value=f"~{predicted_count} riders")
    with col2:
        st.metric(label="Calculated Crowd Class", value=predicted_class)
    with col3:
        # Business logic calculation indicator based on threshold mapping
        confidence_metric = "Optimal" if predicted_class == "Low" else "Nominal" if predicted_class == "Medium" else "Strained"
        st.metric(label="Route Capacity Status", value=confidence_metric)
        
    st.markdown("---")
    
    # Contextual Warning & Callouts Block Based on Classification Categories
    st.subheader("💡 Intelligent Route Advisory")
    
    # Check for core rush hours
    is_rush_hour = (8 <= time_selection <= 10) or (17 <= time_selection <= 19)
    
    if predicted_class == "High":
        st.error(f"🔴 **Peak Alert:** The platform flags severe congestion vectors on **{route_selection}** around **{time_selection}:00**.")
        if is_rush_hour:
            st.info("🕒 **Alternative Strategy:** This window is a verified daily workspace shift rush hour. Shifting departure window backward or forward by 60 minutes is highly recommended.")
            
    elif predicted_class == "Medium":
        st.warning(f"🟡 **Moderate Volume Detected:** Steady passenger velocity observed on **{day_selection}**.")
        st.write("💺 **Seating Availability:** Standee crowds likely; priority seating access will be dense but active.")
        
    else:
        st.success(f"🟢 **Clear Transit Corridors:** Seamless, low-occupancy window confirmed.")
        st.write("🎯 **Optimal Commute Selection:** Unrestricted physical spacing available across the vehicle profile.")

else:
    # Ground Welcome Screen Placeholder
    st.info("👉 Use the sidebar menu on the left to set journey constraints and press **Run Predictive Inference** to generate layout metrics.")
