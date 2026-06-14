import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Public Transport Crowding Predictor", page_icon="🚌", layout="centered")

# Load your validated backend models and encoders
with open('transit_regressor.pkl', 'rb') as f:
    reg_model = pickle.load(f)
with open('transit_classifier.pkl', 'rb') as f:
    clf_model = pickle.load(f)
with open('transit_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

st.title("🚌 Public Transport Crowding Predictor")
st.write("Analyze and predict transit vehicle congestion trends to optimize your commute itinerary.")
st.markdown("---")

st.subheader("📋 Enter Journey Specifications")
col1, col2 = st.columns(2)

with col1:
    route_selection = st.selectbox("Select Route Number", encoders['route'].classes_)
    day_selection = st.selectbox("Day of the Week", encoders['day'].classes_)

with col2:
    time_selection = st.slider("Time of Travel (24hr format)", 6, 22, 9)
    weather_selection = st.selectbox("Current Weather Condition", encoders['weather'].classes_)

if st.button("Predict Congestion Metric", type="primary"):
    # Encode user selection values using the saved label maps
    encoded_route = encoders['route'].transform([route_selection])[0]
    encoded_day = encoders['day'].transform([day_selection])[0]
    encoded_weather = encoders['weather'].transform([weather_selection])[0]
    
    features = np.array([[encoded_route, encoded_day, time_selection, encoded_weather]])
    
    # Run structural predictions
    predicted_count = int(reg_model.predict(features)[0])
    predicted_class = clf_model.predict(features)[0]
    
    st.markdown("---")
    st.subheader("📊 Analytical Results")
    
    if predicted_class == "High":
        st.error(f"🚨 **Estimated Crowd Density Level:** {predicted_class}")
        st.write(f"📈 **Approximate Headcount Forecast:** ~{predicted_count} passengers currently on-board.")
    elif predicted_class == "Medium":
        st.warning(f"⚠️ **Estimated Crowd Density Level:** {predicted_class}")
        st.write(f"📈 **Approximate Headcount Forecast:** ~{predicted_count} passengers currently on-board.")
    else:
        st.success(f"✅ **Estimated Crowd Density Level:** {predicted_class}")
        st.write(f"📈 **Approximate Headcount Forecast:** ~{predicted_count} passengers currently on-board.")
