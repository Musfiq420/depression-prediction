import streamlit as st
import numpy as np
import joblib

# Load the model and scaler
model = joblib.load("depression_detector_xgb.pkl")
scaler = joblib.load("depression_scaler.pkl")

st.set_page_config(page_title="Depression Predictor", layout="centered")

st.title("🧠 Depression Prediction App")
st.write("Fill out the details below to check the likelihood of depression.")

# Input form
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=10, max_value=100, value=30)

work_pressure = st.slider("Work Pressure (1 = Low, 5 = High)", 1.0, 5.0, step=0.5)

job_satisfaction = st.slider("Job Satisfaction (1 = Low, 5 = High)", 1.0, 5.0, step=0.5)

sleep_duration = st.selectbox("Sleep Duration", [
    "Less than 5 hours",
    "5-6 hours",
    "7-8 hours",
    "More than 8 hours"
])

dietary_habits = st.selectbox("Dietary Habits", ["Unhealthy", "Moderate", "Healthy"])

suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["Yes", "No"])

work_hours = st.number_input("Work Hours Per Day", min_value=0, max_value=24, value=8)

financial_stress = st.slider("Financial Stress (0 = None, 5 = High)", 0, 5, value=2)

family_history = st.selectbox("Family History of Mental Illness", ["Yes", "No"])


# Convert inputs to model format
gender_val = 1 if gender == "Male" else 0
sleep_map = {
    "Less than 5 hours": 2.5,
    "5-6 hours": 5,
    "7-8 hours": 7.5,
    "More than 8 hours": 10
}
diet_map = {
    "Unhealthy": 0,
    "Moderate": 5,
    "Healthy": 10
}
suicidal_val = 1 if suicidal_thoughts == "Yes" else 0
family_val = 1 if family_history == "Yes" else 0

input_data = np.array([[
    gender_val,
    age,
    work_pressure,
    job_satisfaction,
    sleep_map[sleep_duration],
    diet_map[dietary_habits],
    suicidal_val,
    work_hours,
    financial_stress,
    family_val
]])

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("⚠️ High Risk of Depression")
    else:
        st.success("✅ Low Risk of Depression")
