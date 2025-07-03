import streamlit as st
import pandas as pd
import joblib
import requests

# -------------------------------
# Load model locally OR via API
# -------------------------------
USE_API = False  # Set to True if using FastAPI service

MODEL_PATH = "../models/insurance_model.pkl"
API_URL = "http://localhost:8000/predict"

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="AutoRisk+ Dashboard",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 AutoRisk+ Insurance Pricing Dashboard")
st.write("Predict claim cost based on policyholder data.")

# -------------------------------
# User Input Form
# -------------------------------
with st.form("input_form"):
    st.subheader("Enter Driver & Vehicle Information")

    driver_age = st.slider("Driver Age", 18, 90, 40)
    vehicle_age = st.slider("Vehicle Age (years)", 0, 20, 5)
    annual_mileage = st.number_input("Annual Mileage (km)", min_value=1000, value=15000)
    accident_history = st.selectbox("Past Accidents", [0, 1, 2, 3])
    location_risk_score = st.slider("Location Risk Score", 0.0, 1.0, 0.5)
    policy_duration_years = st.slider("Policy Duration (years)", 1, 10, 3)
    previous_claims = st.selectbox("Previous Claims", [0, 1, 2, 3])
    credit_score = st.slider("Credit Score", 300, 850, 700)

    vehicle_type = st.selectbox("Vehicle Type", ["Sedan", "SUV", "Truck", "Hatchback", "Convertible"])

    submit = st.form_submit_button("Predict Claim Cost")

# -------------------------------
# Helper: Format Input
# -------------------------------
def get_input_dict():
    vehicle_dict = {
        "vehicle_type_SUV": int(vehicle_type == "SUV"),
        "vehicle_type_Truck": int(vehicle_type == "Truck"),
        "vehicle_type_Hatchback": int(vehicle_type == "Hatchback"),
        "vehicle_type_Convertible": int(vehicle_type == "Convertible")
    }

    return {
        "driver_age": driver_age,
        "vehicle_age": vehicle_age,
        "annual_mileage": annual_mileage,
        "accident_history": accident_history,
        "location_risk_score": location_risk_score,
        "policy_duration_years": policy_duration_years,
        "previous_claims": previous_claims,
        "credit_score": credit_score,
        **vehicle_dict
    }

# -------------------------------
# Prediction Logic
# -------------------------------
def predict_claim_cost(inputs):
    if USE_API:
        response = requests.post(API_URL, json=inputs)
        if response.status_code == 200:
            return response.json()["predicted_claim_cost"]
        else:
            st.error("API error: " + response.text)
            return None
    else:
        model = joblib.load(MODEL_PATH)
        df = pd.DataFrame([inputs])
        return model.predict(df)[0]

# -------------------------------
# Output
# -------------------------------
if submit:
    input_data = get_input_dict()
    prediction = predict_claim_cost(input_data)

    if prediction is not None:
        st.success(f"💰 Predicted Claim Cost: **${prediction:,.2f}**")
        st.markdown("✅ Based on the input data, this is the estimated claim amount.")
