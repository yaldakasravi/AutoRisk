from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Define the expected input data model
class InsuranceFeatures(BaseModel):
    driver_age: int
    vehicle_age: int
    annual_mileage: int
    accident_history: int
    location_risk_score: float
    policy_duration_years: int
    previous_claims: int
    credit_score: int
    vehicle_type_SUV: int = 0
    vehicle_type_Truck: int = 0
    vehicle_type_Hatchback: int = 0
    vehicle_type_Convertible: int = 0

# Create the FastAPI app
app = FastAPI(
    title="AutoRisk+ ML Insurance Pricing API",
    description="""
    Predict auto insurance claim cost based on customer and vehicle data.
    This API is part of the AutoRisk+ project.
    """,
    version="1.0.0"
)

# Path to trained model
MODEL_PATH = "models/insurance_model.pkl"
model = None

# Load model at startup
@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return {"message": "✅ AutoRisk+ API is live. Use /predict to get insurance claim cost predictions."}

@app.post("/predict")
def predict(data: InsuranceFeatures):
    try:
        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)[0]
        return {"predicted_claim_cost": round(prediction, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
