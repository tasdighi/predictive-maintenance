from fastapi import FastAPI
import pandas as pd
import joblib
from src.config import Config

app = FastAPI(title="Predictive Maintenance Service")

# Load the trained pipeline (it handles scaling automatically)
try:
    model = joblib.load(Config.MODEL_SAVE_PATH)
except:
    model = None

@app.post("/predict")
async def predict(features: dict):
    """
    Expects a JSON dictionary of features.
    Example: {"Air temperature K": 300, "Process temperature K": 310, ...}
    """
    if model is None:
        return {"error": "Model not found. Please train the model first."}
    
    # Convert input JSON to DataFrame for the pipeline
    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df).tolist()[0]
    
    return {
        "failure_predicted": bool(prediction),
        "confidence_scores": {"healthy": probability[0], "failure": probability[1]}
    }