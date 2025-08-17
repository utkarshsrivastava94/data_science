import os
import logging
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from datetime import datetime
from pydantic import BaseModel

# Create logs directory if needed
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# Load the trained model
model = joblib.load("models/random_forest_load_forecaster.pkl")

# Create the FastAPI app
app = FastAPI(
    title="Electricity Load Forecasting API",
    description="Predicts power demand based on time and weather features.",
    version="1.0.0"
)

# Define input schema
class LoadForecastInput(BaseModel):
    week_X_2: float
    week_X_3: float
    week_X_4: float
    MA_X_4: float
    dayOfWeek: int
    weekend: int
    holiday: int
    Holiday_ID: int
    hourOfDay: int
    T2M_toc: float

@app.post("/predict")
def predict_demand(data: LoadForecastInput):
    try:
        input_df = pd.DataFrame([[
            data.week_X_2, data.week_X_3, data.week_X_4, data.MA_X_4,
            data.dayOfWeek, data.weekend, data.holiday, data.Holiday_ID,
            data.hourOfDay, data.T2M_toc
        ]], columns=[
            "week_X-2", "week_X-3", "week_X-4", "MA_X-4",
            "dayOfWeek", "weekend", "holiday", "Holiday_ID",
            "hourOfDay", "T2M_toc"
        ])

        prediction = model.predict(input_df)[0]
        result = round(prediction, 2)

        # Log the input and prediction
        logging.info(f"Input: {data.model_dump()} | Prediction: {result}")
        return {"predicted_DEMAND": result}

    except Exception as e:
        logging.error(f"Prediction error: {str(e)} | Input: {data.model_dump()}")
        return {"error": "Something went wrong. Check logs."}
