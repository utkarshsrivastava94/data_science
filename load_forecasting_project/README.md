# Electricity Load Forecasting with FastAPI

This project implements an end-to-end machine learning pipeline for **hourly electricity load forecasting** using historical demand and weather data. It includes model training, saving, and serving real-time predictions via a FastAPI-based REST API.

---

## 1. Project Overview

### What This Project Does:

- **Data Source**: Uses a public Kaggle dataset: [`saurabhshahane/electricity-load-forecasting`](https://www.kaggle.com/datasets/saurabhshahane/electricity-load-forecasting)
- **Data Processing**:
  - Combines weekly train/test Excel files
  - Parses and aligns timestamps
- **Modeling**:
  - Trains a `RandomForestRegressor` using lagged and weather features
  - Achieves high accuracy (R² > 0.99)
- **Model Persistence**:
  - Saves trained model using `joblib`
- **API**:
  - Exposes a FastAPI endpoint `/predict` for real-time forecasting
  - Accepts feature input in JSON
  - Returns predicted electricity demand
- **Logging**:
  - Logs requests and predictions to a local `logs/` file

---

## 2. How to Use This Project

Follow these steps to replicate the process on your machine.

### Step 1: Install Dependencies

Create a virtual environment (recommended) and install packages:

```bash
python -m venv venv
.\venv\Scripts\activate    # Windows
# or
source venv/bin/activate  # Mac/Linux

pip install -r requirements.txt

```

### Step 2: Download the Dataset Programmatically

Instead of downloading files manually, use the helper library provided here to fetch the Kaggle dataset directly:

📦 [Download Kaggle Datasets – GitHub Library](https://github.com/utkarshsrivastava94/libraries)
Make sure your Kaggle API credentials (`kaggle.json`) are properly configured before running the script. If needed, refer to the library’s instructions.

#### Steps:

1. Clone or install the download library
2. Use it to programmatically fetch the dataset:
   - Kaggle dataset name: `saurabhshahane/electricity-load-forecasting`
3. This will create the `datasets/` folder automatically with all required files:
   - `continuous dataset.csv`
   - `train_ddataframes.xlsx`
   - `test_dataframes.xlsx`
   - `weekly pre dispatch forecast.csv`

### Step 3: Train the Model

Use the training code (from src/models/baseline_model.py) or the provided notebook to:

- Load and combine the weekly Excel sheets

- Apply feature engineering

- Train a RandomForestRegressor model

### Step 4: Start the FastAPI Server

Use uvicorn to launch the API locally:

```bash
uvicorn src.api.app:app --reload
```

Visit the interactive docs at: http://127.0.0.1:8000/docs

### Step 5: Test the API

Send a sample JSON input like this in the Swagger UI:

```json
{
  "week_X_2": 1050.0,
  "week_X_3": 1025.0,
  "week_X_4": 1000.0,
  "MA_X_4": 1025.0,
  "dayOfWeek": 1,
  "weekend": 0,
  "holiday": 0,
  "Holiday_ID": 0,
  "hourOfDay": 10,
  "T2M_toc": 27.5
}
```

You will receive a response like:

```json
{
  "predicted_DEMAND": 1132.45
}
```

All requests are logged to logs/predictions.log

# Author Notes

This project simulates a real-world, production-grade forecasting system and serves as a complete example of:

- Time series ML

- Feature engineering

- Model serving

- API logging and monitoring
