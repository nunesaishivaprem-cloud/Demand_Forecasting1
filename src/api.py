from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load model once
model = joblib.load("models/xgboost_demand_model.pkl")

@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])

        # Convert categorical → one-hot encoding
        df = pd.get_dummies(df)

        # Load training columns
        train_cols = joblib.load("models/columns.pkl")

        # Align columns (VERY IMPORTANT)
        df = df.reindex(columns=train_cols, fill_value=0)

        prediction = model.predict(df)

        return {"forecast": prediction.tolist()}

    except Exception as e:
        return {"error": str(e)}