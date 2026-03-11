import pandas as pd
import joblib

# load model
model = joblib.load("models/xgboost_demand_model.pkl")

# load dataset
df = pd.read_csv("data/storage_featured_dataset.csv")

# remove columns not used
X = df.drop(["event_time", "data_volume_tb"], axis=1)

X = pd.get_dummies(X)

# predict demand
predictions = model.predict(X)

df["predicted_demand"] = predictions

df.to_csv("data/demand_predictions.csv", index=False)

print("Predictions saved to data/demand_predictions.csv")