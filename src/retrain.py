import pandas as pd
from xgboost import XGBRegressor
import joblib
import numpy as np

df = pd.read_csv("data/storage_featured_dataset.csv")

# Remove invalid rows
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# Split
X = df.drop("data_volume_tb", axis=1)
y = df["data_volume_tb"]

# Encode categorical
X = pd.get_dummies(X)

# Train
model = XGBRegressor()
model.fit(X, y)

# Save
joblib.dump(model, "models/xgboost_demand_model.pkl")
joblib.dump(X.columns.tolist(), "models/columns.pkl")

print("✅ Model retrained successfully")