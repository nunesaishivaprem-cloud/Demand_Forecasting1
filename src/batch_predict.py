import pandas as pd
import joblib

# Load model
model = joblib.load("models/xgboost_demand_model.pkl")

# Load dataset
df = pd.read_csv("data/storage_featured_dataset.csv")

# Save original column for comparison
actual = df["data_volume_tb"]

# Convert categorical
df_encoded = pd.get_dummies(df)

# Load training columns
train_cols = joblib.load("models/columns.pkl")

# Align columns
df_encoded = df_encoded.reindex(columns=train_cols, fill_value=0)

# Predict
predictions = model.predict(df_encoded)

# Add forecast column
df["forecast"] = predictions

# Save output
df.to_csv("data/forecast_output.csv", index=False)

print("✅ Batch prediction completed!")