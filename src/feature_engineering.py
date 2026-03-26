import pandas as pd

# load cleaned dataset
df = pd.read_csv("data/storage_usage_cleaned.csv")

# convert time
df["event_time"] = pd.to_datetime(df["event_time"])

# sort
df = df.sort_values("event_time")

# --------------------------
# SEASONAL FEATURES
# --------------------------
df["month"] = df["event_time"].dt.month
df["day_of_week"] = df["event_time"].dt.dayofweek
df["quarter"] = df["event_time"].dt.quarter

# --------------------------
# LAG FEATURES
# --------------------------
df["lag_1"] = df["data_volume_tb"].shift(1)
df["lag_7"] = df["data_volume_tb"].shift(7)

# --------------------------
# ROLLING TREND FEATURES
# --------------------------
df["rolling_mean_7"] = df["data_volume_tb"].rolling(window=7).mean()
df["rolling_mean_30"] = df["data_volume_tb"].rolling(window=30).mean()

# --------------------------
# USAGE SPIKE DETECTION
# --------------------------
threshold = df["data_volume_tb"].mean() + df["data_volume_tb"].std()
df["spike_flag"] = df["data_volume_tb"].apply(lambda x: 1 if x > threshold else 0)

# save final dataset
df.to_csv("data/storage_featured_dataset.csv", index=False)

print("Feature engineered dataset ready")