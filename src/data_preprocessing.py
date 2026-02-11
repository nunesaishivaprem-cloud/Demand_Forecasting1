import pandas as pd

# loading generated dataset
df = pd.read_csv("data/storage_usage_raw.csv")

print("records before cleaning:", df.shape)

# convert time column
df["event_time"] = pd.to_datetime(df["event_time"])

# sorting for time-series
df = df.sort_values("event_time")

# remove duplicate logs
df = df.drop_duplicates()

# interpolate missing storage values
df["data_volume_tb"] = df["data_volume_tb"].interpolate()

# derive missing billing from storage usage trend
df["billing_amount"] = df["billing_amount"].fillna(df["data_volume_tb"] * 1.8)

# fix uptime missing values
df["uptime_percent"] = df["uptime_percent"].ffill()

# festival column fix
df["festival_flag"] = df["festival_flag"].fillna(0)

print("records after cleaning:", df.shape)

df.to_csv("data/storage_usage_cleaned.csv", index=False)

print("clean dataset ready")
