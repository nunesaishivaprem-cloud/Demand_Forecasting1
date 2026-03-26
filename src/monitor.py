import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

df = pd.read_csv("data/forecast_output.csv")

rmse = np.sqrt(mean_squared_error(df["data_volume_tb"], df["forecast"]))

print("RMSE:", rmse)

if rmse > 100:
    print("⚠ ALERT: Model performance degraded!")
else:
    print("✅ Model performance is good")