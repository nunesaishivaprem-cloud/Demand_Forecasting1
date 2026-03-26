import joblib
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

# load dataset
df = pd.read_csv("data/storage_featured_dataset.csv")

# convert time
df["event_time"] = pd.to_datetime(df["event_time"])

# drop rows with lag nulls
df = df.dropna()

# target variable
y = df["data_volume_tb"]

# features
X = df.drop(["event_time", "data_volume_tb"], axis=1)

# encode categorical features
X = pd.get_dummies(X)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print("Training size:", X_train.shape)

# -----------------------------
# Model 1 : Linear Regression
# -----------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)

mae_lr = mean_absolute_error(y_test, pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, pred_lr))

print("\nLinear Regression")
print("MAE:", mae_lr)
print("RMSE:", rmse_lr)

# -----------------------------
# Model 2 : XGBoost with Hyperparameter Tuning
# -----------------------------

xgb_model = XGBRegressor()

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [3, 6],
    "learning_rate": [0.05, 0.1]
}

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_absolute_error"
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

pred_xgb = best_model.predict(X_test)

mae_xgb = mean_absolute_error(y_test, pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, pred_xgb))

print("\nXGBoost (Tuned)")
print("Best Parameters:", grid_search.best_params_)
print("MAE:", mae_xgb)
print("RMSE:", rmse_xgb)

# Save the best model
joblib.dump(best_model, "models/xgboost_demand_model.pkl")

print("Model saved in models/xgboost_demand_model.pkl")

# -----------------------------
# Model 3 : ARIMA
# -----------------------------
series = df["data_volume_tb"]

model = ARIMA(series, order=(5,1,0))
model_fit = model.fit()

forecast = model_fit.forecast(steps=len(y_test))

mae_arima = mean_absolute_error(y_test, forecast)
rmse_arima = np.sqrt(mean_squared_error(y_test, forecast))

print("\nARIMA")
print("MAE:", mae_arima)
print("RMSE:", rmse_arima)

print("\nModel comparison completed")

import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual Demand")
plt.plot(pred_xgb, label="Predicted Demand (XGBoost)")
plt.title("Storage Demand Forecast")
plt.xlabel("Time")
plt.ylabel("Storage Usage (TB)")
plt.legend()

plt.savefig("data/demand_forecast_plot.png")
plt.show()

# -----------------------------
# Save Model Comparison
# -----------------------------

results = pd.DataFrame({
    "Model": ["Linear Regression", "XGBoost", "ARIMA"],
    "MAE": [mae_lr, mae_xgb, mae_arima],
    "RMSE": [rmse_lr, rmse_xgb, rmse_arima]
})

results.to_csv("data/model_comparison_results.csv", index=False)
print("\nModel comparison results saved.")

# Save column structure 
joblib.dump(X.columns.tolist(), "models/columns.pkl")