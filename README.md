# Azure Storage Demand Forecasting 

## Overview
This project focuses on preparing historical cloud storage demand data
for South India and Asian regions between 2022 and 2024.

## MileStone-1

The objective of Milestone 1 is to collect, generate, and preprocess
time-series storage usage data for future demand forecasting models.

# Work Completed
- Generated synthetic historical storage dataset (~5200 records)
- Included regional and service-level attributes
- Introduced missing values and inconsistencies
- Cleaned and standardized the dataset
- Prepared model-ready data for forecasting

# Tools Used
Python, Pandas, NumPy, VS Code, GitHub

## Milestone 2 – Feature Engineering

# Overview
The objective of Milestone 2 is to Prepare the dataset for modeling through enrichment and transformation .

# Work Completed 
- Created seasonality indicators from timestamp
- Generated lag features for demand forecasting
- Computed rolling averages for trend detection
- Identified usage spikes using statistical thresholds
- Prepared model-ready dataset for forecasting

# Tools used
Python, Pandas, NumPy, VS Code, GitHub.

## Milestone 3 – Machine Learning Model Development
 
# Overview
In this milestone, multiple machine learning models were trained on the
feature-engineered dataset to predict Azure storage demand.

# Work Completed
Models Implemented:
- Linear Regression
- XGBoost Regressor
- ARIMA Time Series Model

Evaluation Metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

The models were compared using historical demand data, and the best performing
model was selected based on empirical accuracy.

# Tools Used
pandas ,numpy , scikit-learn , statsmodels , xgboost , matplotlib

## Hyperparameter tuning was performed using GridSearchCV to optimize the XGBoost model parameters for improved forecasting accuracy.

## Model Deployment Preparation
The final tuned XGBoost model is saved in the models directory.
A prediction script was implemented to generate demand forecasts using the trained model .

Files:
models/xgboost_demand_model.pkl
src/predict_demand.py

## Installation
Clone the repository:
git clone https://github.com/YOUR_USERNAME/Demand_Forecasting1.git
Install required dependencies:
pip install -r requirements.txt

## Running the Project

- Step 1 – Data Preprocessing
python src/data_preprocessing.py
- Step 2 – Feature Engineering
python src/feature_engineering.py
- Step 3 – Train Machine Learning Models
python src/model_training.py
- Step 4 – Generate Predictions
python src/predict_demand.py

## Milestone 4: Forecast Integration & Deployment

### Features Implemented

- Real-time prediction API using FastAPI
- Batch prediction pipeline
- Interactive dashboard (HTML + Chart.js)
- Automated scheduling using Python scheduler
- Model performance monitoring (RMSE tracking)
- Retraining pipeline for continuous improvement

### How to Run

1. Start API:
   uvicorn src.api:app --reload

2. Run batch prediction:
   python src/batch_predict.py

3. Open dashboard:
   python -m http.server 8001
   Open http://127.0.0.1:8001/dashboard.html

4. Run monitoring:
   python src/monitor.py