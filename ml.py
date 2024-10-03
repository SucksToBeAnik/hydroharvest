from typing import Optional, TypedDict
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime


class PredictionReturnType(TypedDict):
    hourly_rainfall_in_mm: float
    month: int
    year: int


def train_model() -> tuple[RandomForestRegressor, StandardScaler, list[str]]:
    df = pd.read_csv("./precipitation_data.csv")
    df["system:time_start"] = pd.to_datetime(df["system:time_start"])

    # Create more informative features
    df["year"] = df["system:time_start"].dt.year
    df["month"] = df["system:time_start"].dt.month
    df["day"] = df["system:time_start"].dt.day

    # Prepare the data
    feature_columns = [
        "year",
        "month",
        "day",
        "SW_Lon",
        "SW_Lat",
        "NE_Lon",
        "NE_Lat",
        "NW_Lon",
        "NW_Lat",
        "SE_Lon",
        "SE_Lat",
    ]
    X = df[feature_columns]
    y = df["precipitation"]

    X_train: pd.DataFrame
    X_test: pd.DataFrame

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        data=scaler.fit_transform(X_train), columns=feature_columns
    )
    # X_test_scaled = pd.DataFrame(data=scaler.transform(X_test), columns=feature_columns)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Save the model, scaler, and feature columns
    joblib.dump(model, "model.joblib")
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(feature_columns, "feature_columns.joblib")

    # print("Model, scaler, and feature columns saved.")

    # # Evaluate the model
    # y_pred_train = model.predict(X_train_scaled)
    # y_pred_test = model.predict(X_test_scaled)

    # # Calculate accuracy metrics
    # train_r2 = r2_score(y_train, y_pred_train)
    # test_r2 = r2_score(y_test, y_pred_test)
    # test_mae = mean_absolute_error(y_test, y_pred_test)
    # test_mse = mean_squared_error(y_test, y_pred_test)

    # print(f"Train R2: {train_r2:.4f}")
    # print(f"Test R2: {test_r2:.4f}")
    # print(f"Test MAE: {test_mae:.4f}")
    # print(f"Test MSE: {test_mse:.4f}")

    return model, scaler, feature_columns


def predict_precipitation(
    sw_lon: float,
    sw_lat: float,
    ne_lon: float,
    ne_lat: float,
    nw_lon: float,
    nw_lat: float,
    se_lon: float,
    se_lat: float,
    month: Optional[int] = None,
    year: Optional[int] = None,
) -> PredictionReturnType:
    # Load the model, scaler, and feature columns
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    feature_columns = joblib.load("feature_columns.joblib")
    day = 1

    current_date = datetime.now()
    if not month:
        month = current_date.month
    if not year:
        year = current_date.year

    # Prepare input features
    input_data = pd.DataFrame(
        [
            [
                year,
                month,
                day,
                sw_lon,
                sw_lat,
                ne_lon,
                ne_lat,
                nw_lon,
                nw_lat,
                se_lon,
                se_lat,
            ]
        ],
        columns=feature_columns,
    )
    # Scale the input features
    input_scaled = pd.DataFrame(scaler.transform(input_data), columns=feature_columns)

    # Make prediction
    prediction: float = model.predict(input_scaled)[0]

    return PredictionReturnType(
        hourly_rainfall_in_mm=prediction,
        month=month,
        year=year,
    )
