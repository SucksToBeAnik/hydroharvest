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

    return model, scaler, feature_columns


import logging

logger = logging.getLogger(__name__)
#  Global variables to store loaded model, scaler, and feature columns
MODEL = None
SCALER = None
FEATURE_COLUMNS = None


def load_model_and_scaler():
    global MODEL, SCALER, FEATURE_COLUMNS
    if MODEL is None or SCALER is None or FEATURE_COLUMNS is None:
        try:
            logger.info("Loading model, scaler, and feature columns")
            MODEL = joblib.load("model.joblib")
            SCALER = joblib.load("scaler.joblib")
            FEATURE_COLUMNS = joblib.load("feature_columns.joblib")
            logger.info("Successfully loaded model, scaler, and feature columns")
        except Exception as e:
            logger.error(f"Error loading model, scaler, or feature columns: {str(e)}")
            raise


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
    try:
        logger.info("Starting precipitation prediction")

        # Ensure model, scaler, and feature columns are loaded
        load_model_and_scaler()

        day = 1
        current_date = datetime.now()
        month = month or current_date.month
        year = year or current_date.year

        logger.info(f"Preparing input data for year: {year}, month: {month}")

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
            columns=FEATURE_COLUMNS,
        )

        logger.info("Scaling input features")
        input_scaled = pd.DataFrame(
            SCALER.transform(input_data), columns=FEATURE_COLUMNS
        )

        logger.info("Making prediction")
        prediction: float = MODEL.predict(input_scaled)[0]

        logger.info(f"Prediction completed. Result: {prediction}")

        return PredictionReturnType(
            hourly_rainfall_in_mm=prediction,
            month=month,
            year=year,
        )
    except Exception as e:
        logger.error(f"Error in predict_precipitation: {str(e)}")
        raise
