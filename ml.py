import joblib
from datetime import datetime
import pandas as pd
from typing import Optional, TypedDict


class PredictionReturnType(TypedDict):
    hourly_rainfall_in_mm: float
    month: int
    year: int


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
    prediction:float = model.predict(input_scaled)[0]

    return PredictionReturnType(
        hourly_rainfall_in_mm=prediction,
        month=month,
        year=year,
    )
