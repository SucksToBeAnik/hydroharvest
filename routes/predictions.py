from typing import Optional
from fastapi import APIRouter, HTTPException
from utils import get_buffered_bounding_box
from ml import predict_precipitation

router = APIRouter(prefix="/predictions", tags=["Prediction"])

# Assuming get_buffered_bounding_box and predict_precipitation are defined elsewhere


@router.post("/precipitation")
async def get_user_location_precipitation(
    latitude: float, longitude: float, radius: Optional[float] = None
):
    # Step 1: Get the buffered bounding box coordinates
    try:
        # Step 1: Get the buffered bounding box coordinates
        radius = radius or 1.0

        coordinates = get_buffered_bounding_box(
            lat=latitude, lon=longitude, buffer_km=radius
        )

        # Step 2: Call the predict_precipitation function with the extracted coordinates
        prediction = predict_precipitation(
            sw_lon=coordinates["sw_lon"],
            sw_lat=coordinates["sw_lat"],
            ne_lon=coordinates["ne_lon"],
            ne_lat=coordinates["ne_lat"],
            nw_lon=coordinates["nw_lon"],
            nw_lat=coordinates["nw_lat"],
            se_lon=coordinates["se_lon"],
            se_lat=coordinates["se_lat"],
        )

        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
