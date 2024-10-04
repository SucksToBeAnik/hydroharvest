from typing import Optional
from fastapi import APIRouter
from utils import get_buffered_bounding_box
from ml import predict_precipitation

router = APIRouter(
    prefix='/predictions',
    tags=['Prediction']
)

# Assuming get_buffered_bounding_box and predict_precipitation are defined elsewhere


@router.post("/precipitation")
async def get_user_location_precipitation(
    latitude: float, longitude: float, radius: Optional[float] = None
):
    # Step 1: Get the buffered bounding box coordinates
    radius = radius or 1.0

    coordinates = get_buffered_bounding_box(
        lat=latitude, lon=longitude, buffer_km=radius
    )

    # Step 2: Extract the coordinates from the dictionary
    ne_lat = coordinates.get("ne_lat")
    ne_lon = coordinates.get("ne_lon")
    nw_lat = coordinates.get("nw_lat")
    nw_lon = coordinates.get("nw_lon")
    sw_lat = coordinates.get("sw_lat")
    sw_lon = coordinates.get("sw_lon")
    se_lat = coordinates.get("se_lat")
    se_lon = coordinates.get("se_lon")

    # Step 3: Call the predict_precipitation function with the extracted coordinates
    prediction = predict_precipitation(
        ne_lat=ne_lat,
        ne_lon=ne_lon,
        nw_lat=nw_lat,
        nw_lon=nw_lon,
        sw_lat=sw_lat,
        sw_lon=sw_lon,
        se_lat=se_lat,
        se_lon=se_lon,
    )

    # Step 4: Return the predicted precipitation (or whatever your function returns)
    return prediction
