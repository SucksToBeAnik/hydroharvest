from fastapi import FastAPI, HTTPException, status,Depends
from sqlmodel import select, Session
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from routes import predictions
from db.config import get_db_connection, init_db
from db.models import User
from typing import Annotated, Optional
from ml import predict_precipitation, train_model
from utils import get_buffered_bounding_box
import logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("---app starting---")
    init_db()
    train_model()
    yield
    print("---app closing---")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predictions.router)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.get("/")
async def root():
    return "Welcome to the root of HydroHarvest"



@app.post("/users/{name}")
async def add_user(name: str, db: Annotated[Session, Depends(get_db_connection)]):
    try:
        new_user = User(name=name)
        print(new_user)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        return new_user
    except:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Something went wrong when creating the user",
        )


@app.get("/users")
async def get_users(db: Annotated[Session, Depends(get_db_connection)]):
    try:
        statement = select(User)
        results = db.exec(statement)

        return results.all()
    except:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Something went wrong when fetching the users",
        )
    
@app.post("/precipitation")
async def get_user_location_precipitation(
    latitude: float, longitude: float, radius: Optional[float] = None
):
    try:
        # Step 1: Get the buffered bounding box coordinates
        radius = radius or 1.0
        logger.info(f"Calculating bounding box for lat:{latitude}, lon:{longitude}, radius:{radius}")

        coordinates = get_buffered_bounding_box(
            lat=latitude, lon=longitude, buffer_km=radius
        )

        logger.info(f"Calculated coordinates: {coordinates}")

        # Step 2: Extract the coordinates from the dictionary
        ne_lat, ne_lon = coordinates.get("ne_lat"), coordinates.get("ne_lon")
        nw_lat, nw_lon = coordinates.get("nw_lat"), coordinates.get("nw_lon")
        sw_lat, sw_lon = coordinates.get("sw_lat"), coordinates.get("sw_lon")
        se_lat, se_lon = coordinates.get("se_lat"), coordinates.get("se_lon")

        # Step 3: Call the predict_precipitation function with the extracted coordinates
        logger.info("Calling predict_precipitation function")
        prediction = predict_precipitation(
            ne_lat=ne_lat, ne_lon=ne_lon,
            nw_lat=nw_lat, nw_lon=nw_lon,
            sw_lat=sw_lat, sw_lon=sw_lon,
            se_lat=se_lat, se_lon=se_lon,
        )

        logger.info(f"Prediction result: {prediction}")

        # Step 4: Return the predicted precipitation
        return {"prediction": prediction}

    except Exception as e:
        logger.error(f"Error in precipitation prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")