from fastapi import FastAPI, HTTPException, status,Depends
from sqlmodel import select, Session
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from routes import predictions
from db.config import get_db_connection, init_db
from db.models import User
from typing import Annotated
from ml import train_model


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