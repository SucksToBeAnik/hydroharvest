import uuid
from sqlmodel import Field, SQLModel
from uuid import UUID


class User(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid.uuid4, primary_key=True, unique=True)
    name: str