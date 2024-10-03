from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


env_store = Settings() # type: ignore