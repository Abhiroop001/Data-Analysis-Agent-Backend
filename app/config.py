from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    scaledown_api_key: str | None = None
    host: str = "127.0.0.1"
    port: int = 8000

    class Config:
        env_file = ".env"

settings = Settings()
