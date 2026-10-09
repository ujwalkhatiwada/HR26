from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    agnes_api_key: str = "your_agnes_api_key_here"
    agnes_base_url: str = "https://apihub.agnes-ai.com/v1"
    agnes_model: str = "agnes-3.0-flash"
    agnes_image_model: str = "agnes-image-2.5-flash"

    app_env: str = "development"
    max_context_tokens: int = 480_000
    rate_limit_rpm: int = 10
    data_dir: Path = Path("./data")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
