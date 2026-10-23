import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = field(default="", repr=False)
    gemini_model: str = ""


@lru_cache
def get_settings() -> Settings:
    # Explicit environment variables take precedence over the service's .env.
    load_dotenv(ENV_FILE, override=False)
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "").strip(),
    )
