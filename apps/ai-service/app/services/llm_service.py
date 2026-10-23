import logging

import httpx
from google import genai
from google.genai import errors, types

from app.core.config import Settings

logger = logging.getLogger(__name__)


class LLMServiceError(Exception):
    """A safe error message and HTTP status for the API boundary."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def generate_response(message: str, settings: Settings) -> str:
    if not settings.gemini_api_key or not settings.gemini_model:
        raise LLMServiceError(503, "Chat service is not configured.")

    try:
        # A sync FastAPI route runs this blocking call in its worker thread pool.
        with genai.Client(
            api_key=settings.gemini_api_key,
            vertexai=False,
            http_options=types.HttpOptions(timeout=30000),
        ) as client:
            result = client.models.generate_content(
                model=settings.gemini_model,
                contents=message,
            )
            text = result.text
    except (httpx.TimeoutException, TimeoutError):
        raise LLMServiceError(504, "Gemini request timed out.") from None
    except errors.APIError as exc:
        if exc.code == 429:
            raise LLMServiceError(503, "Gemini is temporarily unavailable. Try again later.") from None
        raise LLMServiceError(502, "Gemini could not complete the request.") from None
    except httpx.RequestError:
        raise LLMServiceError(502, "Unable to reach Gemini.") from None
    except Exception:
        # Do not log SDK exceptions: they can contain credentials or prompt data.
        logger.error("Unexpected failure while calling Gemini.")
        raise LLMServiceError(502, "Gemini could not complete the request.") from None

    if not isinstance(text, str) or not text.strip():
        raise LLMServiceError(502, "Gemini returned no text response.")
    return text.strip()
