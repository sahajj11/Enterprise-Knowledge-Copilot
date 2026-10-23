from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException

from app.core.config import Settings, get_settings
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_service import LLMServiceError, generate_response

app = FastAPI(
    title="Enterprise Knowledge Copilot AI Service",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "ai-service"}


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ChatResponse:
    try:
        return ChatResponse(response=generate_response(request.message, settings))
    except LLMServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from None
