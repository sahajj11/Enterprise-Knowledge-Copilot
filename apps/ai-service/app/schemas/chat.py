from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    message: str = Field(strict=True, min_length=1, max_length=10000)


class ChatResponse(BaseModel):
    response: str
