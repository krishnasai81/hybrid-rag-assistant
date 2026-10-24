
from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Question about the indexed documents.",
    )


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    verification: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    status: str
    application: str


class UploadResponse(BaseModel):
    filename: str
    chunks_created: int
    message: str
    