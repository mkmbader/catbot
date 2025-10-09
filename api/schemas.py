"""Schemas for fastAPI endpoints."""
from pydantic import BaseModel

class ChatRequest(BaseModel):
    """
    Model for the incoming chat query.
    """
    query: str

class ChatResponse(BaseModel):
    """
    Model for the outgoing chat response.
    """
    answer: str
    sources: List[str] = []