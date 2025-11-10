from typing import Any, Dict
from pydantic import BaseModel, Field


class ParseRequest(BaseModel):
    html: str = Field(..., description="Raw HTML content")
    query: str = Field(..., description="Natural language extraction instruction")


class ParseResponse(BaseModel):
    data: Any
    meta: Dict[str, Any]


