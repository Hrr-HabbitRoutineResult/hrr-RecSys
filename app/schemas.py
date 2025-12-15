from pydantic import BaseModel, Field
from typing import List, Optional
from pydantic import BaseModel
from typing import List

class ChallengeItemIn(BaseModel):
    challengeId: int
    title: str
    description: Optional[str] = None
    category: Optional[str] = None           
    cert_time_slots: Optional[str] = None    
    goal_text: Optional[str] = None          
    embedding: List[float]

class RecommendRequest(BaseModel):
    query: str = Field(..., description='Must start with "query:"')
    items: List[ChallengeItemIn]
    top_k: int = 5

class RecommendItemOut(BaseModel):
    challengeId: int
    title: str
    description: Optional[str] = None
    matchScore: float

class RecommendResponse(BaseModel):
    recommendations: List[RecommendItemOut]
    modelVersion: str
    latencyMs: int

class HealthResponse(BaseModel):
    ok: bool
    modelLoaded: bool
    modelVersion: str

class EmbeddingRequest(BaseModel):
    text: str

class EmbeddingResponse(BaseModel):
    embedding: List[float]