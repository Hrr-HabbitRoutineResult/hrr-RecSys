from pydantic import BaseModel, Field
from typing import List, Optional

class ChallengeItemIn(BaseModel):
    challengeId: int
    title: str
    description: Optional[str] = None
    category: Optional[str] = None           
    cert_time_slots: Optional[str] = None    
    goal_text: Optional[str] = None          
    embedding: List[float]                   # DB에서 꺼낸 normalized 벡터

class RecommendRequest(BaseModel):
    # 백엔드는 여기로 "query: 20대 직장인, ..." 형태로 넘김
    query: str = Field(..., description='Must start with "query:" (E5 스타일 권장)')
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
