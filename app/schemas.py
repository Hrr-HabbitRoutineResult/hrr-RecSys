from pydantic import BaseModel, Field
from typing import List, Optional
from pydantic import BaseModel
from typing import List

from typing import List, Optional
from pydantic import BaseModel, Field

class ChallengeItemIn(BaseModel):
    challengeId: int
    title: str
    description: Optional[str] = None
    category: Optional[str] = None
    cert_time_slots: Optional[str] = None
    goal_text: Optional[str] = None
    embedding: List[float]

    # ===== 참여자 분포(%) =====
    participants_male_pct: int = Field(0, ge=0, le=100)
    participants_female_pct: int = Field(0, ge=0, le=100)

    # ===== 연령 분포(%) =====
    age_10s_pct: int = Field(0, ge=0, le=100)
    age_20s_pct: int = Field(0, ge=0, le=100)
    age_30s_pct: int = Field(0, ge=0, le=100)
    age_40s_pct: int = Field(0, ge=0, le=100)
    age_50p_pct: int = Field(0, ge=0, le=100)

    # ===== 직업 분포(%) =====
    job_student_middle_high_pct: int = Field(0, ge=0, le=100)
    job_student_university_pct: int = Field(0, ge=0, le=100)
    job_job_seeker_pct: int = Field(0, ge=0, le=100)
    job_employee_pct: int = Field(0, ge=0, le=100)
    job_homemaker_pct: int = Field(0, ge=0, le=100)
    job_etc_pct: int = Field(0, ge=0, le=100)

class RecommendRequest(BaseModel):
    query: str = Field(..., description='Must start with "query:"')
    items: List[ChallengeItemIn]
    top_k: int = 5

    user_gender: Optional[str] = None
    user_age_group: Optional[str] = None
    user_job: Optional[str] = None
    user_available_time: List[str] = []

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