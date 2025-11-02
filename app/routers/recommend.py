from fastapi import APIRouter, HTTPException
from app.schemas import RecommendRequest, RecommendResponse
from app.services.recommender import recommend as run_recommend
from app import config
from sentence_transformers import SentenceTransformer

router = APIRouter()

@router.get("/health")
def health():
    # 모델 로드 상태 확인
    ok = True
    loaded = True
    try:
        # 간단히 인스턴스 생성 확인만
        SentenceTransformer
    except Exception:
        loaded = False
        ok = False

    return {
        "ok": ok,
        "modelLoaded": loaded,
        "modelVersion": config.MODEL_VERSION
    }

@router.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if not req.items:
        raise HTTPException(status_code=400, detail="items list is empty")

    return run_recommend(query=req.query, items=req.items, top_k=req.top_k)
