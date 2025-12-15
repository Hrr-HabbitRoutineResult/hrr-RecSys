from fastapi import APIRouter, HTTPException
from app.schemas import RecommendRequest, RecommendResponse, EmbeddingRequest, EmbeddingResponse
from app.services.recommender import run_recommend
from app.services.embedder import embed_text
from app import config
from sentence_transformers import SentenceTransformer

router = APIRouter()

@router.get("/")
def health():
    # 모델 로드 상태 확인
    ok = True
    loaded = True
    try:
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

@router.post("/embedding", response_model=EmbeddingResponse)
def embedding(req: EmbeddingRequest):
    if not req.text or req.text.strip() == "":
        raise HTTPException(status_code=400, detail="text is empty")

    emb = embed_text(req.text)
    return EmbeddingResponse(embedding=emb)