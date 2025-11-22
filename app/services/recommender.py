import time
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

from app import config
from app.schemas import ChallengeItemIn, RecommendItemOut, RecommendResponse
from app.utils.text import extract_from_query, has_time_overlap

_model: SentenceTransformer | None = None

def load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.MODEL_PATH, device=config.DEVICE)
    return _model

def embed_text(text: str) -> List[float]:
    """ /embedding 에서 쓰는 함수: 항상 모델 로드 후 사용 """
    model = load_model()
    emb = model.encode(text, normalize_embeddings=True)
    return emb.tolist()

def _base_similarity(q_vec: np.ndarray, emb_matrix: np.ndarray) -> np.ndarray:
    return emb_matrix @ q_vec  # cosine similarity

def _apply_boosts(scores: np.ndarray, items: List[ChallengeItemIn], qtext: str) -> np.ndarray:
    """카테고리/시간/목표 보너스 적용"""
    w_cat, w_time, w_goal = config.W_CAT, config.W_TIME, config.W_GOAL
    cat_pref, time_pref, goal_pref = extract_from_query(qtext)

    boosted = scores.copy()

    if cat_pref:
        cat_mask = np.array([(it.category == cat_pref) for it in items], dtype=np.float32)
        boosted += w_cat * cat_mask

    if time_pref:
        time_mask = np.array([
            has_time_overlap(time_pref, it.cert_time_slots)
            for it in items
        ], dtype=np.float32)
        boosted += w_time * time_mask

    if goal_pref:
        goal_mask = np.array([(it.goal_text == goal_pref) for it in items], dtype=np.float32)
        boosted += w_goal * goal_mask

    return boosted

def recommend(query: str, items: List[ChallengeItemIn], top_k: int) -> RecommendResponse:
    model = load_model()
    t0 = time.time()

    # 1) 쿼리 임베딩
    qtext = query if query.lower().startswith("query:") else f"query: {query}"
    q_vec = model.encode([qtext], normalize_embeddings=True)[0].astype(np.float32)   # (D,)

    # 2) 아이템 임베딩 행렬
    embs = np.vstack([
        np.asarray(it.embedding, dtype=np.float32)
        for it in items
    ])   # (N, D)

    # 3) 기본 점수 + 보너스
    base_scores = _base_similarity(q_vec, embs)
    boosted = _apply_boosts(base_scores, items, qtext)

    # 4) Top-K
    k = min(top_k, len(items))
    top_idx = np.argsort(boosted)[::-1][:k]

    recs = [
        RecommendItemOut(
            challengeId=items[i].challengeId,
            title=items[i].title,
            description=items[i].description,
            matchScore=float(boosted[i]),
        )
        for i in top_idx
    ]

    latency_ms = int((time.time() - t0) * 1000)
    return RecommendResponse(
        recommendations=recs,
        modelVersion=config.MODEL_VERSION,
        latencyMs=latency_ms,
    )
