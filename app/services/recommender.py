import time
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

from app import config
from app.schemas import ChallengeItemIn, RecommendItemOut, RecommendResponse
from app.utils.text import extract_from_query, has_time_overlap, normalize_goal_label

_model: SentenceTransformer | None = None

def load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(
                config.MODEL_PATH,
                device=config.DEVICE,
                tokenizer_kwargs={"fix_mistral_regex": True},
            )
        except TypeError:
            _model = SentenceTransformer(config.MODEL_PATH, device=config.DEVICE)
    return _model


def _ensure_query_prefix(text: str) -> str:
    t = text.strip()
    return t if t.lower().startswith("query:") else f"query: {t}"


def _safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default


def _dist_fit_score(qtext: str, it: ChallengeItemIn) -> float:
    """
    user demographic을 query 템플릿에서 얕게 추출해서
    챌린지 참여자 분포와의 fit 점수를 계산 (성별, 나이, 직업)
    """
    # gender
    u_gender = None
    if "gender=FEMALE" in qtext or "여성" in qtext:
        u_gender = "FEMALE"
    elif "gender=MALE" in qtext or "남성" in qtext:
        u_gender = "MALE"

    # age
    u_age = None
    if "10대" in qtext:
        u_age = "10s"
    elif "20대" in qtext:
        u_age = "20s"
    elif "30대" in qtext:
        u_age = "30s"
    elif "40대" in qtext:
        u_age = "40s"
    elif "50대" in qtext:
        u_age = "50p"

    # job
    u_job = None
    if "중고등학생" in qtext:
        u_job = "student_middle_high"
    elif "대학생" in qtext:
        u_job = "student_university"
    elif "취준생" in qtext:
        u_job = "job_job_seeker"
    elif "직장인" in qtext:
        u_job = "job_employee"
    elif "주부" in qtext:
        u_job = "job_homemaker"
    elif "기타" in qtext:
        u_job = "job_etc"

    s = 0.0

    # gender pct
    if u_gender == "FEMALE":
        s += _safe_float(getattr(it, "participants_female_pct", 0.0))
    elif u_gender == "MALE":
        s += _safe_float(getattr(it, "participants_male_pct", 0.0))

    # age pct
    if u_age == "10s":
        s += _safe_float(getattr(it, "age_10s_pct", 0.0))
    elif u_age == "20s":
        s += _safe_float(getattr(it, "age_20s_pct", 0.0))
    elif u_age == "30s":
        s += _safe_float(getattr(it, "age_30s_pct", 0.0))
    elif u_age == "40s":
        s += _safe_float(getattr(it, "age_40s_pct", 0.0))
    elif u_age == "50p":
        s += _safe_float(getattr(it, "age_50p_pct", 0.0))

    # job pct
    if u_job == "student_middle_high":
        s += _safe_float(getattr(it, "job_student_middle_high_pct", 0.0))
    elif u_job == "student_university":
        s += _safe_float(getattr(it, "job_student_university_pct", 0.0))
    elif u_job == "job_job_seeker":
        s += _safe_float(getattr(it, "job_job_seeker_pct", 0.0))
    elif u_job == "job_employee":
        s += _safe_float(getattr(it, "job_employee_pct", 0.0))
    elif u_job == "job_homemaker":
        s += _safe_float(getattr(it, "job_homemaker_pct", 0.0))
    elif u_job == "job_etc":
        s += _safe_float(getattr(it, "job_etc_pct", 0.0))

    return s


def _apply_boosts(scores: np.ndarray, items: List[ChallengeItemIn], qtext: str) -> np.ndarray:
    """
    boosted = sim + cat/time/goal + dist-fit
    """
    cat_pref, time_pref, goal_pref = extract_from_query(qtext)
    boosted = scores.astype(np.float32).copy()

    # category boost
    if cat_pref:
        cat_mask = np.array([(it.category == cat_pref) for it in items], dtype=np.float32)
        boosted += config.W_CAT * cat_mask

    # time boost
    if time_pref:
        time_mask = np.array([has_time_overlap(time_pref, it.cert_time_slots) for it in items], dtype=np.float32)
        boosted += config.W_TIME * time_mask

    # goal boost (label normalize)
    if goal_pref:
        goal_mask = np.array([(normalize_goal_label(it.goal_text) == goal_pref) for it in items], dtype=np.float32)
        boosted += config.W_GOAL * goal_mask

    # dist boost
    w_dist = float(getattr(config, "W_DIST", 0.0))
    if w_dist != 0.0:
        dist_scores = np.array([_dist_fit_score(qtext, it) for it in items], dtype=np.float32)
        boosted += w_dist * dist_scores

    return boosted


def _filter_candidates(items: List[ChallengeItemIn], qtext: str) -> List[ChallengeItemIn]:
    """
    2-stage 후보 필터:
    - category 우선
    - time overlap
    너무 줄어들면 자동 완화
    """
    cat_pref, time_pref, _ = extract_from_query(qtext)

    cand = items

    if cat_pref:
        c1 = [it for it in cand if it.category == cat_pref]
        if len(c1) >= max(10, int(0.05 * len(items))):
            cand = c1

    if time_pref:
        c2 = [it for it in cand if has_time_overlap(time_pref, it.cert_time_slots)]
        if len(c2) >= max(10, int(0.05 * len(items))):
            cand = c2

    return cand


def run_recommend(query: str, items: List[ChallengeItemIn], top_k: int) -> RecommendResponse:
    model = load_model()
    t0 = time.time()

    qtext = _ensure_query_prefix(query)

    # 1) 후보군 필터
    cand = _filter_candidates(items, qtext)

    # 2) query(사용자) embed
    q_vec = model.encode([qtext], normalize_embeddings=True)[0].astype(np.float32)

    # 3) candidate embeddings
    embs = np.vstack([np.asarray(it.embedding, dtype=np.float32) for it in cand])

    # 4) base sim
    base_scores = embs @ q_vec

    # 5) boosts
    boosted = _apply_boosts(base_scores, cand, qtext)

    # 6) top-k
    k = min(top_k, len(cand))
    if k <= 0:
        return RecommendResponse(
            recommendations=[],
            modelVersion=config.MODEL_VERSION,
            latencyMs=int((time.time() - t0) * 1000),
        )

    top_idx = np.argpartition(-boosted, k - 1)[:k]
    top_idx = top_idx[np.argsort(-boosted[top_idx])]

    recs = [
        RecommendItemOut(
            challengeId=cand[i].challengeId,
            title=cand[i].title,
            description=cand[i].description,
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
