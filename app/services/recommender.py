import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from app import config
from app.schemas import ChallengeItemIn, RecommendItemOut, RecommendResponse
from app.utils.text import extract_from_query, has_time_overlap, normalize_goal_label


# =========================================================
# Model loader
# =========================================================

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


# =========================================================
# Utils
# =========================================================

def _ensure_query_prefix(text: str) -> str:
    t = text.strip()
    return t if t.lower().startswith("query:") else f"query: {t}"


def _safe_float(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def _pct01(pct_value) -> float:
    """
    0~100(%) 또는 0~1 값을 0~1로 정규화
    """
    v = _safe_float(pct_value, 0.0)
    if v <= 1.0:
        return _clamp(v, 0.0, 1.0)
    return _clamp(v / 100.0, 0.0, 1.0)


# =========================================================
# Demographic preference
# =========================================================

@dataclass(frozen=True)
class DistPref:
    gender: Optional[str] = None       # MALE / FEMALE
    age_group: Optional[str] = None    # TEENS / TWENTIES / ...
    job: Optional[str] = None          # EMPLOYEE / STUDENT_...


def _dist_fit_score(pref: DistPref, it: ChallengeItemIn) -> float:
    """
    user demographic과 challenge 참여자 분포의 fit score
    - 반환값: 0 ~ 1
    """
    if pref.gender is None and pref.age_group is None and pref.job is None:
        return 0.0

    parts: List[Tuple[float, float]] = []  # (value01, weight)

    # gender
    if pref.gender == "FEMALE":
        parts.append((_pct01(it.participants_female_pct), 1.0))
    elif pref.gender == "MALE":
        parts.append((_pct01(it.participants_male_pct), 1.0))

    # age
    if pref.age_group == "TEENS":
        parts.append((_pct01(it.age_10s_pct), 1.0))
    elif pref.age_group == "TWENTIES":
        parts.append((_pct01(it.age_20s_pct), 1.0))
    elif pref.age_group == "THIRTIES":
        parts.append((_pct01(it.age_30s_pct), 1.0))
    elif pref.age_group == "FORTIES":
        parts.append((_pct01(it.age_40s_pct), 1.0))
    elif pref.age_group == "FIFTIES_PLUS":
        parts.append((_pct01(it.age_50p_pct), 1.0))

    # job
    if pref.job == "STUDENT_MIDDLE_HIGH":
        parts.append((_pct01(it.job_student_middle_high_pct), 1.0))
    elif pref.job == "STUDENT_UNIVERSITY":
        parts.append((_pct01(it.job_student_university_pct), 1.0))
    elif pref.job == "JOB_SEEKER":
        parts.append((_pct01(it.job_job_seeker_pct), 1.0))
    elif pref.job == "EMPLOYEE":
        parts.append((_pct01(it.job_employee_pct), 1.0))
    elif pref.job == "HOMEMAKER":
        parts.append((_pct01(it.job_homemaker_pct), 1.0))
    elif pref.job == "ETC":
        parts.append((_pct01(it.job_etc_pct), 1.0))

    if not parts:
        return 0.0

    return float(sum(v * w for v, w in parts) / sum(w for _, w in parts))


# =========================================================
# Boosting logic
# =========================================================

def _apply_boosts(
    scores: np.ndarray,
    items: List[ChallengeItemIn],
    qtext: str,
    pref: DistPref,
    user_available_time: List[str],
) -> np.ndarray:
    """
    boosted = embedding_sim
            + category boost
            + time boost
            + goal boost
            + demographic distribution boost
    """
    boosted = scores.astype(np.float32).copy()
    cat_pref, _, goal_pref = extract_from_query(qtext)

    # category boost
    if cat_pref:
        cat_mask = np.array([(it.category == cat_pref) for it in items], dtype=np.float32)
        boosted += float(getattr(config, "W_CAT", 0.0)) * cat_mask

    # time boost
    if user_available_time:
        time_mask = np.array(
            [has_time_overlap(user_available_time, it.cert_time_slots) for it in items],
            dtype=np.float32,
        )
        boosted += float(getattr(config, "W_TIME", 0.0)) * time_mask

    # goal boost
    if goal_pref:
        goal_mask = np.array(
            [(normalize_goal_label(it.goal_text) == goal_pref) for it in items],
            dtype=np.float32,
        )
        boosted += float(getattr(config, "W_GOAL", 0.0)) * goal_mask

    # demographic distribution boost
    w_dist = float(getattr(config, "W_DIST", 0.0))
    if w_dist > 0.0:
        dist_scores = np.array(
            [_dist_fit_score(pref, it) for it in items],
            dtype=np.float32,
        )
        boosted += w_dist * dist_scores

    return boosted


# =========================================================
# Candidate filter
# =========================================================

def _filter_candidates(items: List[ChallengeItemIn], qtext: str) -> List[ChallengeItemIn]:
    cat_pref, _, _ = extract_from_query(qtext)
    cand = items

    if cat_pref:
        c1 = [it for it in cand if it.category == cat_pref]
        if len(c1) >= max(10, int(0.05 * len(items))):
            cand = c1

    return cand


# =========================================================
# Main entry
# =========================================================

def run_recommend(
    query: str,
    items: List[ChallengeItemIn],
    top_k: int,
    user_gender: Optional[str] = None,
    user_age_group: Optional[str] = None,
    user_job: Optional[str] = None,
    user_available_time: List[str] = [],
) -> RecommendResponse:

    model = load_model()
    t0 = time.time()

    qtext = _ensure_query_prefix(query)

    # user demographic preference
    pref = DistPref(
        gender=user_gender,
        age_group=user_age_group,
        job=user_job,
    )

    # 1) candidate filter
    cand = _filter_candidates(items, qtext)
    if not cand:
        return RecommendResponse(
            recommendations=[],
            modelVersion=config.MODEL_VERSION,
            latencyMs=int((time.time() - t0) * 1000),
        )

    # 2) embedding similarity
    q_vec = model.encode([qtext], normalize_embeddings=True)[0].astype(np.float32)
    embs = np.vstack([np.asarray(it.embedding, dtype=np.float32) for it in cand])
    base_scores = embs @ q_vec

    # 3) boosts
    boosted = _apply_boosts(
        base_scores,
        cand,
        qtext,
        pref,
        user_available_time,
    )

    # 4) top-k
    k = min(top_k, len(cand))
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

    return RecommendResponse(
        recommendations=recs,
        modelVersion=config.MODEL_VERSION,
        latencyMs=int((time.time() - t0) * 1000),
    )
