import re
from typing import Optional, Tuple, Set

def passage(r) -> str:
    """E5 포맷 passage 텍스트 생성"""
    return f"passage: {r.title} — {r.description} (카테고리={r.category} / 인증시간={r.cert_time_slots} / 목표={r.goal_text})"

def extract_from_query(qtext: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """쿼리에서 관심 카테고리/시간/목표를 파싱"""
    cat = None
    m_cat = re.search(r"(관심|카테고리)\s*=\s*([^\s,;]+)", qtext)
    if m_cat:
        cat = m_cat.group(2).strip()

    time = None
    m_time = re.search(r"(\d{1,2}-\d{1,2}(?:;\d{1,2}-\d{1,2})*)", qtext)
    if m_time:
        time = m_time.group(1).strip()

    goal = None
    m_goal = re.search(r"목표\s*=\s*([^,]+)", qtext)
    if m_goal:
        goal = m_goal.group(1).strip()

    return cat, time, goal

def slot_set(s: Optional[str]) -> Set[str]:
    if not s:
        return set()
    return set(str(s).split(";"))

def has_time_overlap(pref: Optional[str], item: Optional[str]) -> bool:
    return len(slot_set(pref) & slot_set(item)) > 0
