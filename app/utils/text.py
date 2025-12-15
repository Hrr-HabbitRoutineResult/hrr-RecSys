import re
from typing import Optional, Tuple, Set

GOAL_LABELS = {"운동습관", "건강", "준비", "취미발견", "취미공유", "몰입", "꾸준함"}

GOAL_TEXT_TO_LABEL_RULES = [
    (re.compile(r"(운동\s*습관|습관을\s*만들|루틴|꾸준히\s*운동)", re.I), "운동습관"),
    (re.compile(r"(건강한\s*하루|컨디션|수면|수분|스트레칭|회복|웰빙)", re.I), "건강"),
    (re.compile(r"(시험|취업|자소서|자기소개서|면접|포트폴리오|준비)", re.I), "준비"),
    (re.compile(r"(새로운\s*취미|취미\s*발견|입문|처음\s*시작)", re.I), "취미발견"),
    (re.compile(r"(함께\s*취미|취미\s*공유|커뮤니티|후기\s*공유|피드백|응원)", re.I), "취미공유"),
    (re.compile(r"(몰입|딥워크|집중|방해\s*차단)", re.I), "몰입"),
    (re.compile(r"(꾸준|연속|이어가|계속\s*하기|루틴\s*유지)", re.I), "꾸준함"),
]

TIME_ENUMS = [
    "EARLY_MORNING", "MORNING", "LUNCH", "AFTERNOON", "EVENING", "NIGHT", "LATE_NIGHT"
]
TIME_KO_HINTS_TO_ENUM = [
    (re.compile(r"(이른\s*아침|05-09|5-9)"), "EARLY_MORNING"),
    (re.compile(r"(아침|09-12|9-12)"), "MORNING"),
    (re.compile(r"(점심|12-14)"), "LUNCH"),
    (re.compile(r"(오후|14-18)"), "AFTERNOON"),
    (re.compile(r"(저녁|18-21)"), "EVENING"),
    (re.compile(r"(밤|21-24)"), "NIGHT"),
    (re.compile(r"(심야|00-05|0-5)"), "LATE_NIGHT"),
]

CATEGORY_ENUMS = {"HEALTH", "STUDY", "HOBBY", "CAREER", "HABIT"}
CATEGORY_KO_TO_ENUM = {"운동": "HEALTH", "학업": "STUDY", "취미": "HOBBY", "취업준비": "CAREER", "생활습관": "HABIT"}

_ENUM_KV_RE = re.compile(r"(category|available_time|goal)\s*=\s*([A-Z_가-힣0-9\-]+)")


def split_set(s: Optional[str]) -> Set[str]:
    if not s:
        return set()
    return set(x.strip() for x in str(s).split(";") if x.strip())


def normalize_goal_label(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s in GOAL_LABELS:
        return s
    for rgx, lab in GOAL_TEXT_TO_LABEL_RULES:
        if rgx.search(s):
            return lab
    return None


def normalize_time(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s in TIME_ENUMS:
        return s
    for rgx, enum in TIME_KO_HINTS_TO_ENUM:
        if rgx.search(s):
            return enum
    return None


def normalize_category(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if s in CATEGORY_ENUMS:
        return s
    if s in CATEGORY_KO_TO_ENUM:
        return CATEGORY_KO_TO_ENUM[s]
    for ko, enum in CATEGORY_KO_TO_ENUM.items():
        if ko in s:
            return enum
    return None


def extract_from_query(qtext: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns (cat_enum, time_enum, goal_label)
    Supports:
      - enum-style: category=HEALTH; available_time=EVENING; goal=운동습관
      - natural template
    """
    t = qtext.strip()
    cat = tm = goal = None

    for k, v in _ENUM_KV_RE.findall(t):
        if k == "category":
            cat = normalize_category(v)
        elif k == "available_time":
            tm = normalize_time(v)
        elif k == "goal":
            goal = normalize_goal_label(v)

    if cat is None:
        cat = normalize_category(t)
    if tm is None:
        tm = normalize_time(t)
    if goal is None:
        goal = normalize_goal_label(t)

    return cat, tm, goal


def has_time_overlap(time_pref: str, item_time_slots: Optional[str]) -> bool:
    if not time_pref:
        return True
    item_set = split_set(item_time_slots)
    if not item_set:
        return True
    return time_pref in item_set
