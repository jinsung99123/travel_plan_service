"""
agents/multi_agent/course_agent.py

Course Agent — 코스 조합 담당

역할:
  Retrieval Agent가 반환한 장소 목록을 카테고리 기반으로 단계별 코스로 조합한다.
  LLM 미사용 — 순수 Python 로직만 사용.
"""

import itertools

_FOOD_CATEGORIES = {
    "냉면", "한식", "육류", "갈비", "돈까스", "초밥", "칼국수", "보쌈",
    "해산물", "해물", "두부", "닭요리", "일식", "양식", "베트남", "태국",
    "중식", "치킨", "삼계탕", "참치", "조개", "장어", "떡볶이", "이탈리안",
}


def build_course(places: list[dict]) -> list[list[dict]]:
    cafe  = [p for p in places if "카페" in p.get("카테고리", "")]
    food  = [p for p in places if p.get("카테고리", "") in _FOOD_CATEGORIES]
    other = [p for p in places if p not in cafe and p not in food]

    steps = [s for s in [cafe, food, other] if s]
    if len(steps) < 2:
        return []

    return [list(combo) for combo in list(itertools.product(*steps))[:3]]


def validate_course(course: list[list[dict]]) -> dict:
    if not course or len(course[0]) < 2:
        return {"passed": False, "reason": "장소 수 부족 (최소 2개)"}
    cats = [p.get("카테고리", "") for p in course[0]]
    if len(set(cats)) < 2:
        return {"passed": False, "reason": "카테고리 다양성 부족"}
    names = [p.get("장소명", "") for p in course[0]]
    if len(names) != len(set(names)):
        return {"passed": False, "reason": "중복 장소 존재"}
    return {"passed": True, "reason": "검증 통과"}


# ── Agent 인터페이스 ───────────────────────────────────────────────────────────

def run_course(places: list[dict]) -> list[list[dict]]:
    return build_course(places)
