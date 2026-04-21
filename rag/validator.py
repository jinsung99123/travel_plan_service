"""
[6단계] Hallucination 방지 검증 로직

생성된 일정에서 장소명을 추출하고,
RAG로 검색된 장소 목록과 대조하여 미검증 장소를 탐지한다.
"""

import re
from collections import defaultdict


def extract_place_names(itinerary: str) -> list[str]:
    """
    일정 텍스트에서 'HH:MM - HH:MM 장소명' 패턴으로 장소명을 추출한다.
    Day n 헤더, 설명/추천이유/대안 행은 건너뜀.
    """
    names = []
    for line in itinerary.splitlines():
        stripped = line.strip()
        # 'HH:MM - HH:MM 장소명' 형식
        match = re.match(r"^\d{2}:\d{2}\s*-\s*\d{2}:\d{2}\s+(.+)$", stripped)
        if match:
            names.append(match.group(1).strip())
    return names


def _is_matched(name: str, retrieved_names: set[str]) -> bool:
    """
    정확 일치 또는 부분 문자열 일치로 장소명 검증.
    LLM이 장소명을 약간 다르게 출력할 수 있으므로 양방향 포함 검사 사용.
    """
    if name in retrieved_names:
        return True
    for r in retrieved_names:
        if name in r or r in name:
            return True
    return False


_STAY_MINUTES: dict[str, int] = {
    "30~60":   45,
    "60~120":  90,
    "120~180": 150,
}


def check_density(
    itinerary: str,
    retrieved_places: list[dict],
    max_daily_minutes: int = 480,
) -> dict:
    """
    일정의 하루별 예상 체류 시간 합계를 계산하여 과밀 여부를 반환한다.

    반환값:
      day_reports    : {day_num: {places, estimated_minutes, is_overcrowded}}
      any_overcrowded: 하나라도 초과한 날이 있으면 True
    """
    place_stay: dict[str, int] = {}
    for p in retrieved_places:
        name = p.get("장소명", "")
        stay_key = p.get("stay_time", "")
        place_stay[name] = _STAY_MINUTES.get(stay_key, 60)

    day_buckets: dict[int, list[str]] = defaultdict(list)
    current_day = 0
    for line in itinerary.splitlines():
        stripped = line.strip()
        day_match = re.match(r"^Day\s+(\d+)", stripped, re.IGNORECASE)
        if day_match:
            current_day = int(day_match.group(1))
            continue
        place_match = re.match(r"^\d{2}:\d{2}\s*-\s*\d{2}:\d{2}\s+(.+)$", stripped)
        if place_match and current_day > 0:
            day_buckets[current_day].append(place_match.group(1).strip())

    day_reports: dict[int, dict] = {}
    for day_num, places in day_buckets.items():
        total = sum(place_stay.get(name, 60) for name in places)
        day_reports[day_num] = {
            "places":            places,
            "estimated_minutes": total,
            "is_overcrowded":    total > max_daily_minutes,
        }

    return {
        "day_reports":     day_reports,
        "any_overcrowded": any(r["is_overcrowded"] for r in day_reports.values()),
    }


def check_diversity(itinerary: str, retrieved_places: list[dict]) -> dict:
    """
    일정의 카테고리 다양성을 검증한다.

    경고 조건:
      - 동일 카테고리가 3개 이상 등장
      - 카테고리 종류가 2개 미만

    반환값:
      category_counts : {카테고리: 등장 횟수}
      overused        : 3개 이상 등장한 카테고리 목록
      is_diverse      : 경고 없으면 True
      issues          : 경고 메시지 목록
    """
    place_category: dict[str, str] = {
        p["장소명"]: p.get("카테고리", "") for p in retrieved_places
    }

    extracted = extract_place_names(itinerary)

    category_counts: dict[str, int] = {}
    for name in extracted:
        cat = place_category.get(name, "")
        if cat:
            category_counts[cat] = category_counts.get(cat, 0) + 1

    issues: list[str] = []
    overused = [cat for cat, cnt in category_counts.items() if cnt >= 3]
    if overused:
        issues.append(f"동일 카테고리 과다: {', '.join(overused)}")
    if len(category_counts) < 2:
        issues.append(f"카테고리 다양성 부족: {len(category_counts)}개 카테고리만 포함")

    return {
        "category_counts": category_counts,
        "overused":        overused,
        "is_diverse":      len(issues) == 0,
        "issues":          issues,
    }


def check_quality(
    itinerary: str,
    retrieved_places: list[dict],
    max_daily_minutes: int = 480,
) -> dict:
    """
    밀도(density)와 다양성(diversity)을 통합 검증한다.

    반환값:
      density_ok   : 과밀 없으면 True
      diversity_ok : 카테고리 다양성 경고 없으면 True
      issues       : 모든 경고 메시지 목록
    """
    density   = check_density(itinerary, retrieved_places, max_daily_minutes)
    diversity = check_diversity(itinerary, retrieved_places)

    issues: list[str] = []
    if density["any_overcrowded"]:
        overcrowded_days = [
            f"Day {d}"
            for d, r in density["day_reports"].items()
            if r["is_overcrowded"]
        ]
        issues.append(f"일정 과밀: {', '.join(overcrowded_days)}")
    issues.extend(diversity["issues"])

    return {
        "density_ok":   not density["any_overcrowded"],
        "diversity_ok": diversity["is_diverse"],
        "issues":       issues,
    }


def validate_itinerary(itinerary: str, retrieved_places: list[dict]) -> dict:
    """
    일정의 각 장소가 RAG 검색 결과에 포함되어 있는지 검증한다.

    반환값:
      total      : 일정에서 추출된 장소 수
      verified   : RAG 결과에서 확인된 장소 목록
      unverified : 확인되지 않은 장소 목록 (hallucination 의심)
      score      : 검증 비율 (0.0 ~ 1.0)
      is_clean   : 미검증 장소가 없으면 True
    """
    retrieved_names = {p["장소명"] for p in retrieved_places}
    extracted = extract_place_names(itinerary)

    verified = [n for n in extracted if _is_matched(n, retrieved_names)]
    unverified = [n for n in extracted if not _is_matched(n, retrieved_names)]

    total = len(extracted)
    return {
        "total":      total,
        "verified":   verified,
        "unverified": unverified,
        "score":      len(verified) / total if total else 0.0,
        "is_clean":   len(unverified) == 0,
    }
