"""
[6단계] Hallucination 방지 검증 로직

생성된 일정에서 장소명을 추출하고,
RAG로 검색된 장소 목록과 대조하여 미검증 장소를 탐지한다.
"""

import re


def extract_place_names(itinerary: str) -> list[str]:
    """
    일정 텍스트에서 '- 장소명: 설명' 패턴으로 장소명을 추출한다.
    Day n: 헤더 행은 건너뜀.
    """
    names = []
    for line in itinerary.splitlines():
        stripped = line.strip()
        # '- 장소명: ...' 형식만 처리
        match = re.match(r"^-\s+(.+?)\s*:", stripped)
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
