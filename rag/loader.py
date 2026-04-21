"""
[1단계] CSV 데이터 로드
[2단계] 성향 키워드 매핑
[Contextual Retrieval] 문서 빌드 시 성향/지역/카테고리 컨텍스트를 page_content에 포함
"""

import re
import pandas as pd
from pathlib import Path
from functools import lru_cache
from langchain_core.documents import Document

DATA_DIR = Path(__file__).parent.parent / "data"


@lru_cache(maxsize=1)
def load_places() -> pd.DataFrame:
    """places_extended.csv 우선 로드, 없으면 places.csv fallback."""
    path = DATA_DIR / "places_extended.csv"
    if path.exists():
        return pd.read_csv(path, encoding="utf-8")
    return pd.read_csv(DATA_DIR / "places.csv", encoding="utf-8")


@lru_cache(maxsize=1)
def load_persona() -> pd.DataFrame:
    """persona.csv 로드. 반복 I/O 방지를 위해 캐시 적용."""
    return pd.read_csv(DATA_DIR / "persona.csv", encoding="utf-8")


def get_personality_keywords(personality: str) -> str:
    """persona.csv에서 성향에 해당하는 핵심+보조 키워드를 반환한다."""
    df = load_persona()
    row = df[df["성향"] == personality]
    if row.empty:
        return personality
    core = str(row.iloc[0]["핵심키워드"]).replace(",", " ")
    sub = str(row.iloc[0]["보조키워드"]).replace(",", " ")
    return f"{core} {sub}"


# ── 지역 정보 구조화 추출 ────────────────────────────────────────────────────
def extract_region(address: str) -> dict[str, str]:
    """
    주소 → {sido, sigungu, dong, region} 분리 추출.
    "서울 노원구 동일로 1015" → {"sido": "서울", "sigungu": "노원구", ...}
    """
    parts = address.split()
    sido = parts[0] if len(parts) > 0 else ""
    sigungu = parts[1] if len(parts) > 1 else ""
    dong_match = re.search(r"(\S+동)", address)
    dong = dong_match.group(1) if dong_match else ""
    return {
        "sido": sido,
        "sigungu": sigungu,
        "dong": dong,
        "region": f"{sido} {sigungu}".strip(),
    }


# ── 성향 태그 매핑 ──────────────────────────────────────────────────────────
def map_personality_tags(keywords_str: str, persona_df: pd.DataFrame) -> list[str]:
    """
    장소 키워드 ∩ 성향 키워드 교집합으로 매칭 성향 태그를 반환한다.
    "카페,감성,여유" → ["힐링형"]
    """
    place_kws = {k.strip() for k in keywords_str.split(",")}
    tags: list[str] = []
    for _, row in persona_df.iterrows():
        persona_kws = {
            k.strip()
            for k in f"{row['핵심키워드']},{row['보조키워드']}".split(",")
        }
        if place_kws & persona_kws:
            tags.append(str(row["성향"]))
    return tags


# ── 장소 Document 빌드 — Contextual Embedding 포맷 ─────────────────────────
def build_place_documents(
    places_df: pd.DataFrame,
    persona_df: pd.DataFrame,
) -> list[Document]:
    """
    [Contextual Embedding 개선]

    기존 page_content: "장소명 카테고리 키워드 설명 지역" 단순 구조
    변경 page_content: 성향·지역·카테고리를 앞에 배치한 자연어 컨텍스트
      → 임베딩 모델이 "이 장소가 누구를 위해, 어디에, 어떤 목적으로 있는가"를 벡터에 반영

    예시:
      [힐링형, 미식형] 성향 여행자가 서울 노원구에서 방문할 수 있는 카페.
      장소명: 노원두물마루 커피&스낵
      키워드: 카페 감성 여유
      설명: 간단한 커피와 간식을 즐길 수 있는 소규모 카페

    plain_text 메타데이터: BM25·디버깅·디스플레이용으로 원본 구조화 텍스트 보존
    """
    docs: list[Document] = []
    for _, row in places_df.iterrows():
        kw_str = str(row["키워드"])
        address = str(row["주소"])
        region_info = extract_region(address)
        p_tags = map_personality_tags(kw_str, persona_df)

        # ── Contextual page_content (임베딩 대상) ─────────────────────────
        # 성향 컨텍스트: 매칭 성향이 있으면 앞에 명시, 없으면 일반 여행자
        if p_tags:
            persona_ctx = f"[{', '.join(p_tags)}] 성향 여행자가"
        else:
            persona_ctx = "여행자가"

        page_content = (
            f"{persona_ctx} {region_info['region']}에서 방문할 수 있는 {row['카테고리']}.\n"
            f"장소명: {row['장소명']}\n"
            f"키워드: {kw_str.replace(',', ' ')}\n"
            f"설명: {row['설명']}"
        )

        # ── 원본 구조화 텍스트 (BM25·디버깅용) ─────────────────────────────
        plain_text = (
            f"장소명: {row['장소명']}\n"
            f"카테고리: {row['카테고리']}\n"
            f"키워드: {kw_str.replace(',', ' ')}\n"
            f"설명: {row['설명']}\n"
            f"지역: {region_info['region']}"
        )

        metadata: dict = {
            # ── 기존 필드 (하위 호환 유지) ──────────────────────────────────
            "id":      int(row["id"]),
            "장소명":   str(row["장소명"]),
            "카테고리":  str(row["카테고리"]),
            "키워드":    kw_str,
            "설명":     str(row["설명"]),
            "주소":     address,
            # ── 확장 메타데이터 ──────────────────────────────────────────────
            "category":         str(row["카테고리"]),
            "keywords":         [k.strip() for k in kw_str.split(",")],
            "region":           region_info["region"],
            "sido":             region_info["sido"],
            "sigungu":          region_info["sigungu"],
            "dong":             region_info["dong"],
            "personality_tags": p_tags,
            # ── 확장 메타데이터 (places_extended.csv) ───────────────────────
            "stay_time":        str(row.get("stay_time", "")),
            "crowd_level":      str(row.get("crowd_level", "")),
            "best_time":        str(row.get("best_time", "")),
            "price_level":      str(row.get("price_level", "")),
            "indoor_outdoor":   str(row.get("indoor_outdoor", "")),
            "weather_fit":      str(row.get("weather_fit", "")),
            # ── Contextual Retrieval 전용 ────────────────────────────────────
            "plain_text":       plain_text,  # BM25·디스플레이용 원본 텍스트
        }
        docs.append(Document(page_content=page_content, metadata=metadata))
    return docs


# ── 성향 Document 빌드 — 장소와 다른 전략 ────────────────────────────────────
def build_persona_documents(persona_df: pd.DataFrame) -> list[Document]:
    """
    성향 인덱싱 전략 (장소와 분리):
    - 성향명을 반복 포함한 자연어 문장 → 성향 기반 검색 정확도 향상
    - 향후 성향 벡터스토어로 확장 가능
    """
    docs: list[Document] = []
    for _, row in persona_df.iterrows():
        persona = str(row["성향"])
        core_kws = str(row["핵심키워드"]).replace(",", ", ")
        sub_kws = str(row["보조키워드"]).replace(",", ", ")

        page_content = (
            f"{persona} 여행자를 위한 키워드.\n"
            f"핵심 관심사: {core_kws}.\n"
            f"선호 활동: {sub_kws}.\n"
            f"유형 설명: {row['설명']}"
        )

        metadata: dict = {
            "id":           f"persona_{persona}",
            "성향":          persona,
            "핵심키워드":      core_kws,
            "보조키워드":      sub_kws,
            "설명":          str(row["설명"]),
            "all_keywords": [
                k.strip()
                for k in f"{row['핵심키워드']},{row['보조키워드']}".split(",")
            ],
        }
        docs.append(Document(page_content=page_content, metadata=metadata))
    return docs
