"""
장소 Document 로더 — places_enriched.json 기반

변경 이력:
  - CSV 기반 build_place_documents() 제거
  - places_enriched.json 직접 로드 방식으로 교체
  - page_content / plain_text: JSON 값 그대로 사용 (가공 금지)
  - metadata: retriever.py 호환 전체 필드 보존
"""

import json
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

DATA_DIR = Path(__file__).parent.parent / "data"
_ENRICHED_JSON = DATA_DIR / "places_enriched.json"


# ── persona 로드 (app.py / test_rag.py에서 직접 사용) ─────────────────────────
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
    sub  = str(row.iloc[0]["보조키워드"]).replace(",", " ")
    return f"{core} {sub}"


# ── 지역 정보 추출 (retriever 등 외부에서 참조 가능) ──────────────────────────
def extract_region(address: str) -> dict[str, str]:
    """주소 → {sido, sigungu, dong, region} 분리 추출."""
    parts = address.split()
    sido    = parts[0] if len(parts) > 0 else ""
    sigungu = parts[1] if len(parts) > 1 else ""
    dong_match = re.search(r"(\S+동)", address)
    dong = dong_match.group(1) if dong_match else ""
    return {
        "sido":    sido,
        "sigungu": sigungu,
        "dong":    dong,
        "region":  f"{sido} {sigungu}".strip(),
    }


# ── 장소 Document 빌드 — JSON 기반 ────────────────────────────────────────────
def build_place_documents(json_path: Path | str | None = None) -> list[Document]:
    """
    places_enriched.json → list[Document]

    page_content: FAISS 임베딩용 자연어 텍스트 (enriched, 가공 없이 그대로 사용)
    plain_text:   BM25 코퍼스용 키워드 열거 텍스트 (metadata에 저장)
    metadata:     retriever.py 호환 전체 필드
    """
    path = Path(json_path) if json_path else _ENRICHED_JSON
    with open(path, encoding="utf-8") as f:
        data: list[dict] = json.load(f)

    docs: list[Document] = []
    for item in data:
        meta = dict(item.get("metadata", {}))
        meta["plain_text"] = item.get("plain_text", "")

        docs.append(Document(
            page_content=item.get("page_content", ""),
            metadata=meta,
        ))
    return docs


# ── 성향 Document 빌드 ────────────────────────────────────────────────────────
def build_persona_documents(persona_df: pd.DataFrame) -> list[Document]:
    """
    성향 인덱싱 전략 (장소와 분리):
    - 성향명을 반복 포함한 자연어 문장 → 성향 기반 검색 정확도 향상
    """
    docs: list[Document] = []
    for _, row in persona_df.iterrows():
        persona  = str(row["성향"])
        core_kws = str(row["핵심키워드"]).replace(",", ", ")
        sub_kws  = str(row["보조키워드"]).replace(",", ", ")

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
