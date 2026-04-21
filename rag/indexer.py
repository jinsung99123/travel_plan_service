"""
[신규] 인덱싱 파이프라인 — 저장 / 로드 / 증분 인덱싱

흐름:
  get_or_build_vectorstore(api_key)
    ├─ 디스크 인덱스 없음  → build_and_save()   전체 빌드
    └─ 디스크 인덱스 있음  → load_saved_index()
                              └─ add_new_documents()  증분 업데이트 (새 행만)
"""

import json
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from .loader import load_places, load_persona, build_place_documents

# ── 저장 경로 ─────────────────────────────────────────────────────────────────
_INDEX_DIR = Path(__file__).parent.parent / "index"
_PLACES_INDEX_DIR = _INDEX_DIR / "places"
_IDS_FILE = _INDEX_DIR / "indexed_ids.json"


# ── ID 영속성 ─────────────────────────────────────────────────────────────────
def _get_indexed_ids() -> set[int]:
    """디스크에 저장된 인덱싱 완료 ID 집합 반환. 파일 없으면 빈 집합."""
    if not _IDS_FILE.exists():
        return set()
    with open(_IDS_FILE, encoding="utf-8") as f:
        return set(json.load(f))


def _save_indexed_ids(ids: set[int]) -> None:
    _IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(ids), f)


def _make_embeddings(api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(openai_api_key=api_key)


# ── 전체 재빌드 ───────────────────────────────────────────────────────────────
def build_and_save(api_key: str) -> FAISS:
    """
    places.csv 전체를 임베딩 후 FAISS 인덱스를 디스크에 저장한다.
    loader.build_place_documents()로 구조화 Document를 생성하여
    기존 단순 concat 방식보다 임베딩 품질을 향상시킨다.
    """
    places_df = load_places()
    persona_df = load_persona()
    docs = build_place_documents(places_df, persona_df)

    embeddings = _make_embeddings(api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)

    _PLACES_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(_PLACES_INDEX_DIR))

    ids = {int(row["id"]) for _, row in places_df.iterrows()}
    _save_indexed_ids(ids)

    return vectorstore


# ── 로드 ─────────────────────────────────────────────────────────────────────
def load_saved_index(api_key: str) -> FAISS | None:
    """
    디스크의 FAISS 인덱스를 로드한다.
    index.faiss 파일이 없으면 None 반환 → 호출자가 전체 빌드로 폴백.
    """
    if not (_PLACES_INDEX_DIR / "index.faiss").exists():
        return None
    embeddings = _make_embeddings(api_key)
    return FAISS.load_local(
        str(_PLACES_INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ── 증분 인덱싱 ───────────────────────────────────────────────────────────────
def add_new_documents(vectorstore: FAISS, api_key: str) -> tuple[FAISS, int]:
    """
    places.csv에서 아직 인덱싱되지 않은 행(id 기준)만 추출하여
    기존 벡터스토어에 추가한다. 전체 재빌드 없이 신규 데이터만 처리.

    반환: (updated_vectorstore, 추가된 문서 수)
    """
    indexed_ids = _get_indexed_ids()
    places_df = load_places()
    persona_df = load_persona()

    # id 기반 중복 방지
    new_df = places_df[~places_df["id"].isin(indexed_ids)]
    if new_df.empty:
        return vectorstore, 0

    new_docs = build_place_documents(new_df, persona_df)
    vectorstore.add_documents(new_docs)

    # 인덱스와 ID 목록 동기화
    vectorstore.save_local(str(_PLACES_INDEX_DIR))
    _save_indexed_ids(indexed_ids | {int(r["id"]) for _, r in new_df.iterrows()})

    return vectorstore, len(new_docs)


# ── 메인 진입점 ───────────────────────────────────────────────────────────────
def get_or_build_vectorstore(api_key: str, force_rebuild: bool = False) -> FAISS:
    """
    1. force_rebuild=True  → 전체 재빌드 후 저장
    2. 디스크 인덱스 존재   → 로드 후 신규 데이터 증분 추가
    3. 디스크 인덱스 없음   → 전체 빌드 후 저장
    """
    if force_rebuild:
        return build_and_save(api_key)

    vs = load_saved_index(api_key)
    if vs is not None:
        vs, added = add_new_documents(vs, api_key)
        return vs

    return build_and_save(api_key)
