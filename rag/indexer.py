"""
인덱싱 파이프라인 — 저장 / 로드 / 증분 인덱싱

흐름:
  get_or_build_vectorstore(api_key)
    ├─ 디스크 인덱스 없음  → build_and_save()   전체 빌드
    └─ 디스크 인덱스 있음  → load_saved_index()
                              └─ add_new_documents()  증분 업데이트 (새 행만)

데이터 소스: places_enriched.json (CSV 기반 빌드 제거)
"""

import json
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from .loader import build_place_documents

_INDEX_DIR        = Path(__file__).parent.parent / "index"
_PLACES_INDEX_DIR = _INDEX_DIR / "places"
_IDS_FILE         = _INDEX_DIR / "indexed_ids.json"
_ENRICHED_JSON    = Path(__file__).parent.parent / "data" / "places_enriched.json"


# ── ID 영속성 ─────────────────────────────────────────────────────────────────
def _get_indexed_ids() -> set[int]:
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
    """places_enriched.json 전체를 임베딩 후 FAISS 인덱스를 디스크에 저장한다."""
    docs = build_place_documents()

    embeddings  = _make_embeddings(api_key)
    vectorstore = FAISS.from_documents(docs, embeddings)

    _PLACES_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(_PLACES_INDEX_DIR))

    ids = {int(d.metadata["id"]) for d in docs}
    _save_indexed_ids(ids)

    return vectorstore


# ── 로드 ─────────────────────────────────────────────────────────────────────
def load_saved_index(api_key: str) -> FAISS | None:
    """디스크의 FAISS 인덱스를 로드한다. 파일 없으면 None 반환."""
    if not (_PLACES_INDEX_DIR / "index.faiss").exists():
        return None
    embeddings = _make_embeddings(api_key)
    return FAISS.load_local(
        str(_PLACES_INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ── 증분 인덱싱 ───────────────────────────────────────────────────────────────
def add_new_documents(vectorstore: FAISS) -> tuple[FAISS, int]:
    """
    places_enriched.json에서 아직 인덱싱되지 않은 항목(id 기준)만 추가한다.
    전체 재빌드 없이 신규 데이터만 처리.

    반환: (updated_vectorstore, 추가된 문서 수)
    """
    indexed_ids = _get_indexed_ids()

    with open(_ENRICHED_JSON, encoding="utf-8") as f:
        all_items: list[dict] = json.load(f)

    new_items = [item for item in all_items if int(item["id"]) not in indexed_ids]
    if not new_items:
        return vectorstore, 0

    # 신규 항목만 임시 JSON으로 전달
    import tempfile, os
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", delete=False
    ) as tmp:
        json.dump(new_items, tmp, ensure_ascii=False)
        tmp_path = tmp.name

    try:
        new_docs = build_place_documents(tmp_path)
    finally:
        os.unlink(tmp_path)

    vectorstore.add_documents(new_docs)
    vectorstore.save_local(str(_PLACES_INDEX_DIR))
    _save_indexed_ids(indexed_ids | {int(item["id"]) for item in new_items})

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
        vs, _ = add_new_documents(vs)
        return vs

    return build_and_save(api_key)
