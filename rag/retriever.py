"""
[3단계] OpenAI 임베딩 + FAISS 벡터 DB 구축
[4단계] 성향 + 지역 기반 장소 검색

[Contextual Retrieval 개선]
  - Contextual Embedding : loader가 생성한 성향/지역/카테고리 컨텍스트 포함 page_content로 임베딩
  - Contextual BM25      : rank_bm25 기반 키워드 검색 (카테고리 + keywords + 설명 토큰화)
  - Hybrid Search        : FAISS(0.6) + BM25(0.4) 정규화 점수 결합, 동일 문서 합산
  - Metadata Score       : personality_tags·category 불일치 시 점수 penalty 적용
"""

import random
import re
from functools import lru_cache

import streamlit as st
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

from .indexer import get_or_build_vectorstore
from .loader import load_places, load_persona, build_place_documents


# ── 벡터 스토어 ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="장소 데이터를 벡터화하는 중...")
def build_vectorstore(api_key: str) -> FAISS:
    """
    [변경 없음] indexer.get_or_build_vectorstore 위임.
    loader의 contextual page_content로 임베딩 품질 자동 향상.
    인덱스 포맷 변경 시 index/ 디렉터리 삭제 후 재빌드 필요.
    """
    return get_or_build_vectorstore(api_key)


def build_vectorstore_standalone(api_key: str) -> FAISS:
    """Non-Streamlit 환경(테스트/스크립트)용. @cache_resource 없음."""
    return get_or_build_vectorstore(api_key)


# ── BM25 토큰화 ─────────────────────────────────────────────────────────────
def _tokenize_for_bm25(metadata: dict) -> list[str]:
    """
    BM25 문서 토큰화 전략:
    카테고리 + keywords(list) + 설명 → 공백·쉼표 분리 토큰
    page_content(contextual)가 아닌 원본 필드 사용 → 키워드 검색 정확도 유지
    """
    category = metadata.get("카테고리", "")
    keywords = metadata.get("keywords", [])   # 이미 list[str]
    description = metadata.get("설명", "")
    text = f"{category} {' '.join(keywords)} {description}"
    return [t for t in re.split(r"[\s,，.。]+", text) if t]


def _tokenize_query(query: str) -> list[str]:
    """쿼리 토큰화: 공백·쉼표·마침표 분리 (문서 토큰화와 동일 방식)."""
    return [t for t in re.split(r"[\s,，.。]+", query) if t]


# ── BM25 인덱스 빌드 ────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def build_bm25_index() -> tuple:
    """
    BM25Okapi 인덱스와 코퍼스 메타데이터를 빌드한다.

    - lru_cache: 프로세스 전체에서 1회만 빌드 (OpenAI API 호출 없음)
    - Streamlit·스크립트 모두 동일 함수 사용 가능
    - 반환: (BM25Okapi, list[dict])  ← corpus_metadata 순서 = BM25 인덱스 순서
    """
    places_df = load_places()
    persona_df = load_persona()
    docs = build_place_documents(places_df, persona_df)
    corpus_metadata = [doc.metadata for doc in docs]
    tokenized_corpus = [_tokenize_for_bm25(m) for m in corpus_metadata]
    return BM25Okapi(tokenized_corpus), corpus_metadata


# ── 점수 정규화 ─────────────────────────────────────────────────────────────
def _normalize_scores(
    items: list[tuple[dict, float]],
) -> list[tuple[dict, float]]:
    """
    Min-max 정규화 → [0, 1] 범위로 통일.
    FAISS(L2 변환값)와 BM25(자연수 범위) 점수를 동일 스케일로 결합하기 위해 필요.
    모든 점수가 동일하면 1.0으로 통일.
    """
    if not items:
        return items
    scores = [s for _, s in items]
    min_s, max_s = min(scores), max(scores)
    rng = max_s - min_s
    if rng == 0:
        return [(m, 1.0) for m, _ in items]
    return [(m, (s - min_s) / rng) for m, s in items]


# ── 개별 검색 ───────────────────────────────────────────────────────────────
def _faiss_search(
    vectorstore: FAISS,
    query: str,
    top_n: int,
) -> list[tuple[dict, float]]:
    """
    FAISS similarity_search_with_score 호출.
    반환값은 L2 거리(낮을수록 유사) → 1/(1+L2) 변환으로 높을수록 유사하게 뒤집음.
    """
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=top_n)
    return [(doc.metadata, 1.0 / (1.0 + score)) for doc, score in docs_and_scores]


def _bm25_search(
    bm25: BM25Okapi,
    corpus_metadata: list[dict],
    query: str,
    top_n: int,
) -> list[tuple[dict, float]]:
    """
    전체 코퍼스에 대한 BM25 점수 계산 후 상위 top_n 반환.
    get_scores()는 corpus 전체 배열을 반환하므로 argsort 없이 직접 정렬.
    """
    query_tokens = _tokenize_query(query)
    scores = bm25.get_scores(query_tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    return [(corpus_metadata[i], float(scores[i])) for i in top_indices]


# ── Hybrid Search ────────────────────────────────────────────────────────────
def hybrid_search(
    vectorstore: FAISS,
    query: str,
    top_n: int,
    alpha: float = 0.6,
) -> list[tuple[dict, float]]:
    """
    FAISS + BM25 하이브리드 검색.

    최종 점수 = alpha * normalized_embedding + (1-alpha) * normalized_bm25
              = 0.6  * embedding_score       + 0.4       * bm25_score

    처리 흐름:
      1. FAISS top_n 검색 → L2 → 유사도 변환 → min-max 정규화
      2. BM25 top_n 검색 → min-max 정규화
      3. id 기준 점수 합산 (FAISS에만 있는 문서: alpha 가중치만, BM25에만: (1-alpha)만)
      4. 최종 점수 내림차순 정렬
    """
    bm25, corpus_metadata = build_bm25_index()

    faiss_normalized = _normalize_scores(_faiss_search(vectorstore, query, top_n))
    bm25_normalized = _normalize_scores(_bm25_search(bm25, corpus_metadata, query, top_n))

    combined: dict[int, tuple[dict, float]] = {}

    for meta, score in faiss_normalized:
        doc_id = meta.get("id")
        combined[doc_id] = (meta, score * alpha)

    for meta, score in bm25_normalized:
        doc_id = meta.get("id")
        if doc_id in combined:
            m, s = combined[doc_id]
            combined[doc_id] = (m, s + score * (1 - alpha))
        else:
            combined[doc_id] = (meta, score * (1 - alpha))

    return sorted(combined.values(), key=lambda x: x[1], reverse=True)


# ── 메타데이터 기반 점수 조정 ────────────────────────────────────────────────
_ACTIVITY_ALLOWED = {
    "스포츠", "볼링", "헬스", "피트니스", "체육관", "테니스", "수영", "당구장", "체험"
}


def _apply_metadata_score(meta: dict, score: float, personality: str) -> float:
    """
    성향·카테고리 불일치 시 점수 감소 (hard filter 대신 soft penalty).

    - personality_tags에 요청 성향 없음 → × 0.85  (15% 감점)
    - 액티비티형 + 비허용 카테고리     → 추가 × 0.70  (최대 40% 감점)

    결과적으로 불일치 장소는 하위 랭킹으로 밀리지만 완전 제외되지 않음.
    (지역 내 허용 장소가 충분하지 않을 경우 fallback으로 활용)
    """
    if not personality:
        return score
    p_tags = meta.get("personality_tags", [])
    if p_tags and personality not in p_tags:
        score *= 0.85
    if personality == "액티비티형" and meta.get("카테고리") not in _ACTIVITY_ALLOWED:
        score *= 0.70
    return score


# ── 지역 필터 ────────────────────────────────────────────────────────────────
def _region_match(metadata: dict, region: str) -> bool:
    """sigungu 정확 일치 → region 포함 → 주소 포함 순으로 시도 (하위 호환 포함)."""
    if not region:
        return True
    if metadata.get("sigungu") == region:
        return True
    if region in metadata.get("region", ""):
        return True
    if region in metadata.get("주소", ""):
        return True
    return False


# ── 동선 최적화 ─────────────────────────────────────────────────────────────
def _extract_dong(address: str) -> str:
    match = re.search(r"(\S+동)", address)
    return match.group(1) if match else ""


def _group_by_dong(places: list[dict]) -> list[dict]:
    """같은 동 단위 장소끼리 인접하도록 재정렬. 동 없는 장소는 뒤로."""
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)
    no_dong: list[dict] = []
    for p in places:
        dong = _extract_dong(p.get("주소", ""))
        if dong:
            groups[dong].append(p)
        else:
            no_dong.append(p)
    result: list[dict] = []
    for dong_places in groups.values():
        result.extend(dong_places)
    result.extend(no_dong)
    return result


# ── 메인 검색 함수 ────────────────────────────────────────────────────────────
def retrieve_places(
    vectorstore: FAISS,
    query: str,
    region: str,
    top_k: int = 15,
    personality: str = "",
) -> list[dict]:
    """
    Contextual Hybrid Retrieval — 기존 시그니처 유지, 내부 검색 로직 전면 개선.

    처리 흐름:
      1. hybrid_search()  — FAISS(contextual embedding) + BM25 결합 → top_k*3 후보
      2. _apply_metadata_score() — personality_tags·category penalty → 재정렬
      3. _region_match()  — sigungu 기준 지역 필터 (fallback: 지역 무시)
      4. random.sample()  — 상위 풀에서 다양성 확보
      5. _group_by_dong() — 동선 최적화
    """
    # 1. Hybrid 후보 검색
    candidates = hybrid_search(vectorstore, query, top_k * 3)

    # 2. 메타데이터 penalty 적용 후 재정렬
    candidates = [
        (meta, _apply_metadata_score(meta, score, personality))
        for meta, score in candidates
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)

    # 3. 지역 필터
    filtered = [meta for meta, _ in candidates if _region_match(meta, region)]
    if not filtered:
        # fallback: 지역 무시하고 상위 결과 반환
        filtered = [meta for meta, _ in candidates[:top_k]]

    # 4. 랜덤 샘플링 (다양성)
    pool_size = min(len(filtered), top_k + 5)
    selected = random.sample(filtered[:pool_size], min(top_k, pool_size))

    # 5. 동선 최적화
    return _group_by_dong(selected)


# ── 프롬프트 컨텍스트 포맷 ───────────────────────────────────────────────────
def format_place_context(places: list[dict]) -> str:
    """검색된 장소 목록을 LLM 프롬프트에 삽입할 텍스트로 변환한다."""
    lines = []
    for p in places:
        lines.append(
            f"- {p['장소명']} ({p['카테고리']}) | 키워드: {p['키워드']} | {p['설명']} | 주소: {p['주소']}"
        )
    return "\n".join(lines)
