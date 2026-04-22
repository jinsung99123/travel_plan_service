"""
[3단계] OpenAI 임베딩 + FAISS 벡터 DB 구축
[4단계] 성향 + 지역 기반 장소 검색

[Contextual Retrieval 개선]
  - Contextual Embedding : loader가 생성한 성향/지역/카테고리 컨텍스트 포함 page_content로 임베딩
  - Contextual BM25      : rank_bm25 기반 키워드 검색 (카테고리 + keywords + 설명 토큰화)
  - Hybrid Search        : faiss_weight * FAISS + bm25_weight * BM25 정규화 점수 결합
                           Query Optimization이 쿼리 타입별 weight를 동적으로 조정
                           BM25에 rewrite keywords 추가 토큰으로 강화 가능
  - Metadata Score       : personality_tags·category 불일치 시 점수 penalty 적용
  - Semantic Reranking   : LLM 기반 사용자 의도 적합도 재정렬 (base*(1-rw) + rerank*rw)
"""

import json
import hashlib
import random
import re
from functools import lru_cache
from typing import TYPE_CHECKING

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

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
    faiss_weight: float = 0.6,
    bm25_weight: float = 0.4,
    bm25_extra_tokens: list[str] | None = None,
) -> list[tuple[dict, float]]:
    """
    FAISS + BM25 하이브리드 검색.

    최종 점수 = faiss_weight * normalized_faiss + bm25_weight * normalized_bm25

    처리 흐름:
      1. FAISS top_n 검색 → L2 → 유사도 변환 → min-max 정규화 (query 사용)
      2. BM25 top_n 검색 → min-max 정규화 (query + bm25_extra_tokens 사용)
      3. id 기준 점수 합산 (FAISS에만 있는 문서: faiss_weight만, BM25에만: bm25_weight만)
      4. 최종 점수 내림차순 정렬

    faiss_weight / bm25_weight: Query Optimization이 쿼리 타입별로 동적으로 조정.
      - KEYWORD형  → bm25_weight ↑ (0.7), faiss_weight ↓ (0.3)
      - SEMANTIC형 → faiss_weight ↑ (0.7), bm25_weight ↓ (0.3)
      - MIXED형    → 균등 (0.5 / 0.5)
      weight 합이 1이 아니어도 동작하나, Query Optimization은 합=1을 보장함.

    bm25_extra_tokens: Query Rewrite에서 추출한 키워드를 BM25 쿼리에 추가로 반영.
    """
    bm25, corpus_metadata = build_bm25_index()

    faiss_normalized = _normalize_scores(_faiss_search(vectorstore, query, top_n))

    if bm25_extra_tokens:
        bm25_query = query + " " + " ".join(bm25_extra_tokens)
    else:
        bm25_query = query
    bm25_normalized = _normalize_scores(_bm25_search(bm25, corpus_metadata, bm25_query, top_n))

    combined: dict[int, tuple[dict, float]] = {}

    for meta, score in faiss_normalized:
        doc_id = meta.get("id")
        combined[doc_id] = (meta, score * faiss_weight)

    for meta, score in bm25_normalized:
        doc_id = meta.get("id")
        if doc_id in combined:
            m, s = combined[doc_id]
            combined[doc_id] = (m, s + score * bm25_weight)
        else:
            combined[doc_id] = (meta, score * bm25_weight)

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


# ── 확장 메타데이터 점수 조정 ────────────────────────────────────────────────
_WEATHER_PREFER_INDOOR = {"비", "더위", "추위"}


def _apply_extended_score(
    meta: dict,
    score: float,
    weather: str = "",
    budget: str = "",
    crowd: str = "",
    style: str = "",
) -> float:
    """
    places_extended.csv 6개 필드 기반 추가 점수 조정 (soft scoring).
    모든 조정은 곱셈 배수로 적용하여 기존 점수 체계와 독립적으로 결합.
    """
    if weather in _WEATHER_PREFER_INDOOR:
        io = meta.get("indoor_outdoor", "")
        if io == "실내":
            score *= 1.15
        elif io == "실외":
            score *= 0.80

    if budget and budget != "상관없음":
        if meta.get("price_level") == budget:
            score *= 1.10
        else:
            score *= 0.90

    cl = meta.get("crowd_level", "")
    if crowd == "조용":
        if cl == "낮음":
            score *= 1.10
        elif cl == "높음":
            score *= 0.85
    elif crowd == "활기":
        if cl == "높음":
            score *= 1.10
        elif cl == "낮음":
            score *= 0.90

    stay = meta.get("stay_time", "")
    if style == "빠르게":
        if stay == "30~60":
            score *= 1.10
        elif stay == "120~180":
            score *= 0.85
    elif style == "여유롭게":
        if stay == "120~180":
            score *= 1.10
        elif stay == "30~60":
            score *= 0.90

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


# ── 카테고리 다양성 필터 ─────────────────────────────────────────────────────
def _cap_by_category(places: list[dict], max_per_category: int = 3) -> list[dict]:
    """
    동일 카테고리 과다 방지: 카테고리당 최대 max_per_category개까지만 유지.
    점수 내림차순 정렬 상태를 전제하므로 앞에서부터 순서대로 수집한다.
    """
    counts: dict[str, int] = {}
    result: list[dict] = []
    for meta in places:
        cat = meta.get("카테고리", "")
        if counts.get(cat, 0) < max_per_category:
            result.append(meta)
            counts[cat] = counts.get(cat, 0) + 1
    return result


# ── Semantic Reranking ──────────────────────────────────────────────────────
RERANK_SYSTEM = (
    "너는 여행 추천 시스템의 정밀도를 높이는 랭킹 모델이다.\n\n"
    "사용자의 요청과 각 장소의 적합도를 평가하여 0~1 사이의 점수를 부여하라.\n\n"
    "평가 기준:\n"
    "1. 사용자의 의도와 의미적 유사도 (가장 중요)\n"
    "2. 성향(personality) 적합성\n"
    "3. 날씨 / 예산 / 혼잡도 조건 일치 여부\n"
    "4. 실제 방문 매력도\n\n"
    "점수 기준:\n"
    "- 0.9 이상: 매우 적합\n"
    "- 0.7~0.9: 적합\n"
    "- 0.5~0.7: 보통\n"
    "- 0.5 미만: 부적합\n\n"
    "반드시 JSON 배열만 반환하라. 다른 텍스트 출력 금지.\n"
    '예시: [{"id": 0, "score": 0.92}, {"id": 1, "score": 0.75}]'
)

_rerank_prompt = ChatPromptTemplate.from_messages([
    ("system", RERANK_SYSTEM),
    ("human", "{user_msg}"),
])

# 모듈 레벨 캐시: {cache_key: {idx: score}}
_RERANK_CACHE: dict[str, dict[int, float]] = {}


def _rerank_cache_key(
    query: str,
    candidate_ids: tuple,
    personality: str,
    weather: str,
    budget: str,
    crowd: str,
    style: str,
) -> str:
    raw = f"{query}|{candidate_ids}|{personality}|{weather}|{budget}|{crowd}|{style}"
    return hashlib.md5(raw.encode()).hexdigest()


def rerank_places(
    query: str,
    candidates: list[dict],
    score_map: dict,
    llm: "ChatOpenAI",
    personality: str = "",
    weather: str = "",
    budget: str = "",
    crowd: str = "",
    style: str = "",
    rerank_weight: float = 0.3,
) -> list[dict]:
    """
    LLM 기반 Semantic Reranking.

    - candidates: random.sample() 이후의 metadata list (20~30개)
    - score_map:  {doc_id: base_score} — scoring 단계에서 보존된 원점수
    - 최종 점수:  base_score * 0.7 + rerank_score * 0.3
    - LLM 실패 시 기존 순서 유지 (fallback)
    - 동일 입력에 대한 결과는 모듈 레벨 dict로 캐싱
    """
    if not candidates:
        return candidates

    # ── 캐시 확인 ────────────────────────────────────────────────────────────
    ids_tuple = tuple(c.get("id", i) for i, c in enumerate(candidates))
    cache_key = _rerank_cache_key(query, ids_tuple, personality, weather, budget, crowd, style)
    if cache_key in _RERANK_CACHE:
        rerank_scores = _RERANK_CACHE[cache_key]
    else:
        # ── LLM 호출 ─────────────────────────────────────────────────────────
        places_text = "\n".join(
            f"{idx}. [id={idx}] {c.get('장소명', '')} ({c.get('카테고리', '')}) | "
            f"키워드: {c.get('키워드', '')} | {c.get('설명', '')} | "
            f"체류:{c.get('stay_time', '-')} | 가격:{c.get('price_level', '-')} | "
            f"혼잡:{c.get('crowd_level', '-')} | 실내외:{c.get('indoor_outdoor', '-')}"
            for idx, c in enumerate(candidates)
        )
        user_msg = (
            f"사용자 쿼리: {query}\n"
            f"성향: {personality or '미정'} | 날씨: {weather or '자동'} | "
            f"예산: {budget or '상관없음'} | 혼잡: {crowd or '상관없음'} | "
            f"스타일: {style or '보통'}\n\n"
            f"후보 장소 목록 ({len(candidates)}개):\n{places_text}\n\n"
            f"각 장소에 대해 적합도 점수를 JSON 배열로만 반환하라."
        )
        try:
            chain = _rerank_prompt | llm | StrOutputParser()
            raw = chain.invoke({"user_msg": user_msg}).strip()
            if raw.startswith("```"):
                raw = re.sub(r"```[a-z]*\n?", "", raw).strip("` \n")
            parsed: list[dict] = json.loads(raw)
            rerank_scores = {int(item["id"]): float(item["score"]) for item in parsed}
            _RERANK_CACHE[cache_key] = rerank_scores
        except Exception as exc:
            print(f"[rerank] LLM 응답 파싱 실패: {exc} — 기존 점수 유지")
            return candidates

    # ── 점수 결합 ─────────────────────────────────────────────────────────────
    base_weight = 1.0 - rerank_weight
    scored: list[tuple[dict, float]] = []
    for idx, meta in enumerate(candidates):
        doc_id   = meta.get("id")
        base     = score_map.get(doc_id, 0.5)
        rerank   = rerank_scores.get(idx, 0.5)
        final    = base * base_weight + rerank * rerank_weight
        print(
            f"[rerank log] id={doc_id} 장소={meta.get('장소명', '?')!s:<18} "
            f"base_score={base:.4f}  rerank_score={rerank:.4f}  final_score={final:.4f}"
        )
        scored.append((meta, final))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [m for m, _ in scored]


# ── 메인 검색 함수 ────────────────────────────────────────────────────────────
def retrieve_places(
    vectorstore: FAISS,
    query: str,
    region: str,
    top_k: int = 15,
    personality: str = "",
    weather: str = "",
    budget: str = "",
    crowd: str = "",
    style: str = "",
    llm: "ChatOpenAI | None" = None,
    keywords: list[str] | None = None,
    faiss_weight: float = 0.6,
    bm25_weight: float = 0.4,
    rerank_weight: float = 0.3,
    diversity: float = 0.3,
) -> dict:
    """
    Contextual Hybrid Retrieval — 검색 풀 확장 + 카테고리 다양성 + 대안 후보 반환.

    처리 흐름:
      1. hybrid_search()          — FAISS + BM25 결합 → top_k*6 후보 (풀 확장)
                                    faiss_weight / bm25_weight: Query Optimization이 동적 조정
      2. _apply_metadata_score()  — personality_tags·category penalty
         _apply_extended_score()  — weather·budget·crowd·style bonus/penalty
      3. _region_match()          — 지역 필터 (fallback: 지역 무시)
      4. _cap_by_category()       — 카테고리당 최대 3개 제한
      5. random.sample()          — diversity 기반 풀 크기 조정 후 샘플링
                                    diversity 낮음 → 상위 점수 집중 / 높음 → 넓은 풀에서 다양 추출
      ✅ rerank_places()          — LLM Semantic Reranking (rerank_weight 동적 적용)
      6. _group_by_dong()         — 동선 최적화
      7. candidate_pool           — 카테고리 cap 이후 전체 (대안 추천용)

    반환:
      {
        "main_places":    list[dict],  # 기본 일정 생성용 상위 추천
        "candidate_pool": list[dict],  # 대안 추천·재생성용 전체 후보
      }
    """
    # 1. Hybrid 후보 검색 (풀 확장: top_k*6) — 동적 weight 적용
    candidates = hybrid_search(
        vectorstore, query, top_k * 6,
        faiss_weight=faiss_weight,
        bm25_weight=bm25_weight,
        bm25_extra_tokens=keywords or [],
    )

    # 2. Metadata·Extended scoring (기존 로직 유지)
    candidates = [
        (meta, _apply_extended_score(
            meta,
            _apply_metadata_score(meta, score, personality),
            weather=weather, budget=budget, crowd=crowd, style=style,
        ))
        for meta, score in candidates
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)

    score_map: dict = {meta.get("id"): score for meta, score in candidates}

    # 3. 지역 필터
    filtered = [meta for meta, _ in candidates if _region_match(meta, region)]
    if not filtered:
        filtered = [meta for meta, _ in candidates[:top_k * 2]]

    # 4. 카테고리 다양성 적용 (동일 카테고리 최대 3개)
    diversified = _cap_by_category(filtered, max_per_category=3)

    # 5. diversity 기반 샘플링 풀 조정
    # diversity 0.2(낮음) → 풀 좁게(상위 점수 집중), 0.5(높음) → 풀 넓게(다양성 확보)
    pool_size = min(len(diversified), max(top_k, int(top_k * 2 * (1.0 + diversity))))
    sampled = random.sample(diversified[:pool_size], min(top_k * 2, pool_size))

    # ✅ Semantic Reranking — rerank_weight 동적 적용
    if llm is not None:
        sampled = rerank_places(
            query=query,
            candidates=sampled,
            score_map=score_map,
            llm=llm,
            personality=personality,
            weather=weather,
            budget=budget,
            crowd=crowd,
            style=style,
            rerank_weight=rerank_weight,
        )

    # 6. 동선 최적화
    main_places = _group_by_dong(sampled)

    # 7. candidate_pool: 카테고리 cap 이후 전체 (대안 추천용, 동선 정렬 없음)
    candidate_pool = diversified

    return {
        "main_places":    main_places,
        "candidate_pool": candidate_pool,
    }


# ── 프롬프트 컨텍스트 포맷 ───────────────────────────────────────────────────
def format_place_context(places: list[dict]) -> str:
    """검색된 장소 목록을 LLM 프롬프트에 삽입할 텍스트로 변환한다."""
    lines = []
    for p in places:
        extras = []
        if p.get("best_time"):      extras.append(f"방문시간:{p['best_time']}")
        if p.get("stay_time"):      extras.append(f"체류:{p['stay_time']}분")
        if p.get("price_level"):    extras.append(f"가격:{p['price_level']}")
        if p.get("crowd_level"):    extras.append(f"혼잡:{p['crowd_level']}")
        if p.get("indoor_outdoor"): extras.append(f"실내외:{p['indoor_outdoor']}")
        base = f"- {p['장소명']} ({p['카테고리']}) | 키워드: {p['키워드']} | {p['설명']} | 주소: {p['주소']}"
        lines.append(base + (" | " + " | ".join(extras) if extras else ""))
    return "\n".join(lines)
