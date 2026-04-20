"""
[3단계] OpenAI 임베딩 + FAISS 벡터 DB 구축
[4단계] 성향 + 지역 기반 장소 검색

기존 app.py에 영향 없이 독립 모듈로 구현.
벡터 스토어는 Streamlit cache_resource로 앱 기동 시 1회만 빌드.
"""

import random
import re

import streamlit as st
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .loader import load_places


# ── 문서 변환 ────────────────────────────────────────────────────────────────
def _build_documents(df: pd.DataFrame) -> list[Document]:
    """
    places.csv 각 행을 LangChain Document로 변환한다.
    page_content: 장소명 + 카테고리 + 키워드 + 설명 (벡터 검색 대상)
    metadata    : 원본 컬럼 전체 (검색 결과 반환용)
    """
    docs = []
    for _, row in df.iterrows():
        text = f"{row['장소명']} {row['카테고리']} {row['키워드']} {row['설명']}"
        metadata = {
            "id":    int(row["id"]),
            "장소명": str(row["장소명"]),
            "카테고리": str(row["카테고리"]),
            "키워드":  str(row["키워드"]),
            "설명":   str(row["설명"]),
            "주소":   str(row["주소"]),
        }
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


# ── 벡터 스토어 (앱 기동 시 1회 빌드, 이후 재사용) ─────────────────────────
@st.cache_resource(show_spinner="장소 데이터를 벡터화하는 중...")
def build_vectorstore(api_key: str) -> FAISS:
    """
    [3단계] places.csv 전체를 OpenAI 임베딩으로 변환하여 FAISS 인덱스 생성.
    api_key를 파라미터로 받아 캐시 키로 사용 (키 변경 시 재빌드).
    """
    df = load_places()
    docs = _build_documents(df)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return FAISS.from_documents(docs, embeddings)


# ── 액티비티형 허용 카테고리 ─────────────────────────────────────────────────
_ACTIVITY_ALLOWED = {"스포츠", "볼링", "헬스", "피트니스", "체육관", "테니스", "수영", "당구장", "체험"}


def _extract_dong(address: str) -> str:
    """주소에서 '동' 단위 지역명을 추출한다 (동선 그룹핑용)."""
    match = re.search(r"(\S+동)", address)
    return match.group(1) if match else ""


def _group_by_dong(places: list[dict]) -> list[dict]:
    """
    같은 동 단위 장소끼리 인접하도록 재정렬한다.
    동이 없는 장소는 뒤로 밀린다.
    """
    from collections import defaultdict
    groups: dict[str, list[dict]] = defaultdict(list)
    no_dong: list[dict] = []
    for p in places:
        dong = _extract_dong(p.get("주소", ""))
        if dong:
            groups[dong].append(p)
        else:
            no_dong.append(p)
    result = []
    for dong_places in groups.values():
        result.extend(dong_places)
    result.extend(no_dong)
    return result


# ── 장소 검색 ────────────────────────────────────────────────────────────────
def retrieve_places(
    vectorstore: FAISS,
    query: str,
    region: str,
    top_k: int = 15,
    personality: str = "",
) -> list[dict]:
    """
    [4단계] 쿼리(성향 키워드)로 유사 장소를 검색하고 지역 필터를 적용한다.

    동작:
      1. 코사인 유사도 기준 top_k * 3개 후보 검색
      2. 주소에 region 포함된 장소만 필터링
      3. 액티비티형이면 허용 카테고리만 유지 (2단계 개선)
      4. 상위 top_k+5개 풀에서 랜덤 샘플링 → top_k개 선택 (4단계 개선)
      5. 동 단위 그룹핑으로 동선 최적화 (3단계 개선)
      6. 필터 결과가 없으면 지역 필터 없이 top_k개 반환 (fallback)

    반환: metadata dict 리스트 (장소명, 카테고리, 키워드, 설명, 주소)
    """
    candidates = vectorstore.similarity_search(query, k=top_k * 3)

    filtered = [
        doc.metadata
        for doc in candidates
        if region and region in doc.metadata.get("주소", "")
    ]

    if not filtered:
        # fallback: 지역 데이터 없음 → 성향만으로 검색된 상위 결과 반환
        return [doc.metadata for doc in candidates[:top_k]]

    # [2단계] 액티비티형 카테고리 필터
    if personality == "액티비티형":
        activity_filtered = [
            p for p in filtered
            if p.get("카테고리", "") in _ACTIVITY_ALLOWED
        ]
        if activity_filtered:
            filtered = activity_filtered

    # [4단계] 랜덤 다양성: 상위 풀에서 샘플링
    pool_size = min(len(filtered), top_k + 5)
    pool = filtered[:pool_size]
    selected = random.sample(pool, min(top_k, len(pool)))

    # [3단계] 동선 최적화: 동 단위 그룹핑
    return _group_by_dong(selected)


def build_vectorstore_standalone(api_key: str) -> FAISS:
    """Non-Streamlit version for testing/scripting — no @cache_resource."""
    df = load_places()
    docs = _build_documents(df)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return FAISS.from_documents(docs, embeddings)


def format_place_context(places: list[dict]) -> str:
    """
    검색된 장소 목록을 LLM 프롬프트에 삽입할 텍스트로 변환한다.
    """
    lines = []
    for p in places:
        lines.append(
            f"- {p['장소명']} ({p['카테고리']}) | 키워드: {p['키워드']} | {p['설명']} | 주소: {p['주소']}"
        )
    return "\n".join(lines)
