"""
[3단계] OpenAI 임베딩 + FAISS 벡터 DB 구축
[4단계] 성향 + 지역 기반 장소 검색

기존 app.py에 영향 없이 독립 모듈로 구현.
벡터 스토어는 Streamlit cache_resource로 앱 기동 시 1회만 빌드.
"""

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


# ── 장소 검색 ────────────────────────────────────────────────────────────────
def retrieve_places(
    vectorstore: FAISS,
    query: str,
    region: str,
    top_k: int = 15,
) -> list[dict]:
    """
    [4단계] 쿼리(성향 키워드)로 유사 장소를 검색하고 지역 필터를 적용한다.

    동작:
      1. 코사인 유사도 기준 top_k * 3개 후보 검색
      2. 주소에 region 포함된 장소만 필터링
      3. 필터 결과가 없으면 지역 필터 없이 top_k개 반환 (fallback)

    반환: metadata dict 리스트 (장소명, 카테고리, 키워드, 설명, 주소)
    """
    candidates = vectorstore.similarity_search(query, k=top_k * 3)

    filtered = [
        doc.metadata
        for doc in candidates
        if region and region in doc.metadata.get("주소", "")
    ]

    if filtered:
        return filtered[:top_k]

    # fallback: 지역 데이터 없음 → 성향만으로 검색된 상위 결과 반환
    return [doc.metadata for doc in candidates[:top_k]]


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
