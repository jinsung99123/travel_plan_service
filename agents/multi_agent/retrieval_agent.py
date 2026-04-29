"""
agents/multi_agent/retrieval_agent.py

Retrieval Agent — 장소 검색 담당

역할:
  Orchestrator로부터 쿼리를 받아 RAG 검색 결과(장소 목록)를 반환한다.
  단일 검색(retrieve_places) 또는 코스용 다단계 검색(multi_step_retrieve) 중 하나를 실행.
  LLM 미사용 — 순수 RAG 검색만 수행.
"""

import os
import re
import sys
from pathlib import Path

# 05_travel_plan_service 루트 (이 파일 기준 2단계 상위)
_TRAVEL_SVC = Path(__file__).parent.parent.parent
if str(_TRAVEL_SVC) not in sys.path:
    sys.path.insert(0, str(_TRAVEL_SVC))

from rag.retriever import build_vectorstore_standalone  # type: ignore
from rag.retriever import retrieve_places as _rag_retrieve  # type: ignore


# ── 초기화 ────────────────────────────────────────────────────────────────────

def _load_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        p = _TRAVEL_SVC / ".streamlit" / "secrets.toml"
        if p.exists():
            m = re.search(
                r'OPENAI_API_KEY\s*=\s*["\']([^"\']+)["\']',
                p.read_text(encoding="utf-8"),
            )
            if m:
                key = m.group(1)
    return key


_API_KEY = _load_api_key()
_VECTORSTORE = None  # lazy init


def _get_vs():
    global _VECTORSTORE
    if _VECTORSTORE is None:
        print("[retrieval_agent] vectorstore 로딩 중...")
        _VECTORSTORE = build_vectorstore_standalone(_API_KEY)
    return _VECTORSTORE


# ── 검색 함수 ─────────────────────────────────────────────────────────────────

def retrieve_places(query: str, region: str = "노원") -> list[dict]:
    result = _rag_retrieve(_get_vs(), query, region, top_k=10)
    return result.get("main_places", [])


def multi_step_retrieve(query: str, region: str = "노원") -> list[dict]:
    sub_queries = [
        f"{query} 카페 감성 디저트",
        f"{query} 맛집 점심 저녁",
        f"{query} 공원 산책 활동 체험",
    ]
    seen_ids: set = set()
    merged: list[dict] = []

    for sq in sub_queries:
        for place in _rag_retrieve(_get_vs(), sq, region, top_k=5).get("main_places", []):
            pid = place.get("id")
            if pid not in seen_ids:
                seen_ids.add(pid)
                merged.append(place)

    return merged


# ── Agent 인터페이스 ───────────────────────────────────────────────────────────

def run_retrieval(query: str, region: str = "노원", is_course: bool = False) -> list[dict]:
    if is_course:
        return multi_step_retrieve(query, region)
    return retrieve_places(query, region)
