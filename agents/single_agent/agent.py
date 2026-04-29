"""
agents/single_agent/agent.py

최소 Single Agent — 장소 추천 시스템

흐름:
  1. parse_intent()  : LLM → 의도 파악 (course / single)
  2. controller      : if문 분기 → retrieve_places / multi_step_retrieve + build_course
  3. generate_response() : LLM → 자연어 설명 생성
"""

import os
import sys
import re
import json
import itertools
from pathlib import Path

# 05_travel_plan_service 루트 (이 파일 기준 2단계 상위)
_TRAVEL_SVC = Path(__file__).parent.parent.parent
if str(_TRAVEL_SVC) not in sys.path:
    sys.path.insert(0, str(_TRAVEL_SVC))

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from rag.retriever import build_vectorstore_standalone  # type: ignore
from rag.retriever import retrieve_places as _rag_retrieve  # type: ignore


# ── 초기화 ────────────────────────────────────────────────────────────────────

def _load_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        secrets_path = _TRAVEL_SVC / ".streamlit" / "secrets.toml"
        if secrets_path.exists():
            m = re.search(
                r'OPENAI_API_KEY\s*=\s*["\']([^"\']+)["\']',
                secrets_path.read_text(encoding="utf-8"),
            )
            if m:
                key = m.group(1)
    return key


_API_KEY = _load_api_key()
_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=_API_KEY)
_VECTORSTORE = None  # lazy init


def _get_vs():
    global _VECTORSTORE
    if _VECTORSTORE is None:
        print("[agent] vectorstore 로딩 중...")
        _VECTORSTORE = build_vectorstore_standalone(_API_KEY)
    return _VECTORSTORE


# ── 검색 함수 ─────────────────────────────────────────────────────────────────

DEFAULT_REGION = "노원"

_FOOD_CATEGORIES = {
    "냉면", "한식", "육류", "갈비", "돈까스", "초밥", "칼국수", "보쌈",
    "해산물", "해물", "두부", "닭요리", "일식", "양식", "베트남", "태국",
    "중식", "치킨", "삼계탕", "참치", "조개", "장어", "떡볶이", "이탈리안",
}


def retrieve_places(query: str, region: str = DEFAULT_REGION) -> list[dict]:
    result = _rag_retrieve(_get_vs(), query, region, top_k=10)
    return result.get("main_places", [])


def multi_step_retrieve(query: str, region: str = DEFAULT_REGION) -> list[dict]:
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


def build_course(places: list[dict]) -> list[list[dict]]:
    cafe  = [p for p in places if "카페" in p.get("카테고리", "")]
    food  = [p for p in places if p.get("카테고리", "") in _FOOD_CATEGORIES]
    other = [p for p in places if p not in cafe and p not in food]

    steps = [s for s in [cafe, food, other] if s]
    if len(steps) < 2:
        return []

    return [list(combo) for combo in list(itertools.product(*steps))[:3]]


# ── LLM 기반 함수 ─────────────────────────────────────────────────────────────

def parse_intent(query: str) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "사용자 쿼리를 분석해 JSON으로만 반환하라. 다른 텍스트 출력 금지.\n"
            '코스/루트/플랜 요청이면 {"type":"course"}, 아니면 {"type":"single"}.'
        )),
        ("human", "{query}"),
    ])
    try:
        raw = (prompt | _LLM | StrOutputParser()).invoke({"query": query})
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip("` \n")
        return json.loads(raw)
    except Exception:
        return {"type": "single"}


def generate_response(query: str, data: dict) -> str:
    strategy = data["strategy"]

    if strategy == "course" and data.get("course"):
        first_course = data["course"][0]
        places_text = "\n".join(
            f"{i+1}단계: {p.get('장소명', '')} ({p.get('카테고리', '')})"
            for i, p in enumerate(first_course)
        )
        label = "코스 추천"
    else:
        places_text = "\n".join(
            f"- {p.get('장소명', '')} ({p.get('카테고리', '')}): {p.get('설명', '')[:40]}"
            for p in data.get("places", [])[:5]
        )
        label = "장소 추천"

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "장소 추천 결과를 바탕으로 자연스럽고 간결한 한국어 설명을 작성하라.\n"
            "코스라면 흐름 중심으로, 단일 추천이라면 각 장소에 이유 한 줄씩. 3~5문장 이내."
        )),
        ("human", "쿼리: {query}\n\n{label}:\n{places_text}"),
    ])
    try:
        return (prompt | _LLM | StrOutputParser()).invoke({
            "query":       query,
            "label":       label,
            "places_text": places_text,
        }).strip()
    except Exception:
        return f"{label}:\n{places_text}"


# ── run_agent ─────────────────────────────────────────────────────────────────

def run_agent(user_query: str, region: str = DEFAULT_REGION) -> dict:
    intent = parse_intent(user_query)

    places: list[dict] = []
    course: list[list[dict]] = []

    if intent["type"] == "course":
        results = multi_step_retrieve(user_query, region)
        course = build_course(results)
        if not course:
            places = retrieve_places(user_query, region)
    else:
        places = retrieve_places(user_query, region)

    strategy = "course" if course else "single"
    data = {"places": places, "course": course, "strategy": strategy}

    reply = generate_response(user_query, data)

    return {
        "reply":    reply,
        "complete": bool(places or course),
        "places":   [_fmt(p) for p in places[:5]],
        "course":   [[_fmt(p) for p in step] for step in course],
        "metadata": {"intent": intent, "strategy": strategy},
    }


def _fmt(p: dict) -> dict:
    return {
        "name":        p.get("장소명", ""),
        "category":    p.get("카테고리", ""),
        "description": p.get("설명", ""),
        "address":     p.get("주소", ""),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("query",    nargs="?", default="노원 데이트 코스 추천해줘")
    parser.add_argument("--region", default=DEFAULT_REGION)
    args = parser.parse_args()
    result = run_agent(args.query, region=args.region)

    print(f"\n{result['reply']}")
    print(f"\n전략: {result['metadata']['strategy']}")

    if result["course"]:
        for i, step in enumerate(result["course"], 1):
            print(f"  코스 {i}: {' → '.join(p['name'] for p in step)}")
    else:
        for p in result["places"]:
            print(f"  - {p['name']} ({p['category']})")
