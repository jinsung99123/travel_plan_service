"""
agents/multi_agent/orchestrator.py

Orchestrator Agent — 흐름 제어 담당

아키텍처:
  User Query
    ↓
  [Orchestrator]  ← LLM (intent parsing + 응답 생성)
    ↓
  ├── [Retrieval Agent]  ← RAG 검색 (LLM 미사용)
  └── [Course Agent]     ← 코스 조합 (LLM 미사용)
    ↓
  Response

흐름:
  1. parse_intent()      : LLM → "course" or "single"
  2. run_retrieval()     : Retrieval Agent 호출
  3. run_course()        : Course Agent 호출 (course인 경우만)
  4. generate_response() : LLM → 자연어 응답 생성
"""

import os
import re
import sys
import json
from pathlib import Path

_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from retrieval_agent import run_retrieval          # type: ignore
from course_agent import run_course, validate_course  # type: ignore

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# 05_travel_plan_service 루트 (이 파일 기준 2단계 상위)
_TRAVEL_SVC = Path(__file__).parent.parent.parent
_PROMPT_DIR = _THIS_DIR / "prompts"

_MIN_RESULTS_THRESHOLD = 5


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
_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=_API_KEY)


def _load_prompt(filename: str) -> str:
    return (_PROMPT_DIR / filename).read_text(encoding="utf-8").strip()


# ── 유틸리티 ──────────────────────────────────────────────────────────────────

def detect_region_mismatch(places: list[dict], region: str) -> bool:
    if not places:
        return False
    matches = sum(1 for p in places if region in p.get("주소", ""))
    return (matches / len(places)) < 0.5


def soft_sort(places: list[dict], query: str) -> list[dict]:
    _SORTABLE = ["카페", "공원", "한식", "일식", "양식", "디저트", "볼링", "체육"]
    matched = [kw for kw in _SORTABLE if kw in query]
    if not matched:
        return places
    return sorted(
        places,
        key=lambda p: sum(1 for kw in matched if kw in p.get("카테고리", "")),
        reverse=True,
    )


# ── LLM 기반 함수 ─────────────────────────────────────────────────────────────

def parse_intent(query: str) -> dict:
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=_load_prompt("orchestrator_prompt.txt")),
        ("human", "{query}"),
    ])
    try:
        raw = (prompt | _LLM | StrOutputParser()).invoke({"query": query})
        raw = re.sub(r"```[a-z]*\n?", "", raw).strip("` \n")
        return json.loads(raw)
    except Exception:
        return {"type": "single"}


def generate_response(query: str, data: dict) -> str:
    if data["strategy"] == "course" and data.get("course"):
        places_text = "\n".join(
            f"{i+1}단계: {p.get('장소명', '')} ({p.get('카테고리', '')})"
            for i, p in enumerate(data["course"][0])
        )
        label = "코스 추천"
    else:
        places_text = "\n".join(
            f"- {p.get('장소명', '')} ({p.get('카테고리', '')}): {p.get('설명', '')[:40]}"
            for p in data.get("places", [])[:5]
        )
        label = "장소 추천"

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=_load_prompt("response_prompt.txt")),
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


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_orchestrator(user_query: str, region: str = "노원") -> dict:
    trace: list[str] = []
    agents_used: list[str] = []
    places: list[dict] = []
    course: list[list[dict]] = []
    validation: dict = {"passed": True, "reason": "해당없음(단일추천)"}
    region_mismatch: bool = False

    # Step 1: LLM으로 의도 파악
    intent = parse_intent(user_query)
    trace.append(f"intent_parsed:{intent['type']}")
    print(f"[Orchestrator] intent: {intent['type']}")

    # Step 2: 의도에 따라 Agent 호출
    if intent["type"] == "course":
        places = run_retrieval(user_query, region, is_course=True)
        agents_used.append("retrieval")
        trace.append("retrieval_called:course")
        print(f"[Retrieval] results: {len(places)}")

        region_mismatch = detect_region_mismatch(places, region)
        if region_mismatch:
            trace.append("region_mismatch_detected")
            print("[Orchestrator] region mismatch detected")

        if len(places) < _MIN_RESULTS_THRESHOLD:
            trace.append(f"fallback_insufficient_results:{len(places)}")
            places = run_retrieval(user_query, region, is_course=False)
            places = soft_sort(places, user_query)
            trace.append("retrieval_called:single_fallback")
            print(f"[Retrieval] fallback results: {len(places)}")
        else:
            places = soft_sort(places, user_query)
            course = run_course(places)
            agents_used.append("course")
            trace.append(f"course_called:generated:{len(course)}")
            print(f"[Course] generated: {len(course)}")

            if not course:
                trace.append("fallback_course_build_failed")
                places = run_retrieval(user_query, region, is_course=False)
                places = soft_sort(places, user_query)
                print(f"[Retrieval] fallback results: {len(places)}")
            else:
                validation = validate_course(course)
                if validation["passed"]:
                    trace.append("validation_passed")
                    print("[Validation] passed")
                else:
                    trace.append(f"validation_failed:{validation['reason']}")
                    print(f"[Validation] failed — {validation['reason']}")
                    course = []
                    trace.append("fallback_validation_failed")
                    places = run_retrieval(user_query, region, is_course=False)
                    places = soft_sort(places, user_query)
                    print(f"[Retrieval] fallback results: {len(places)}")
    else:
        places = run_retrieval(user_query, region, is_course=False)
        agents_used.append("retrieval")
        trace.append("retrieval_called:single")
        print(f"[Retrieval] results: {len(places)}")

        region_mismatch = detect_region_mismatch(places, region)
        if region_mismatch:
            trace.append("region_mismatch_detected")
            print("[Orchestrator] region mismatch detected")

        places = soft_sort(places, user_query)

        if len(places) < _MIN_RESULTS_THRESHOLD:
            trace.append("fallback_triggered:single")
            fallback_query = "카페 맛집 추천"
            places = run_retrieval(fallback_query, region, is_course=False)
            places = soft_sort(places, fallback_query)
            trace.append("retrieval_called:single_fallback")
            print(f"[Retrieval] fallback results: {len(places)}")

    strategy = "course" if course else "single"
    data = {"places": places, "course": course, "strategy": strategy}

    # Step 3: LLM으로 응답 생성
    reply = generate_response(user_query, data)

    return {
        "reply":    reply,
        "complete": bool(places or course),
        "places":   [_fmt(p) for p in places[:5]],
        "course":   [[_fmt(p) for p in step] for step in course],
        "metadata": {
            "intent":          intent,
            "agents_used":     agents_used,
            "trace":           trace,
            "validation":      validation,
            "region_mismatch": region_mismatch,
        },
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
    parser = argparse.ArgumentParser(description="Multi Agent Place Recommender")
    parser.add_argument("query",    nargs="?", default="노원 데이트 코스 추천해줘")
    parser.add_argument("--region", default="노원")
    args = parser.parse_args()

    result = run_orchestrator(args.query, region=args.region)

    print(f"\n{result['reply']}")
    print(f"\n사용된 Agent: {result['metadata']['agents_used']}")

    if result["course"]:
        for i, step in enumerate(result["course"], 1):
            print(f"  코스 {i}: {' → '.join(p['name'] for p in step)}")
    else:
        for p in result["places"]:
            print(f"  - {p['name']} ({p['category']})")
