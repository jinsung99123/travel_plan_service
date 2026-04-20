"""
QA 테스트 스크립트 — Streamlit 없이 핵심 RAG 로직 직접 실행
"""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

# secrets.toml에서 키 읽기
from pathlib import Path
import tomllib

secrets_path = Path(__file__).parent / ".streamlit" / "secrets.toml"
with open(secrets_path, "rb") as f:
    secrets = tomllib.load(f)
api_key = secrets.get("OPENAI_API_KEY", "")
os.environ["OPENAI_API_KEY"] = api_key

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag.loader import get_personality_keywords
from rag.retriever import build_vectorstore_standalone, retrieve_places, format_place_context
from rag.validator import validate_itinerary

# ── LLM ──────────────────────────────────────────────────────────────────────
def get_llm(temperature=0.3):
    return ChatOpenAI(model="gpt-4o", temperature=temperature, openai_api_key=api_key)

# [4단계] 일정 생성 temperature: 0.5 → 0.8 (다양성 증가)

# ── 성향 분류 ─────────────────────────────────────────────────────────────────
PERSONALITY_TYPES = ["힐링형", "액티비티형", "탐방형", "미식형", "감성형", "균형형"]

ANALYZE_SYSTEM = (
    "너는 여행 성향 분석 전문가다.\n"
    "다음 정의된 성향 타입 중 하나만 출력하라:\n"
    "힐링형, 액티비티형, 탐방형, 미식형, 감성형, 균형형\n\n"
    "규칙:\n"
    "- 가장 지배적인 성향 하나만 선택\n"
    "- 반드시 성향 코드만 단독 출력 (예: 힐링형)\n"
    "- 추가 설명 절대 금지\n"
    "- 정의된 6가지 성향에 포함되지 않을 경우 '여행자'로 처리"
)

def classify_personality(user_input: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", ANALYZE_SYSTEM),
        ("human", "{input}"),
    ])
    chain = prompt | get_llm() | StrOutputParser()
    result = chain.invoke({"input": user_input}).strip()
    return result if result in PERSONALITY_TYPES else "여행자"

# ── 일정 생성 ─────────────────────────────────────────────────────────────────
ITINERARY_SYSTEM_RAG = (
    "너는 여행 일정 생성 전문가다.\n\n"
    "규칙:\n"
    "- 여행 성향을 최우선 반영할 것\n"
    "- 아래 제공된 장소 목록에서만 선택하여 일정을 구성하라\n"
    "- 목록에 없는 장소는 절대 사용 금지\n"
    "- 하루에 장소는 3~4개 배치하라\n"
    "- 현실적인 동선 고려\n"
    "- 시간 흐름 고려 (오전/오후/저녁)\n\n"
    "출력 형식 (엄격 준수):\n"
    "Day 1:\n"
    "- 장소명: 설명\n\n"
    "출력 규칙:\n"
    "- 반드시 'Day n:' 형식 사용\n"
    "- 각 줄은 '- 장소: 설명' 형식\n"
    "- 특수문자 사용 금지 (이모지, *, # 등 금지)\n"
    "- 일정표 외 텍스트 출력 금지"
)

def generate_itinerary(personality, region, days, retrieved_places):
    place_context = format_place_context(retrieved_places)
    prompt = ChatPromptTemplate.from_messages([
        ("system", ITINERARY_SYSTEM_RAG),
        ("human", (
            "여행 성향: {personality}\n"
            "여행 지역: {region}\n"
            "여행 일수: {days}일\n\n"
            "[참고 장소 목록]\n{place_context}"
        )),
    ])
    chain = prompt | get_llm(temperature=0.8) | StrOutputParser()
    return chain.invoke({
        "personality": personality, "region": region,
        "days": days, "place_context": place_context,
    })

# ── 벡터 스토어 빌드 (1회) ────────────────────────────────────────────────────
print("벡터 스토어 빌드 중...")
vs = build_vectorstore_standalone(api_key)
print("완료\n")

# ── 테스트 케이스 ─────────────────────────────────────────────────────────────
TEST_CASES = [
    {"id": 1, "input": "조용하게 산책하고 카페에서 쉬고 싶어",         "region": "노원", "days": 2},
    {"id": 2, "input": "활동적인 데이트 하고 싶어, 몸 쓰는 거 좋아",  "region": "노원", "days": 1},
    {"id": 3, "input": "맛집 투어 위주로 돌아다니고 싶어",             "region": "노원", "days": 3},
    {"id": 4, "input": "사진 찍기 좋은 감성 카페 많이 가고 싶어",     "region": "노원", "days": 2},
    {"id": 5, "input": "여러 가지 골고루 즐기고 싶어",                 "region": "노원", "days": 2},
]

DIVIDER = "=" * 70

results = []

for tc in TEST_CASES:
    print(f"\n{DIVIDER}")
    print(f"[테스트 {tc['id']}] 입력: \"{tc['input']}\"  |  지역: {tc['region']}  |  {tc['days']}일")
    print(DIVIDER)

    # 1. 성향 분류
    personality = classify_personality(tc["input"])
    kw = get_personality_keywords(personality)
    print(f"\n1. 성향 결과: {personality}")
    print(f"   성향 키워드: {kw}")

    # 2. RAG 검색
    query = f"{personality} {kw} {tc['region']}"
    retrieved = retrieve_places(vs, query, tc["region"], top_k=15, personality=personality)
    print(f"\n2. 검색 결과: {len(retrieved)}개 장소")
    for p in retrieved:
        print(f"   - {p['장소명']} ({p['카테고리']}) | {p['키워드']}")

    # 3. 일정 생성
    itinerary = generate_itinerary(personality, tc["region"], tc["days"], retrieved)
    print(f"\n3. 생성된 일정:")
    for line in itinerary.splitlines():
        print(f"   {line}")

    # 4. 검증
    validation = validate_itinerary(itinerary, retrieved)
    print(f"\n4. 검증 결과:")
    print(f"   전체 장소: {validation['total']}개")
    print(f"   검증 완료: {len(validation['verified'])}개  → {validation['verified']}")
    print(f"   미검증:    {len(validation['unverified'])}개  → {validation['unverified']}")
    print(f"   점수:      {validation['score']:.0%}")
    print(f"   상태:      {'정상' if validation['is_clean'] else '⚠ Hallucination 의심'}")

    results.append({
        "id": tc["id"],
        "input": tc["input"],
        "personality": personality,
        "retrieved_count": len(retrieved),
        "retrieved": retrieved,
        "itinerary": itinerary,
        "validation": validation,
    })

# ── 종합 통계 ─────────────────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("종합 통계")
print(DIVIDER)
for r in results:
    v = r["validation"]
    status = "정상" if v["is_clean"] else f"미검증 {len(v['unverified'])}개"
    print(f"테스트 {r['id']} | 성향: {r['personality']:6s} | "
          f"검색: {r['retrieved_count']}개 | 검증: {v['score']:.0%} | {status}")
