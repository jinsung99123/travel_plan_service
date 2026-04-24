import hashlib
import json
import re
import uuid
import streamlit as st
from datetime import date, datetime, timedelta
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── RAG 모듈 임포트 ───────────────────────────────────────────────────────────
from rag.loader import get_personality_keywords
from rag.retriever import build_vectorstore, retrieve_places, format_place_context
from rag.validator import validate_itinerary

# ── 인증 모듈 임포트 ──────────────────────────────────────────────────────────
from auth import get_auth_headers, logout, require_auth, show_login_ui

# ── 페이지 설정 ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI 여행 플래너",
    page_icon="✈️",
    layout="centered",
)

# ── 상수 ──────────────────────────────────────────────────────────────────────
QUESTIONS = [
    "Q1. 여행을 떠나기 전, 보통 어떻게 준비하는 편인가요?",
    "Q2. 여행 중 이동수단을 선택할 때 어떤 점을 가장 중요하게 생각하나요?",
    "Q3. 누구와 여행하는 것을 선호하시나요? 그 이유도 함께 설명해주세요.",
    "Q4. 여행에서 가장 기대하는 활동이나 경험은 무엇인가요?",
    "Q5. 여행 중 하루 일정을 보통 어떻게 보내는 편인가요?",
    "Q6. 여행 중 사진이나 기록은 어떻게 남기는 편인가요?",
]

PERSONALITY_TYPES = ["힐링형", "액티비티형", "탐방형", "미식형", "감성형", "균형형"]

PERSONALITY_DESC = {
    "힐링형": "조용하고 여유로운 여행을 즐기는 타입",
    "액티비티형": "체험·스포츠·활동적인 여행을 즐기는 타입",
    "탐방형": "관광지·문화유산·도시 탐험을 즐기는 타입",
    "미식형": "음식·맛집 중심의 여행을 즐기는 타입",
    "감성형": "카페·사진·분위기 중심의 여행을 즐기는 타입",
    "균형형": "활동과 휴식의 균형을 추구하는 타입",
}

# ── 세션 상태 초기화 ───────────────────────────────────────────────────────────
defaults = {
    "stage": 1,              # 1: 질문 | 2: 성향분석 | 3: 여행정보 | 4: 일정생성
    "answers": {},
    "personality": None,
    "region": "",
    "start_date": None,
    "end_date": None,
    "itinerary": "",
    "retrieved_places": [],
    "candidate_pool":   [],
    # ── 여행 조건 ────────────────────────────────────────────────────────────
    "weather":          "자동",
    "travel_style":     0.5,
    "budget":           "상관없음",
    "crowd":            "상관없음",
    # ── 인증 상태 ────────────────────────────────────────────────────────────
    "is_logged_in": False,
    "access_token": "",
    "username": "",
    # ── 저장된 일정 ──────────────────────────────────────────────────────────
    "saved_plans": [],
    "viewing_plan_id": None,
    # ── TL;DR 요약 ───────────────────────────────────────────────────────────
    "tldr_summary": "",
    # ── 추천 근거 요약 ────────────────────────────────────────────────────────
    "reason_summary": "",
    # ── 추천 신뢰도 ──────────────────────────────────────────────────────────
    "confidence_result": None,
    # ── 장소 교체 UX ─────────────────────────────────────────────────────────
    "_replace_flash": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── LLM 팩토리 ────────────────────────────────────────────────────────────────
def get_llm(streaming: bool = False) -> ChatOpenAI:
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        st.error("OPENAI_API_KEY가 설정되지 않았습니다. .streamlit/secrets.toml 파일을 확인하세요.")
        st.stop()
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.8,  # [4단계] 다양성 증가 (0.7 → 0.8)
        streaming=streaming,
        openai_api_key=api_key,
    )


# ── 성향 분석 ─────────────────────────────────────────────────────────────────
ANALYZE_SYSTEM = (
    "너는 여행 성향 분석 전문가다.\n"
    "다음 정의된 성향 타입 중 하나만 출력하라:\n"
    "힐링형, 액티비티형, 탐방형, 미식형, 감성형, 균형형\n\n"
    "분석 기준:\n"
    "- 계획성 vs 즉흥성\n"
    "- 활동성 vs 휴식 지향\n"
    "- 도시 vs 자연 선호\n"
    "- 경험 중심 vs 결과 중심\n"
    "- 사회성 vs 독립성\n\n"
    "규칙:\n"
    "- 모든 답변을 종합적으로 분석할 것\n"
    "- 단순 키워드 매칭이 아닌 의미 기반 해석\n"
    "- 가장 지배적인 성향 하나만 선택\n"
    "- 반드시 성향 코드만 단독 출력 (예: 힐링형)\n"
    "- 추가 설명 절대 금지\n"
    "- 다른 문장 포함 금지\n"
    "- 정의된 6가지 성향에 포함되지 않을 경우 '여행자'로 처리"
)

analyze_prompt = ChatPromptTemplate.from_messages([
    ("system", ANALYZE_SYSTEM),
    ("human", "{qa_text}"),
])


def analyze_personality(answers: dict) -> str:
    qa_text = "\n".join(
        f"{QUESTIONS[i]} => {answers.get(f'q{i+1}', '(미응답)')}"
        for i in range(len(QUESTIONS))
    )
    chain = analyze_prompt | get_llm(streaming=False) | StrOutputParser()
    result = chain.invoke({"qa_text": qa_text}).strip()
    return result if result in PERSONALITY_TYPES else "여행자"


# ── 일정 생성 (스트리밍) ──────────────────────────────────────────────────────
ITINERARY_SYSTEM = (
    "너는 여행 일정 생성 전문가다.\n\n"
    "규칙:\n"
    "- 여행 성향을 최우선 반영할 것\n"
    "- 실제 존재하는 장소만 사용할 것\n"
    "- 하루 단위 일정 구성\n"
    "- 현실적인 동선 고려\n"
    "- 시간 흐름 고려 (오전/오후/저녁)\n\n"
    "출력 형식 (엄격 준수):\n"
    "Day 1:\n"
    "- 장소1: 설명\n"
    "- 장소2: 설명\n\n"
    "Day 2:\n"
    "- 장소1: 설명\n"
    "- 장소2: 설명\n\n"
    "출력 규칙:\n"
    "- 반드시 'Day n:' 형식 사용\n"
    "- 각 줄은 '- 장소: 설명' 형식\n"
    "- 특수문자 사용 금지 (이모지, *, # 등 금지)\n"
    "- 불필요한 설명 금지\n"
    "- 일정표 외 텍스트 출력 금지\n"
    "- 추천 장소가 없을 경우 '추천 결과 없음'만 출력"
)

# [수정 3-a] RAG 컨텍스트 주입용 시스템 프롬프트 (신규 추가)
ITINERARY_SYSTEM_RAG = """너는 여행 일정 생성 전문가다.
아래 제공된 [주요 장소 목록]과 [대안 장소 목록]에 있는 장소만 사용하라.
목록에 없는 장소를 임의로 생성하는 것은 절대 금지다.

===== 장소 수 결정 규칙 =====
입력된 일정 스타일(travel_style) 값에 따라 하루 장소 수를 결정하라.
- 0.0 이상 0.3 미만 (빠르게): 하루 2~3개 장소
- 0.3 이상 0.7 미만 (보통):   하루 3~4개 장소
- 0.7 이상 1.0 이하 (여유롭게): 하루 4~5개 장소

===== 시간 배치 규칙 =====
- 하루 시작 시간은 10:00로 고정한다.
- 각 장소의 체류 시간(stay_time)을 누적하여 다음 장소 시작 시간을 계산한다.
  체류 시간 기준:
    · 30~60분  → 45분 사용
    · 60~120분 → 90분 사용
    · 120~180분 → 150분 사용
    · 명시 없음 → 60분 사용
- 이동 시간은 다음 기준으로 추가한다:
    · 같은 동(洞) 내 이동: 10분 추가
    · 다른 동 간 이동: 20분 추가
- 시간 겹침은 절대 금지다. 앞 장소 종료 후 이동 시간을 더한 시점이 다음 장소 시작 시간이다.
- 시간은 HH:MM 형식으로 표기하고, 종료 시간은 시작 + 체류 시간으로 계산한다.

===== 다양성 규칙 =====
- 하루 일정 내 동일 카테고리는 최대 2개까지만 허용한다.
- 하루 일정에는 반드시 2개 이상의 서로 다른 카테고리를 포함해야 한다.
- 날씨가 비/더위/추위인 경우 실내 장소를 가능한 우선 배치하라. 실내 장소가 부족하면 혼합·실외 장소도 포함하라.
- 날씨가 맑음/자동인 경우 실외·실내 구분 없이 자유롭게 선택하라.
- 예산 수준과 혼잡도 선호는 참고 기준이다. 완벽히 맞는 장소가 없어도 제공된 목록에서 가장 적합한 장소를 반드시 선택하라.

===== 추천 이유 규칙 =====
각 장소마다 추천이유를 반드시 2줄로 작성한다.
- 첫 번째 줄: 여행 성향 기반 이유 (예: "{personality}형 여행자에게 어울리는 이유")
- 두 번째 줄: 장소 데이터 기반 이유 (체류 시간, 혼잡도, 실내외, 가격대 등 메타데이터 활용)
추천이유는 구체적이고 사실에 근거해야 하며, 막연한 표현(예: "좋은 곳", "추천할 만한") 금지.

===== 대안 장소 규칙 =====
각 장소마다 [대안 장소 목록]에서 동일 카테고리 장소 2개를 대안으로 선택한다.
동일 카테고리 대안이 없으면 인접 카테고리(예: 카페↔디저트) 기준으로 선택한다.

===== 문장 스타일 규칙 =====
- "~에서 시간을 보냅니다", "~을 즐길 수 있습니다" 같은 반복 표현 금지.
- 각 장소 설명은 서로 다른 문장 구조와 어휘를 사용하라.
- 실제 여행 가이드처럼 감정·분위기·오감을 담아 서술하라.
- 설명은 1~2문장, 간결하되 생생하게.

===== 출력 형식 (엄격 준수) =====
Day 1

HH:MM - HH:MM 장소명
설명: (장소 분위기·특징 1~2문장)
추천이유:
- (성향 기반 이유)
- (데이터 기반 이유)
대안: 장소A, 장소B

HH:MM - HH:MM 장소명
설명: ...
추천이유:
- ...
- ...
대안: 장소A, 장소B

Day 2
...

===== 출력 금지 사항 =====
- 이모지, *, #, ** 등 특수문자 사용 금지
- 위 형식 외 추가 텍스트 출력 금지
- "추천 결과 없음"은 [주요 장소 목록]이 완전히 비어 있을 때만 출력하라. 날씨·혼잡도·예산 조건이 일부 맞지 않아도 제공된 장소 중 최선을 선택해 반드시 일정을 생성하라.
"""

itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", ITINERARY_SYSTEM),
    ("human", "여행 성향: {personality}\n여행 지역: {region}\n여행 일수: {days}일"),
])

# ── 추천 근거 요약 프롬프트 ───────────────────────────────────────────────────
REASON_SUMMARY_SYSTEM = (
    "너는 여행 추천 시스템의 설명 생성기다.\n"
    "사용자의 조건과 추천된 장소 목록을 기반으로\n"
    "왜 이 일정이 구성되었는지 간결하게 설명하라.\n\n"
    "규칙:\n"
    "- 3~5줄 bullet 형태로 작성\n"
    "- 각 줄은 '- ' 으로 시작\n"
    "- 각 줄은 '~을 반영하여 ~ 구성했습니다' 형태\n"
    "- 실제 제공된 장소 데이터에 근거한 사실만 작성\n"
    "- 존재하지 않는 정보나 추측 표현 금지\n"
    "- 중복 표현 금지, 과장 금지\n"
    "- 한국어, 자연스럽고 간결하게\n"
    "- bullet 외 다른 텍스트 출력 금지"
)

reason_summary_prompt = ChatPromptTemplate.from_messages([
    ("system", REASON_SUMMARY_SYSTEM),
    ("human", (
        "성향: {personality}\n"
        "지역: {region}\n"
        "날씨: {weather}\n"
        "예산: {budget}\n"
        "혼잡도: {crowd}\n"
        "일정스타일: {travel_style}\n\n"
        "추천 장소 (상위 5개):\n{top_places}"
    )),
])

# ── TL;DR 요약 프롬프트 ───────────────────────────────────────────────────────
TLDR_SYSTEM = (
    "너는 여행 일정 요약 전문가다.\n"
    "주어진 일정 내용을 핵심만 간결하게 요약하라.\n\n"
    "규칙:\n"
    "- 정확히 4줄만 출력하라\n"
    "- 줄 순서와 이모지·태그를 반드시 지켜라:\n"
    "  줄1: 🌿 컨셉: (여행 전반의 분위기·테마 한 문장)\n"
    "  줄2: 📍 주요 구성: (일정에 등장하는 장소 유형·카테고리 나열)\n"
    "  줄3: ⏱️ 일정 스타일: (이동 여유·밀도 특징 한 문장)\n"
    "  줄4: 💡 특징: (이 일정만의 차별 포인트 한 문장)\n"
    "- 실제 일정 텍스트에 존재하는 장소·내용만 사용\n"
    "- 없는 장소 추가·추측 표현·과장 금지\n"
    "- 중복 표현 금지\n"
    "- 한국어, 자연스럽고 간결하게\n"
    "- 4줄 bullet 외 다른 텍스트 출력 금지"
)

tldr_prompt = ChatPromptTemplate.from_messages([
    ("system", TLDR_SYSTEM),
    ("human", (
        "성향: {personality}\n"
        "일정 스타일: {travel_style}\n\n"
        "일정:\n{itinerary}"
    )),
])

# [수정 3-b] RAG 전용 프롬프트 템플릿 (신규 추가)
itinerary_prompt_rag = ChatPromptTemplate.from_messages([
    ("system", ITINERARY_SYSTEM_RAG),
    ("human", (
        "여행 성향: {personality}\n"
        "여행 일수: {days}일\n"
        "날씨 조건: {weather}\n"
        "일정 스타일(travel_style): {travel_style} (0=빠르게, 1=여유롭게)\n"
        "예산 수준: {budget}\n"
        "혼잡도 선호: {crowd}\n\n"
        "[주요 장소 목록]\n{place_context}\n\n"
        "[대안 장소 목록]\n{candidate_context}"
    )),
])


# [수정 3-c] generate_itinerary에 retrieved_places 파라미터 추가
# 기존 시그니처 유지 (retrieved_places=None → 기존 동작 그대로)
def _time_emoji(time_str: str) -> str:
    """'HH:MM' 문자열을 시간대 이모지로 변환한다."""
    try:
        hour = int(time_str.split(":")[0])
    except (ValueError, IndexError):
        return "📍"
    if hour < 11:
        return "🌅"
    if hour < 13:
        return "🍽️"
    if hour < 17:
        return "☕"
    return "🌙"


def _time_label(time_str: str) -> str:
    """'HH:MM' 문자열을 한글 시간대 라벨로 변환한다."""
    try:
        hour = int(time_str.split(":")[0])
    except (ValueError, IndexError):
        return ""
    if hour < 11:
        return "오전"
    if hour < 13:
        return "점심"
    if hour < 17:
        return "오후"
    return "저녁"


def _parse_itinerary(itinerary: str) -> list[dict]:
    """
    일정 텍스트를 Day 단위 구조체로 파싱한다.

    반환값: [
      {
        "day": 1,
        "places": [
          {
            "time_range": "10:00 - 11:30",
            "start_time": "10:00",
            "name": "장소명",
            "description": "설명 텍스트",
            "reasons": ["이유1", "이유2"],
            "alternatives": "장소A, 장소B",
          }, ...
        ]
      }, ...
    ]
    """
    days: list[dict] = []
    current_day: dict | None = None
    current_place: dict | None = None

    def _flush_place():
        if current_day is not None and current_place is not None:
            current_day["places"].append(current_place)

    in_reasons = False
    for raw in itinerary.splitlines():
        line = raw.strip()
        if not line:
            in_reasons = False
            continue

        day_m = re.match(r"^Day\s+(\d+)", line, re.IGNORECASE)
        if day_m:
            _flush_place()
            current_day = {"day": int(day_m.group(1)), "places": []}
            days.append(current_day)
            current_place = None
            in_reasons = False
            continue

        time_m = re.match(r"^(\d{2}:\d{2})\s*-\s*(\d{2}:\d{2})\s+(.+)$", line)
        if time_m:
            _flush_place()
            current_place = {
                "time_range":   f"{time_m.group(1)} - {time_m.group(2)}",
                "start_time":   time_m.group(1),
                "name":         time_m.group(3).strip(),
                "description":  "",
                "reasons":      [],
                "alternatives": "",
            }
            in_reasons = False
            continue

        if current_place is None:
            continue

        if line.startswith("설명:"):
            current_place["description"] = line[3:].strip()
            in_reasons = False
        elif line.startswith("추천이유"):
            in_reasons = True
        elif in_reasons and line.startswith("-"):
            current_place["reasons"].append(line[1:].strip())
        elif line.startswith("대안:"):
            current_place["alternatives"] = line[3:].strip()
            in_reasons = False
        elif not in_reasons and current_place["description"]:
            current_place["description"] += " " + line

    _flush_place()
    return days


def render_timeline(itinerary: str, interactive: bool = False) -> None:
    """파싱된 일정을 시간 흐름 기반 타임라인 UI로 렌더링한다."""
    days = _parse_itinerary(itinerary)

    if not days:
        st.code(itinerary, language=None)
        return

    # 장소 교체 후 플래시 메시지 표시
    if interactive:
        flash = st.session_state.get("_replace_flash", "")
        if flash:
            st.session_state["_replace_flash"] = ""
            st.success(flash)

    for day_data in days:
        st.markdown(
            f"<h3 style='margin:24px 0 8px 0; color:#3a3f5c;'>📆 Day {day_data['day']}</h3>",
            unsafe_allow_html=True,
        )

        for place_idx, place in enumerate(day_data["places"]):
            emoji   = _time_emoji(place["start_time"])
            label   = _time_label(place["start_time"])
            tr      = place["time_range"]
            name    = place["name"]
            desc    = place["description"]
            alts    = place["alternatives"]
            reasons = place["reasons"]

            reason_html = ""
            if reasons:
                items = "".join(
                    f"<li style='margin:2px 0; color:#555;'>{r}</li>"
                    for r in reasons
                )
                reason_html = (
                    f"<ul style='margin:6px 0 0 0; padding-left:18px;'>{items}</ul>"
                )

            alt_html = ""
            if alts:
                alt_html = (
                    f"<p style='margin:6px 0 0 0; font-size:12px; color:#888;'>"
                    f"🔀 대안: {alts}</p>"
                )

            card_html = f"""
                <div style="
                    display:flex; gap:14px; align-items:flex-start;
                    background:#ffffff; border:1px solid #e0e6f0;
                    border-left:4px solid #4a7afe;
                    border-radius:10px; padding:14px 16px;
                    margin-bottom:10px;
                ">
                  <div style="text-align:center; min-width:48px;">
                    <div style="font-size:28px;">{emoji}</div>
                    <div style="font-size:10px; color:#888; margin-top:2px;">{label}</div>
                  </div>
                  <div style="flex:1;">
                    <div style="font-size:11px; color:#aaa; margin-bottom:2px;">{tr}</div>
                    <div style="font-size:16px; font-weight:700; color:#1a1a2e;">{name}</div>
                    <div style="font-size:13px; color:#444; margin-top:4px;">{desc}</div>
                    {reason_html}
                    {alt_html}
                  </div>
                </div>
                """

            if interactive:
                col_card, col_btn = st.columns([9, 1])
                with col_card:
                    st.markdown(card_html, unsafe_allow_html=True)
                with col_btn:
                    st.write("")
                    btn_key = f"replace_{day_data['day']}_{place_idx}"
                    if st.button("🔄", key=btn_key, help=f"'{name}' 을(를) 다른 장소로 교체"):
                        replaced = _replace_place(name)
                        if not replaced:
                            st.session_state["_replace_flash"] = (
                                f"⚠️ '{name}' 을(를) 대체할 후보 장소가 없습니다."
                            )
                        st.rerun()
            else:
                st.markdown(card_html, unsafe_allow_html=True)

        st.divider()


def save_plan(days: int, title: str = "") -> None:
    """현재 session_state의 일정을 saved_plans에 저장한다."""
    personality = st.session_state.get("personality", "미정")
    region      = st.session_state.get("region", "미입력")
    plan = {
        "id":           str(uuid.uuid4()),
        "title":        title or f"{personality} · {region} {days}일",
        "personality":  personality,
        "region":       region,
        "days":         days,
        "weather":      st.session_state.get("weather", "자동"),
        "budget":       st.session_state.get("budget", "상관없음"),
        "travel_style": float(st.session_state.get("travel_style", 0.5)),
        "crowd":        st.session_state.get("crowd", "상관없음"),
        "itinerary":    st.session_state.get("itinerary", ""),
        "tldr_summary": st.session_state.get("tldr_summary", ""),
        "created_at":   datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    st.session_state["saved_plans"].insert(0, plan)  # 최신순


def load_plan(plan_id: str) -> None:
    """저장된 일정을 현재 활성 일정으로 불러온다."""
    plan = next(
        (p for p in st.session_state["saved_plans"] if p["id"] == plan_id),
        None,
    )
    if not plan:
        return
    st.session_state.itinerary      = plan["itinerary"]
    st.session_state.personality    = plan["personality"]
    st.session_state.region         = plan["region"]
    st.session_state.weather        = plan.get("weather", "자동")
    st.session_state.budget         = plan.get("budget", "상관없음")
    st.session_state.travel_style   = plan.get("travel_style", 0.5)
    st.session_state.crowd          = plan.get("crowd", "상관없음")
    st.session_state["tldr_summary"]       = plan.get("tldr_summary", "")
    st.session_state["reason_summary"]     = ""
    st.session_state["confidence_result"]  = None
    st.session_state.retrieved_places      = []
    st.session_state.candidate_pool        = []
    st.session_state.pop("validation", None)
    st.session_state.stage               = 4
    st.session_state["viewing_plan_id"]  = None


def render_saved_plans() -> None:
    """사이드바에 저장된 일정 목록을 렌더링한다."""
    plans: list[dict] = st.session_state.get("saved_plans", [])

    st.sidebar.divider()
    header_col, clear_col = st.sidebar.columns([3, 1])
    with header_col:
        st.sidebar.markdown(f"**📁 저장된 일정** ({len(plans)}개)")
    with clear_col:
        if plans and st.sidebar.button("전체삭제", key="clear_all_plans"):
            st.session_state["saved_plans"] = []
            st.session_state["viewing_plan_id"] = None
            st.rerun()

    if not plans:
        st.sidebar.caption("저장된 일정이 없습니다.")
        return

    # 전체 JSON 내보내기
    st.sidebar.download_button(
        label="📥 전체 내보내기 (JSON)",
        data=json.dumps(plans, ensure_ascii=False, indent=2),
        file_name="travel_plans_all.json",
        mime="application/json",
        use_container_width=True,
        key="export_all_plans",
    )

    for plan in plans:
        with st.sidebar.container():
            # 제목 (타이틀 우선, 없으면 구 형식 폴백)
            title = plan.get("title") or f"{plan['personality']} · {plan['region']} {plan['days']}일"
            st.sidebar.markdown(
                f"**{title}**  \n"
                f"<span style='font-size:11px;color:#888;'>{plan['created_at']}</span>",
                unsafe_allow_html=True,
            )
            view_col, load_col, del_col = st.sidebar.columns([2, 2, 1])
            with view_col:
                if st.button("보기", key=f"view_{plan['id']}"):
                    st.session_state["viewing_plan_id"] = plan["id"]
                    st.rerun()
            with load_col:
                if st.button("불러오기", key=f"load_{plan['id']}"):
                    load_plan(plan["id"])
                    st.rerun()
            with del_col:
                if st.button("삭제", key=f"del_{plan['id']}"):
                    st.session_state["saved_plans"] = [
                        p for p in st.session_state["saved_plans"]
                        if p["id"] != plan["id"]
                    ]
                    if st.session_state.get("viewing_plan_id") == plan["id"]:
                        st.session_state["viewing_plan_id"] = None
                    st.rerun()
            st.sidebar.markdown("---")


def _style_label(travel_style: float) -> str:
    """0.0~1.0 슬라이더 값을 프롬프트용 텍스트 라벨로 변환한다."""
    if travel_style < 0.4:
        return "빠르게"
    if travel_style > 0.6:
        return "여유롭게"
    return "보통"


# ── Query Rewrite + Intent Parser ────────────────────────────────────────────
_REWRITE_SYSTEM_MSG = SystemMessage(content=
    "너는 RAG 기반 여행 추천 시스템의 Query Rewrite + Intent Parser이다.\n\n"
    "사용자의 자연어 요청을 분석하여 아래 5단계를 순서대로 수행하고,\n"
    "결과를 단일 JSON으로만 출력하라. 다른 텍스트 출력 금지.\n\n"

    "【STEP 1】 query_type 분류\n"
    "  KEYWORD  : 명사형 단어 나열, 토큰 ≤3, 키워드 ≤2  (예: '노원 카페')\n"
    "  SEMANTIC : 추상적·감성적 자연어, 장소 카테고리 미포함 (예: '혼자 쉬고 싶어')\n"
    "  MIXED    : 카테고리 키워드 + 자연어 혼합 (예: '조용한 카페에서 책 읽고 싶어')\n"
    "  COMPLEX  : '도 하고', '그리고', '뿐만 아니라', '면서' 등 복합 연결 패턴 포함\n\n"

    "【STEP 2】 rewritten_query 생성\n"
    "  - FAISS 임베딩 검색에 유리한 자연어 문장으로 재작성하라.\n"
    "  - 장소 유형(카페, 음식점, 공원 등), 분위기(조용한, 감성, 활기찬 등),\n"
    "    활동(산책, 사진, 데이트 등) 키워드를 포함하라.\n"
    "  - '좀', '약간', '괜찮은', '느낌' 등 검색에 불필요한 표현을 제거하라.\n"
    "  - '쉬고 싶다' → '조용한 카페 힐링 휴식 공간'처럼 구체적 장소·활동으로 확장하라.\n"
    "  - COMPLEX형은 전체를 아우르는 통합 검색 문장 1개를 생성하라.\n\n"

    "【STEP 3】 filters 추출\n"
    "  아래 필드를 추출하라. 언급 없으면 null 또는 빈 리스트.\n"
    "  - category    : 장소 유형 (카페, 음식점, 공원, 헬스, 볼링, 공연장 등) — 리스트\n"
    "  - purpose     : 목적/동반 유형 (데이트, 혼자, 친구, 가족, 활동, 힐링) — 리스트\n"
    "  - weather     : 날씨 언급 시 (맑음, 흐림, 비, 눈) — null 또는 문자열\n"
    "  - indoor_outdoor : 실내/실외 명시 시 (실내, 실외, 실내외) — null 또는 문자열\n"
    "  - crowd       : 혼잡도 언급 시 (높음, 보통, 낮음) — null 또는 문자열\n"
    "  - budget      : 예산 언급 시 (고가, 중가, 저가) — null 또는 문자열\n"
    "      · 저렴한, 싸게, 가성비, 알뜰 → \"저가\"\n"
    "      · 비싼, 고급, 프리미엄, 럭셔리 → \"고가\"\n"
    "      · 적당한 가격, 보통 → \"중가\"\n"
    "  - style        : 분위기/스타일 (조용한, 감성, 활기찬, 힐링, 여유) — null 또는 문자열\n"
    "  - visual_intent: 감성, 분위기, 사진, 인스타, SNS, 예쁜, 데이트 키워드 포함 시 true — boolean\n\n"

    "【STEP 4】 sub_queries 생성 (COMPLEX 전용)\n"
    "  - 독립 검색 가능한 하위 쿼리 최대 3개로 분해하라.\n"
    "  - 의미 중복 금지, 각 쿼리는 장소 유형과 목적이 명확해야 한다.\n"
    "  - COMPLEX가 아니면 빈 리스트 []를 반환하라.\n\n"

    "【STEP 5】 출력 제약\n"
    "  - JSON 외 텍스트 금지. 마크다운 코드블록 금지.\n"
    "  - 모든 문자열 값은 한국어.\n\n"

    "출력 형식:\n"
    '{"original_query": "원본 쿼리",'
    ' "rewritten_query": "검색 최적화 문장",'
    ' "query_type": "KEYWORD|SEMANTIC|MIXED|COMPLEX",'
    ' "filters": {"category": [], "purpose": [], "weather": null,'
    ' "indoor_outdoor": null, "crowd": null, "budget": null, "style": null},'
    ' "visual_intent": false,'
    ' "sub_queries": []}\n\n'

    "입출력 예시:\n"
    '입력: "노원 카페"\n'
    '출력: {"original_query": "노원 카페", "rewritten_query": "노원구 카페 조용한 분위기",'
    ' "query_type": "KEYWORD", "filters": {"category": ["카페"], "purpose": [],'
    ' "weather": null, "indoor_outdoor": null, "crowd": null, "budget": null, "style": null},'
    ' "visual_intent": false, "sub_queries": []}\n\n'
    '입력: "혼자 조용히 쉬고 싶어"\n'
    '출력: {"original_query": "혼자 조용히 쉬고 싶어",'
    ' "rewritten_query": "혼자 조용한 카페 힐링 여유 휴식 공간",'
    ' "query_type": "SEMANTIC", "filters": {"category": [], "purpose": ["혼자", "힐링"],'
    ' "weather": null, "indoor_outdoor": null, "crowd": null, "budget": null, "style": "조용한"},'
    ' "visual_intent": false, "sub_queries": []}\n\n'
    '입력: "카페에서 커피 마시고 공원도 산책하고 싶어"\n'
    '출력: {"original_query": "카페에서 커피 마시고 공원도 산책하고 싶어",'
    ' "rewritten_query": "카페 커피 공원 산책 힐링 여유",'
    ' "query_type": "COMPLEX", "filters": {"category": ["카페", "공원"], "purpose": ["힐링"],'
    ' "weather": null, "indoor_outdoor": null, "crowd": null, "budget": null, "style": null},'
    ' "visual_intent": false, "sub_queries": ["조용한 카페 커피 여유", "공원 산책 자연 힐링"]}\n\n'
    '입력: "가성비 좋은 맛집"\n'
    '출력: {"original_query": "가성비 좋은 맛집",'
    ' "rewritten_query": "저렴한 가성비 맛집 저가 식당",'
    ' "query_type": "MIXED", "filters": {"category": ["음식점"], "purpose": ["친구"],'
    ' "weather": null, "indoor_outdoor": null, "crowd": null, "budget": "저가", "style": null},'
    ' "visual_intent": false, "sub_queries": []}\n\n'
    '입력: "감성 카페 추천"\n'
    '출력: {"original_query": "감성 카페 추천",'
    ' "rewritten_query": "감성 분위기 사진 찍기 좋은 카페 인스타 SNS",'
    ' "query_type": "MIXED", "filters": {"category": ["카페"], "purpose": ["데이트"],'
    ' "weather": null, "indoor_outdoor": null, "crowd": null, "budget": null, "style": "감성"},'
    ' "visual_intent": true, "sub_queries": []}'
)

_rewrite_prompt = ChatPromptTemplate.from_messages([
    _REWRITE_SYSTEM_MSG,
    ("human", '사용자 쿼리:\n"{user_query}"'),
])

# 모듈 레벨 캐시: {md5_key: result_dict}
_REWRITE_CACHE: dict[str, dict] = {}


def _rewrite_cache_key(query: str, personality: str, region: str) -> str:
    raw = f"{query}|{personality}|{region}"
    return hashlib.md5(raw.encode()).hexdigest()


def rewrite_query(
    user_input: str,
    personality: str = "",
    region: str = "",
) -> dict:
    """
    사용자 쿼리를 검색 최적화 형태로 재작성하고 의도를 파싱한다.

    반환:
      {
        "original_query":  원본 쿼리,
        "rewritten_query": 검색 최적화 문장 (FAISS 임베딩 입력),
        "query_type":      "KEYWORD" | "SEMANTIC" | "MIXED" | "COMPLEX",
        "filters":         {category, purpose, weather, indoor_outdoor, crowd, budget, style},
        "sub_queries":     COMPLEX형 하위 쿼리 리스트 (그 외 []),
        "keywords":        filters["category"] 기반 BM25 추가 토큰 (하위 호환),
      }
    LLM 실패 또는 API 키 없음 시 원본 쿼리 그대로 반환 (fallback).
    동일 입력은 모듈 레벨 dict로 캐싱하여 LLM 재호출 방지.
    """
    cache_key = _rewrite_cache_key(user_input, personality, region)
    if cache_key in _REWRITE_CACHE:
        return _REWRITE_CACHE[cache_key]

    _empty_filters = {
        "category": [], "purpose": [], "weather": None,
        "indoor_outdoor": None, "crowd": None, "budget": None, "style": None,
    }
    fallback = {
        "original_query":  user_input,
        "rewritten_query": user_input,
        "query_type":      "SEMANTIC",
        "filters":         _empty_filters,
        "visual_intent":   False,
        "sub_queries":     [],
        "keywords":        [],
    }

    try:
        api_key = st.secrets.get("OPENAI_API_KEY", "")
        if not api_key:
            return fallback

        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.3,
            openai_api_key=api_key,
        )
        chain = _rewrite_prompt | llm | StrOutputParser()
        raw = chain.invoke({"user_query": user_input}).strip()

        if raw.startswith("```"):
            raw = re.sub(r"```[a-z]*\n?", "", raw).strip("` \n")

        parsed = json.loads(raw)

        filters = parsed.get("filters", {})
        if not isinstance(filters, dict):
            filters = {}
        # 각 필드 타입 보정
        filters = {
            "category":       [str(c) for c in filters.get("category", [])],
            "purpose":        [str(p) for p in filters.get("purpose", [])],
            "weather":        filters.get("weather") or None,
            "indoor_outdoor": filters.get("indoor_outdoor") or None,
            "crowd":          filters.get("crowd") or None,
            "budget":         filters.get("budget") or None,
            "style":          filters.get("style") or None,
        }

        valid_types = {"KEYWORD", "SEMANTIC", "MIXED", "COMPLEX"}
        query_type = parsed.get("query_type", "SEMANTIC")
        if query_type not in valid_types:
            query_type = "SEMANTIC"

        sub_queries = [str(q) for q in parsed.get("sub_queries", [])][:3]
        if query_type == "COMPLEX" and not sub_queries:
            sub_queries = [user_input]

        result = {
            "original_query":  str(parsed.get("original_query", user_input)),
            "rewritten_query": str(parsed.get("rewritten_query", user_input)),
            "query_type":      query_type,
            "filters":         filters,
            "visual_intent":   bool(parsed.get("visual_intent", False)),
            "sub_queries":     sub_queries,
            "keywords":        filters["category"],  # BM25 하위 호환
        }

        print(
            f"[query rewrite]\n"
            f"  input:        {result['original_query']}\n"
            f"  rewritten:    {result['rewritten_query']}\n"
            f"  type:         {result['query_type']}\n"
            f"  visual_intent:{result['visual_intent']}\n"
            f"  filters:      {result['filters']}\n"
            f"  sub_queries:  {result['sub_queries']}"
        )

        _REWRITE_CACHE[cache_key] = result
        return result

    except Exception as exc:
        print(f"[query rewrite] 실패: {exc} — 원본 쿼리 유지")
        return fallback


# ── Query Optimization ───────────────────────────────────────────────────────
# 복합 연결 패턴: "도 하고", "하고 싶어", "도 가고", "그리고" 등
_COMPLEX_PATTERN = re.compile(
    r"도\s*하고|하고\s*싶|도\s*가고|도\s*먹고|그리고|뿐만\s*아니라|면서|며\s"
)
# 장소 카테고리 키워드 집합 (KEYWORD/MIXED 판별용)
_PLACE_CATEGORIES = {
    "카페", "음식점", "맛집", "공원", "박물관", "갤러리", "시장",
    "쇼핑", "체험", "산책", "공연", "전시", "식당", "카페", "베이커리",
}

# 쿼리 타입별 전략 테이블
_STRATEGY_MAP: dict[str, dict] = {
    "KEYWORD":  {
        "use_multi_step": False,
        "faiss_weight":   0.3,
        "bm25_weight":    0.7,
        "top_k":          15,
        "diversity":      0.2,
    },
    "SEMANTIC": {
        "use_multi_step": False,
        "faiss_weight":   0.7,
        "bm25_weight":    0.3,
        "top_k":          15,
        "diversity":      0.3,
    },
    "MIXED": {
        "use_multi_step": False,
        "faiss_weight":   0.5,
        "bm25_weight":    0.5,
        "top_k":          20,
        "diversity":      0.3,
    },
    "COMPLEX": {
        "use_multi_step": True,
        "faiss_weight":   0.5,
        "bm25_weight":    0.5,
        "top_k":          30,
        "diversity":      0.5,
    },
}


def _classify_query_type(rewritten_query: str, keywords: list[str], original_query: str) -> str:
    """
    쿼리를 KEYWORD / SEMANTIC / MIXED / COMPLEX 4종으로 분류한다.

    분류 우선순위:
      1. COMPLEX  — 원본 쿼리에 복합 연결 패턴 존재 (도 하고, 그리고, 면서 …)
      2. KEYWORD  — 재작성 쿼리가 ≤3 토큰이고 키워드가 ≤2개 (단순 명사형)
      3. SEMANTIC — 추출 키워드가 없거나 카테고리 교집합이 없음 (의미 중심 자연어)
      4. MIXED    — 나머지 (키워드 + 의미 혼합)
    """
    # 1. COMPLEX
    if _COMPLEX_PATTERN.search(original_query):
        return "COMPLEX"

    tokens = rewritten_query.split()
    kw_count = len(keywords)

    # 2. KEYWORD
    if len(tokens) <= 3 and kw_count <= 2:
        return "KEYWORD"

    # 3. SEMANTIC
    kw_set = {k.strip() for k in keywords}
    if not (kw_set & _PLACE_CATEGORIES):
        return "SEMANTIC"

    # 4. MIXED (default)
    return "MIXED"


def optimize_query(rewrite_result: dict) -> dict:
    """
    Query Rewrite 결과를 분석하여 최적 검색 전략을 결정한다.

    입력: rewrite_query()의 반환값
      {original_query, rewritten_query, query_type, filters, sub_queries, keywords}

    반환:
      {
        "query_type": "KEYWORD" | "SEMANTIC" | "MIXED" | "COMPLEX",
        "strategy": {
          "use_multi_step": bool,
          "faiss_weight":   float,
          "bm25_weight":    float,
          "top_k":          int,
          "diversity":      float,
        }
      }
    """
    original  = rewrite_result.get("original_query", "")
    rewritten = rewrite_result.get("rewritten_query", original)

    # LLM이 분류한 query_type 우선 사용, 없거나 유효하지 않으면 규칙 기반 fallback
    llm_type = rewrite_result.get("query_type", "")
    valid_types = {"KEYWORD", "SEMANTIC", "MIXED", "COMPLEX"}
    if llm_type in valid_types:
        query_type = llm_type
    else:
        keywords   = rewrite_result.get("keywords", [])
        query_type = _classify_query_type(rewritten, keywords, original)

    strategy = dict(_STRATEGY_MAP[query_type])  # shallow copy — 원본 보호

    print(
        f"[query optimization]\n"
        f"  query:    {original}\n"
        f"  type:     {query_type}\n"
        f"  strategy: {strategy}"
    )

    return {"query_type": query_type, "strategy": strategy}


# ── Multi-step Retrieval ──────────────────────────────────────────────────────
_DECOMPOSE_SYSTEM_MSG = SystemMessage(content=
    "너는 여행 추천 시스템의 쿼리 분해 전문가다.\n\n"
    "사용자의 복합 요청을 분석하여 검색에 적합한 sub-query를 생성하라.\n\n"
    "[규칙]\n"
    "1. 각 sub-query는 완전한 의미를 가진 자연어 문장이어야 한다\n"
    "2. '목적' (데이트, 혼자, 가족 등)을 모든 sub-query에 유지해야 한다\n"
    "3. '분위기/조건' (조용한, 감성적인, 여유로운 등)을 반영해야 한다\n"
    "4. 단순 키워드 (카페, 공원)만 출력하지 말 것\n"
    "5. 2~4개의 sub-query만 생성할 것\n"
    "6. 공통 목적(purpose), 대표 분위기(mood), 카테고리 목록(categories)도 추출할 것\n"
    "7. JSON 외 텍스트 금지. 마크다운 코드블록 금지.\n\n"
    "출력 형식:\n"
    '{"sub_queries": ["자연어 쿼리1", "자연어 쿼리2"],'
    ' "purpose": "데이트|혼자|가족|친구|활동|힐링",'
    ' "mood": "조용한|감성|활기찬|여유|힐링",'
    ' "categories": ["카테고리1", "카테고리2"]}\n\n'
    "입출력 예시:\n"
    '입력: "데이트하면서 산책하고 카페 가고 싶어"\n'
    '출력: {"sub_queries": ["데이트하기 좋은 조용한 산책 공원", "데이트하기 좋은 분위기 좋은 카페"],'
    ' "purpose": "데이트", "mood": "조용한", "categories": ["공원", "카페"]}\n\n'
    '입력: "혼자 여유롭게 쉬면서 커피 마시고 싶어"\n'
    '출력: {"sub_queries": ["혼자 조용히 쉬기 좋은 공간", "혼자 가기 좋은 여유로운 카페"],'
    ' "purpose": "혼자", "mood": "여유", "categories": ["공원", "카페"]}\n\n'
    '입력: "가족과 함께 맛있는 거 먹고 공원도 가고 싶어"\n'
    '출력: {"sub_queries": ["가족과 함께 가기 좋은 분위기 좋은 식당", "가족이 즐기기 좋은 넓은 공원 산책"],'
    ' "purpose": "가족", "mood": "편안한", "categories": ["식당", "공원"]}'
)

_decompose_prompt = ChatPromptTemplate.from_messages([
    _DECOMPOSE_SYSTEM_MSG,
    ("human", '사용자 쿼리:\n"{user_query}"'),
])

_DECOMPOSE_CACHE: dict[str, dict] = {}


def _decompose_cache_key(query: str) -> str:
    return hashlib.md5(query.encode()).hexdigest()


# ── sub-query 품질 필터 / dedup / metadata 추출 ───────────────────────────────

# 순수 카테고리 단어 단독 출력 방지
_PURE_CATEGORY_WORDS: frozenset[str] = frozenset({
    "카페", "공원", "맛집", "식당", "헬스", "볼링", "공연장", "커피",
    "산책", "전시", "공연", "운동", "쇼핑", "박물관", "갤러리",
})

# 규칙 기반 fallback: 카테고리 키워드 → 의도 보존 자연어 쿼리
_FALLBACK_TEMPLATES: dict[str, str] = {
    "카페":   "방문하기 좋은 여유로운 카페",
    "커피":   "커피 마시며 쉬기 좋은 카페",
    "공원":   "산책하기 좋은 자연 공원",
    "산책":   "산책하며 힐링하기 좋은 공원",
    "맛집":   "분위기 좋은 음식점 맛집",
    "식당":   "분위기 좋은 식사 공간",
    "헬스":   "운동하기 좋은 헬스 피트니스 시설",
    "볼링":   "즐기기 좋은 볼링장",
    "공연장": "감상하기 좋은 공연 문화시설",
    "전시":   "전시 관람하기 좋은 문화시설",
}

# 카테고리 힌트 토큰 → category 이름 (per-sub-query 키워드 추출용)
_SQ_CATEGORY_HINTS: dict[str, str] = {
    "카페": "카페", "커피": "카페", "디저트": "카페",
    "공원": "공원", "산책": "공원", "자연": "공원",
    "맛집": "식당", "식당": "식당", "음식": "식당", "식사": "식당",
    "헬스": "헬스", "운동": "헬스", "피트니스": "헬스",
    "볼링": "볼링", "공연": "공연장", "전시": "공연장", "문화": "공연장",
}


def _is_quality_subquery(q: str) -> bool:
    """토큰 수 < 5이거나 순수 카테고리 단어 단독 출력이면 False."""
    stripped = q.strip()
    if stripped in _PURE_CATEGORY_WORDS:
        return False
    return len(stripped.split()) >= 5


def _dedup_subqueries(queries: list[str]) -> list[str]:
    """
    핵심 토큰 자카드 유사도 ≥ 0.6이면 중복으로 판단하여 뒤쪽 제거.
    의미가 다른 쿼리("조용한 카페" vs "분위기 좋은 카페")는 둘 다 유지.
    """
    result: list[str] = []
    for q in queries:
        q_tok = set(q.split())
        is_dup = any(
            len(q_tok & set(e.split())) / max(min(len(q_tok), len(set(e.split()))), 1) >= 0.6
            for e in result
        )
        if not is_dup:
            result.append(q)
    return result


def _extract_sq_category(q: str) -> list[str]:
    """sub-query 텍스트에서 BM25 부스팅용 카테고리 키워드를 추출한다."""
    seen: set[str] = set()
    cats: list[str] = []
    for token, cat in _SQ_CATEGORY_HINTS.items():
        if token in q and cat not in seen:
            seen.add(cat)
            cats.append(cat)
    return cats


def _fallback_decompose(query: str) -> list[str]:
    """
    LLM 실패 시 규칙 기반 분해.
    쿼리에서 카테고리 키워드를 찾아 의도 보존 자연어 쿼리로 변환한다.
    """
    found: list[str] = []
    for kw, template in _FALLBACK_TEMPLATES.items():
        if kw in query and template not in found:
            found.append(template)
    return found[:3] if found else [query]


def decompose_query(query: str) -> dict:
    """
    복잡한 쿼리를 의미 보존형 하위 쿼리로 분해한다.

    반환:
      {
        "original_query": 원본 쿼리,
        "sub_queries":    의도 보존 자연어 쿼리 리스트 (품질 필터·dedup 적용),
        "metadata":       {"purpose": str, "mood": str, "categories": list[str]},
      }
    LLM 실패 시 규칙 기반 fallback.
    동일 입력은 모듈 레벨 dict로 캐싱.
    """
    cache_key = _decompose_cache_key(query)
    if cache_key in _DECOMPOSE_CACHE:
        return _DECOMPOSE_CACHE[cache_key]

    _empty_meta = {"purpose": "", "mood": "", "categories": []}

    def _build_result(sqs: list[str], meta: dict) -> dict:
        return {"original_query": query, "sub_queries": sqs, "metadata": meta}

    try:
        api_key = st.secrets.get("OPENAI_API_KEY", "")
        if not api_key:
            return _build_result(_fallback_decompose(query), _empty_meta)

        llm = ChatOpenAI(model="gpt-4o", temperature=0.0, openai_api_key=api_key)
        chain = _decompose_prompt | llm | StrOutputParser()
        raw = chain.invoke({"user_query": query}).strip()

        if raw.startswith("```"):
            raw = re.sub(r"```[a-z]*\n?", "", raw).strip("` \n")

        parsed = json.loads(raw)

        # ── sub-query 추출 및 품질 필터 ──────────────────────────────────────
        raw_sqs = [str(q) for q in parsed.get("sub_queries", [])][:4]
        filtered = [q for q in raw_sqs if _is_quality_subquery(q)]

        # 품질 필터 후 빈 리스트면 규칙 기반 fallback
        if not filtered:
            filtered = _fallback_decompose(query)

        # ── dedup ─────────────────────────────────────────────────────────────
        deduped = _dedup_subqueries(filtered)
        if not deduped:
            deduped = [query]

        # ── metadata 추출 ─────────────────────────────────────────────────────
        meta = {
            "purpose":    str(parsed.get("purpose", "")),
            "mood":       str(parsed.get("mood", "")),
            "categories": [str(c) for c in parsed.get("categories", [])],
        }

        # ── 로그 ─────────────────────────────────────────────────────────────
        sq_lines = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(deduped))
        print(
            f"[Multi-step log]\n"
            f"원본 쿼리:\n  \"{query}\"\n"
            f"생성된 sub-query:\n{sq_lines}\n"
            f"추출된 metadata:\n"
            f"  - purpose: {meta['purpose']}\n"
            f"  - mood:    {meta['mood']}\n"
            f"  - categories: {meta['categories']}"
        )

        result = _build_result(deduped, meta)
        _DECOMPOSE_CACHE[cache_key] = result
        return result

    except Exception as exc:
        print(f"[Multi-step] 분해 실패: {exc} — 규칙 기반 fallback 적용")
        return _build_result(_fallback_decompose(query), _empty_meta)


def multi_step_retrieve(
    vectorstore,
    sub_queries: list[str],
    region: str,
    personality: str = "",
    weather: str = "",
    budget: str = "",
    crowd: str = "",
    style: str = "",
    llm=None,
    keywords: list[str] | None = None,
    top_k: int = 15,
    sub_query_metas: list[dict] | None = None,
) -> dict:
    """
    sub_queries 각각에 대해 retrieve_places()를 호출하고 결과를 병합한다.

    sub_query_metas: decompose_query()가 반환한 per-sub-query 메타데이터 리스트.
      각 항목에 "categories" 키가 있으면 해당 sub-query의 BM25 keywords로 사용.
      없으면 _extract_sq_category(sq)로 규칙 기반 추출.
      전역 keywords가 명시된 경우 그것을 우선한다.

    병합 전략:
    - main_places: id 기준 중복 제거, 먼저 나온 순서(max-score first-seen) 유지
    - candidate_pool: 동일하게 중복 제거 후 병합
    """
    seen_ids: set[int] = set()
    merged_main: list[dict] = []
    merged_pool: list[dict] = []

    for i, sq in enumerate(sub_queries):
        # per-sub-query keywords: 전역 keywords 우선, 없으면 메타 or 규칙 추출
        if keywords:
            sq_keywords = keywords
        elif sub_query_metas and i < len(sub_query_metas):
            sq_keywords = sub_query_metas[i].get("categories") or _extract_sq_category(sq)
        else:
            sq_keywords = _extract_sq_category(sq)

        result = retrieve_places(
            vectorstore, sq, region, top_k=top_k,
            personality=personality,
            weather=weather, budget=budget, crowd=crowd,
            style=style, llm=llm,
            keywords=sq_keywords,
        )
        main = result.get("main_places", [])
        pool = result.get("candidate_pool", [])

        print(f"[results per query] '{sq}' → main={len(main)}, pool={len(pool)}")

        for place in main:
            pid = place.get("id")
            if pid not in seen_ids:
                seen_ids.add(pid)
                merged_main.append(place)

        for place in pool:
            pid = place.get("id")
            if pid not in seen_ids:
                seen_ids.add(pid)
                merged_pool.append(place)

    return {"main_places": merged_main, "candidate_pool": merged_pool}


def generate_reason_summary(
    personality: str,
    region: str,
    weather: str,
    budget: str,
    crowd: str,
    travel_style: float,
    retrieved_places: list[dict],
) -> str:
    """
    검색된 장소와 사용자 조건을 바탕으로 추천 근거를 LLM으로 생성한다.
    streaming=False: 짧은 출력이므로 단순 invoke 사용.
    """
    top5 = retrieved_places[:5]
    top_places_text = "\n".join(
        f"- {p['장소명']} ({p['카테고리']}) | 혼잡:{p.get('crowd_level', '-')} "
        f"| 실내외:{p.get('indoor_outdoor', '-')} | 가격:{p.get('price_level', '-')}"
        for p in top5
    )
    style_label = _style_label(travel_style)

    chain = reason_summary_prompt | get_llm(streaming=False) | StrOutputParser()
    return chain.invoke({
        "personality":  personality,
        "region":       region,
        "weather":      weather,
        "budget":       budget,
        "crowd":        crowd,
        "travel_style": style_label,
        "top_places":   top_places_text,
    }).strip()


def generate_tldr_summary(
    itinerary: str,
    personality: str,
    travel_style: float,
) -> str:
    """일정 텍스트를 기반으로 4줄 TL;DR 요약을 LLM으로 생성한다."""
    style_label = _style_label(travel_style)
    chain = tldr_prompt | get_llm(streaming=False) | StrOutputParser()
    return chain.invoke({
        "personality":   personality,
        "travel_style":  style_label,
        "itinerary":     itinerary,
    }).strip()


def calculate_confidence(
    itinerary: str,
    retrieved_places: list[dict],
    personality: str,
    region: str,
    validation: dict | None = None,
) -> dict:
    """
    4가지 기준으로 추천 신뢰도를 계산한다.

    반환:
      score       : 0.0~1.0 가중 합산 점수
      level       : "높음" / "보통" / "낮음"
      level_emoji : 🟢 / 🟡 / 🔴
      details     : 각 세부 점수 및 표시용 텍스트
    """
    from rag.validator import extract_place_names

    extracted = extract_place_names(itinerary)
    total = len(extracted)

    # ── 1. RAG 커버리지 ────────────────────────────────────────────────────
    retrieved_names = {p["장소명"] for p in retrieved_places}

    def _matched(name: str) -> bool:
        if name in retrieved_names:
            return True
        return any(name in r or r in name for r in retrieved_names)

    if total > 0:
        matched_count = sum(1 for n in extracted if _matched(n))
        score_coverage = matched_count / total
    else:
        matched_count = 0
        score_coverage = 0.0

    # ── 2. 지역 일치도 ────────────────────────────────────────────────────
    region_scores: list[float] = []
    for p in retrieved_places:
        sigungu = p.get("sigungu", "")
        reg     = p.get("region", "")
        address = p.get("주소", "")
        if region and (sigungu == region or reg == region):
            region_scores.append(1.0)
        elif region and (region in reg or region in address):
            region_scores.append(0.7)
        else:
            region_scores.append(0.4)
    score_region = sum(region_scores) / len(region_scores) if region_scores else 0.5

    # ── 3. 성향 일치도 ────────────────────────────────────────────────────
    if retrieved_places:
        matched_persona = sum(
            1 for p in retrieved_places if personality in p.get("personality_tags", [])
        )
        score_personality = matched_persona / len(retrieved_places)
    else:
        matched_persona = 0
        score_personality = 0.0

    # ── 4. 데이터 충분성 ─────────────────────────────────────────────────
    n = len(retrieved_places)
    if n >= 15:
        score_data = 1.0
    elif n >= 10:
        score_data = 0.8
    elif n >= 5:
        score_data = 0.6
    else:
        score_data = 0.4

    # ── 할루시네이션 보너스 ───────────────────────────────────────────────
    hallucination_bonus = 0.0
    if validation and validation.get("is_clean") and total > 0:
        hallucination_bonus = 0.03

    # ── 최종 점수 ─────────────────────────────────────────────────────────
    raw_score = (
        0.35 * score_coverage
        + 0.25 * score_region
        + 0.25 * score_personality
        + 0.15 * score_data
        + hallucination_bonus
    )
    score = min(raw_score, 1.0)

    if score >= 0.85:
        level, level_emoji = "높음", "🟢"
    elif score >= 0.65:
        level, level_emoji = "보통", "🟡"
    else:
        level, level_emoji = "낮음", "🔴"

    return {
        "score":         score,
        "level":         level,
        "level_emoji":   level_emoji,
        "details": {
            "coverage":         round(score_coverage * 100),
            "matched_count":    matched_count,
            "total_places":     total,
            "region_score":     round(score_region * 100),
            "region":           region,
            "personality_pct":  round(score_personality * 100),
            "data_count":       n,
            "hallucination_ok": bool(validation and validation.get("is_clean")),
        },
    }


def _category_of(name: str, places: list[dict]) -> str:
    """장소 목록에서 이름으로 카테고리를 조회한다. 정확 일치 우선, 부분 일치 차순."""
    for p in places:
        if p["장소명"] == name:
            return p.get("카테고리", "")
    for p in places:
        if name in p["장소명"] or p["장소명"] in name:
            return p.get("카테고리", "")
    return ""


def _replace_place(old_name: str) -> bool:
    """
    일정 텍스트에서 old_name을 후보 장소로 교체한다.

    - candidate_pool + retrieved_places 합산에서 현재 일정에 없는 장소를 선별
    - 같은 카테고리 → 성향 일치 순으로 정렬하여 최선 후보를 선택
    - st.session_state.itinerary 를 직접 수정하고 True 반환
    - 교체 불가 시 False 반환
    """
    itinerary = st.session_state.itinerary
    retrieved = st.session_state.get("retrieved_places", [])
    pool = st.session_state.get("candidate_pool", [])
    personality = st.session_state.get("personality", "")

    all_data = retrieved + pool

    # 현재 일정에 등장하는 장소명 집합
    used_names: set[str] = set()
    for d in _parse_itinerary(itinerary):
        for p in d["places"]:
            used_names.add(p["name"])

    category = _category_of(old_name, all_data)

    # 이미 일정에 있는 장소 제외 (장소명 완전 일치)
    candidates = [
        p for p in all_data
        if p["장소명"] not in used_names
    ]

    if not candidates:
        return False

    # 점수: 같은 카테고리 +2, 성향 일치 +1
    def _score(p: dict) -> int:
        s = 0
        if category and p.get("카테고리", "") == category:
            s += 2
        if personality and personality in p.get("personality_tags", []):
            s += 1
        return s

    candidates.sort(key=_score, reverse=True)
    new_place = candidates[0]
    new_name = new_place["장소명"]

    # 일정 텍스트에서 해당 타임슬롯 줄의 장소명만 정밀 교체
    pattern = re.compile(
        r"^(\d{2}:\d{2}\s*-\s*\d{2}:\d{2}\s+)" + re.escape(old_name) + r"(\s*)$",
        re.MULTILINE,
    )
    new_itinerary, count = pattern.subn(r"\g<1>" + new_name + r"\g<2>", itinerary)

    if count == 0:
        return False

    st.session_state.itinerary = new_itinerary
    st.session_state["_replace_flash"] = (
        f"✅ '{old_name}' → '{new_name}' 으로 교체되었습니다."
    )
    return True


def generate_itinerary(
    personality: str,
    region: str,
    days: int,
    retrieved_places: list[dict] | None = None,
    travel_style: float = 0.5,
    weather: str = "자동",
    budget: str = "상관없음",
    crowd: str = "상관없음",
    candidate_pool: list[dict] | None = None,
) -> str:
    if retrieved_places:
        # RAG 경로: 검색된 장소를 컨텍스트로 주입
        prompt = itinerary_prompt_rag
        inputs = {
            "personality":       personality,
            "days":              days,
            "weather":           weather,
            "travel_style":      travel_style,
            "budget":            budget,
            "crowd":             crowd,
            "place_context":     format_place_context(retrieved_places),
            "candidate_context": format_place_context(candidate_pool) if candidate_pool else "",
        }
    else:
        # Fallback 경로: 기존 LLM-only 동작 유지
        prompt = itinerary_prompt
        inputs = {"personality": personality, "region": region, "days": days}

    chain = prompt | get_llm(streaming=True) | StrOutputParser()
    placeholder = st.empty()
    full_text = ""
    for chunk in chain.stream(inputs):
        full_text += chunk
        placeholder.code(full_text, language=None)
    return full_text


# ── 추천 파이프라인 ────────────────────────────────────────────────────────────
def run_recommendation_pipeline(
    personality: str,
    region: str,
    days: int,
    weather: str,
    budget: str,
    travel_style: float,
    crowd: str,
) -> None:
    """RAG 검색 → 일정 생성 → 검증의 전체 파이프라인을 단계별 로딩 UX와 함께 실행한다."""
    import time

    _status = st.empty()
    _bar    = st.progress(0)

    # ── STEP 1: 장소 검색 ────────────────────────────────────────────────────
    _status.info("🔍 **여행 장소 검색 중...** RAG 데이터베이스에서 맞춤 장소를 찾고 있습니다.")
    _bar.progress(10)

    api_key      = st.secrets.get("OPENAI_API_KEY", "")
    vectorstore  = build_vectorstore(api_key)
    _bar.progress(20)

    query = (
        f"{personality} "
        f"{get_personality_keywords(personality)} "
        f"{region}"
    )
    style_label = _style_label(travel_style)

    # ✅ Query Rewrite — retrieve_places 호출 직전
    _status.info("✏️ **쿼리 최적화 중...** 검색 성능 향상을 위해 쿼리를 재작성하고 있습니다.")
    _bar.progress(27)
    rewrite      = rewrite_query(query, personality=personality, region=region)
    search_query = rewrite["rewritten_query"]
    filters      = rewrite.get("filters", {})

    # filters에서 UI 미입력 필드 보완 (사용자가 입력하지 않은 경우에만)
    if not weather and filters.get("weather"):
        weather = filters["weather"]
    if not budget and filters.get("budget"):
        budget = filters["budget"]
    if not crowd and filters.get("crowd"):
        crowd = filters["crowd"]

    # ✅ Query Optimization — LLM 분류 query_type 기반 검색 전략 결정
    opt      = optimize_query(rewrite)
    strategy = opt["strategy"]

    retrieve_kwargs = dict(
        region=region,
        top_k=strategy["top_k"],
        personality=personality,
        weather=weather, budget=budget, crowd=crowd,
        style=style_label,
        llm=get_llm(streaming=False),
        keywords=rewrite["keywords"],
        faiss_weight=strategy["faiss_weight"],
        bm25_weight=strategy["bm25_weight"],
        diversity=strategy["diversity"],
        query_type=rewrite["query_type"],
        indoor_outdoor=filters.get("indoor_outdoor") or "",
        visual_intent=rewrite.get("visual_intent", False),
        filter_purposes=filters.get("purpose", []),
    )

    # ✅ Multi-step Retrieval — COMPLEX형일 때만 분해 실행
    _status.info("🔍 **Hybrid 검색 중...** FAISS + BM25로 후보 장소를 추출하고 있습니다.")
    if strategy["use_multi_step"]:
        # rewrite_query()가 이미 분해한 sub_queries 재사용 (LLM 재호출 없음)
        sub_queries = rewrite.get("sub_queries") or [search_query]
        if len(sub_queries) > 1:
            rag_result = multi_step_retrieve(vectorstore, sub_queries, **retrieve_kwargs)
        else:
            rag_result = retrieve_places(vectorstore, sub_queries[0], **retrieve_kwargs)
    else:
        rag_result = retrieve_places(vectorstore, search_query, **retrieve_kwargs)
    _bar.progress(33)
    _status.info("🎯 **의미 기반 재정렬 중...** 사용자 의도에 맞게 장소 순위를 조정하고 있습니다.")
    _bar.progress(40)
    retrieved      = rag_result["main_places"]
    candidate_pool = rag_result["candidate_pool"]
    st.session_state.retrieved_places = retrieved
    st.session_state.candidate_pool   = candidate_pool

    # ── STEP 2: 일정 생성 ────────────────────────────────────────────────────
    place_count = len(retrieved)
    _status.info(
        f"🗺️ **일정 생성 중...** {place_count}개 장소를 바탕으로 "
        f"{days}일 일정을 구성하고 있습니다."
    )
    _bar.progress(45)

    st.session_state.itinerary = generate_itinerary(
        personality, region, days,
        retrieved if retrieved else None,
        travel_style=travel_style,
        weather=weather, budget=budget, crowd=crowd,
        candidate_pool=candidate_pool if candidate_pool else None,
    )
    _bar.progress(80)

    # ── STEP 3: 검증 & 신뢰도 계산 ──────────────────────────────────────────
    if retrieved:
        _status.info("🔎 **추천 결과 검증 중...** 장소 신뢰도를 계산하고 있습니다.")

        result = validate_itinerary(st.session_state.itinerary, retrieved)
        st.session_state["validation"] = result
        _bar.progress(85)

        st.session_state["reason_summary"] = generate_reason_summary(
            personality=personality, region=region,
            weather=weather, budget=budget, crowd=crowd,
            travel_style=travel_style, retrieved_places=retrieved,
        )
        _bar.progress(91)

        st.session_state["confidence_result"] = calculate_confidence(
            itinerary=st.session_state.itinerary,
            retrieved_places=retrieved,
            personality=personality, region=region,
            validation=st.session_state.get("validation"),
        )
        _bar.progress(96)
    else:
        st.session_state.pop("validation", None)
        st.session_state.pop("reason_summary", None)
        st.session_state["confidence_result"] = None
        st.session_state["tldr_summary"] = ""

    # ── STEP 4: TL;DR 요약 생성 ──────────────────────────────────────────────
    _status.info("📝 **일정 요약 생성 중...** 핵심 내용을 정리하고 있습니다.")
    st.session_state["tldr_summary"] = generate_tldr_summary(
        itinerary=st.session_state.itinerary,
        personality=personality,
        travel_style=travel_style,
    )

    # ── 완료 ─────────────────────────────────────────────────────────────────
    _bar.progress(100)
    _status.success("✅ 여행 일정 생성 완료!")
    time.sleep(0.8)
    _status.empty()
    _bar.empty()


# ── 진행 단계 표시 ────────────────────────────────────────────────────────────
def render_step_indicator() -> None:
    """현재 진행 단계를 상단에 항상 표시한다."""
    stage = st.session_state.get("stage", 1)
    has_itinerary = bool(st.session_state.get("itinerary", ""))

    # session_state → UX 단계 (1~4) 변환
    if has_itinerary:
        ux_step = 4
    elif stage == 4:
        ux_step = 3
    elif stage == 3:
        ux_step = 2
    else:
        ux_step = 1

    steps = [
        ("성향 입력",  "여행 스타일과 취향을 분석합니다"),
        ("조건 선택",  "날씨·예산·혼잡도를 설정합니다"),
        ("추천 생성",  "RAG 기반으로 맞춤 일정을 생성합니다"),
        ("결과 확인",  "생성된 여행 일정을 확인하고 교체합니다"),
    ]

    progress_pct = {1: 0, 2: 33, 3: 66, 4: 100}[ux_step]
    current_label, current_desc = steps[ux_step - 1]

    def _circle(idx: int) -> str:
        """단계별 원형 아이콘 HTML 반환."""
        if idx < ux_step:        # 완료
            bg, fg, inner = "#4a7afe", "white", "✓"
            label_color = "#4a7afe"
            label_weight = "600"
        elif idx == ux_step:     # 현재
            bg, fg, inner = "#4a7afe", "white", str(idx)
            label_color = "#1a1a2e"
            label_weight = "700"
        else:                    # 미진행
            bg, fg, inner = "#e0e6f0", "#aaa", str(idx)
            label_color = "#aaa"
            label_weight = "400"

        label = steps[idx - 1][0]
        return (
            f"<div style='display:flex;flex-direction:column;align-items:center;min-width:72px;'>"
            f"  <div style='width:34px;height:34px;border-radius:50%;"
            f"    background:{bg};color:{fg};display:flex;align-items:center;"
            f"    justify-content:center;font-size:14px;font-weight:700;"
            f"    box-shadow:0 2px 6px {bg}55;'>{inner}</div>"
            f"  <div style='font-size:11px;margin-top:5px;color:{label_color};"
            f"    font-weight:{label_weight};text-align:center;'>{label}</div>"
            f"</div>"
        )

    def _connector(idx: int) -> str:
        """단계 사이 연결선 HTML 반환."""
        filled = idx < ux_step
        color = "#4a7afe" if filled else "#e0e6f0"
        return (
            f"<div style='flex:1;height:2px;background:{color};"
            f"margin-bottom:20px;min-width:20px;'></div>"
        )

    # 원 + 연결선 조합
    row_html = "<div style='display:flex;align-items:center;justify-content:center;'>"
    for i in range(1, 5):
        row_html += _circle(i)
        if i < 4:
            row_html += _connector(i)
    row_html += "</div>"

    # 진행률 바
    bar_html = (
        f"<div style='background:#e0e6f0;border-radius:4px;height:4px;"
        f"margin:10px 0 6px 0;overflow:hidden;'>"
        f"  <div style='width:{progress_pct}%;height:100%;background:#4a7afe;"
        f"    transition:width 0.4s ease;'></div>"
        f"</div>"
    )

    # 현재 단계 설명
    desc_html = (
        f"<div style='text-align:center;font-size:12px;color:#666;margin-top:2px;'>"
        f"  <span style='color:#4a7afe;font-weight:600;'>STEP {ux_step}</span>"
        f"  &nbsp;·&nbsp; {current_desc}"
        f"  &nbsp;<span style='color:#aaa;'>({progress_pct}% 완료)</span>"
        f"</div>"
    )

    st.markdown(
        f"<div style='padding:14px 18px 10px 18px;background:#f8f9ff;"
        f"border:1px solid #e0e6f0;border-radius:12px;margin-bottom:16px;'>"
        f"{row_html}{bar_html}{desc_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── 인증 게이트 ───────────────────────────────────────────────────────────────
if not require_auth():
    show_login_ui()
    st.stop()

# ── 헤더 (로그인 완료 후에만 렌더링) ─────────────────────────────────────────
st.title("AI 여행 플래너")
st.caption("여행 성향을 분석하고 맞춤형 여행 일정을 생성합니다.")

# 로그아웃 버튼 + 저장 목록 (사이드바)
with st.sidebar:
    st.write(f"**{st.session_state.get('username', '')}** 님")
    if st.button("로그아웃", use_container_width=True):
        logout()
        st.rerun()

render_saved_plans()

render_step_indicator()

st.divider()

# ── 1단계: 여행 성향 질문 ─────────────────────────────────────────────────────
if st.session_state.stage == 1:
    st.subheader("1단계  여행 성향 질문")
    st.write("아래 6가지 질문에 자유롭게 답변해 주세요.")

    with st.form("survey_form"):
        for i, q in enumerate(QUESTIONS, start=1):
            st.session_state.answers[f"q{i}"] = st.text_area(
                label=q,
                value=st.session_state.answers.get(f"q{i}", ""),
                height=80,
                key=f"q{i}_input",
            )
        submitted = st.form_submit_button("분석하기", use_container_width=True)

    if submitted:
        missing = [
            f"Q{i}" for i in range(1, 7)
            if not st.session_state.answers.get(f"q{i}", "").strip()
        ]
        if missing:
            st.warning(f"다음 질문에 답변해 주세요: {', '.join(missing)}")
        else:
            st.session_state.stage = 2
            st.rerun()

# ── 2단계: 성향 분석 ──────────────────────────────────────────────────────────
elif st.session_state.stage == 2:
    st.subheader("2단계  여행 성향 분석 중...")

    if st.session_state.personality is None:
        import time as _t
        _s2_status = st.empty()
        _s2_bar    = st.progress(0)
        _s2_status.info("🧠 **성향 분석 중...** 답변을 기반으로 여행 스타일을 분석하고 있습니다.")
        _s2_bar.progress(30)
        st.session_state.personality = analyze_personality(st.session_state.answers)
        _s2_bar.progress(100)
        _s2_status.success("✅ 성향 분석 완료!")
        _t.sleep(0.6)
        _s2_status.empty()
        _s2_bar.empty()

    personality = st.session_state.personality
    desc = PERSONALITY_DESC.get(personality, "나만의 여행 스타일을 가진 타입")

    st.success(f"당신의 여행 성향은 **{personality}** 입니다.")
    st.info(desc)

    if st.button("다음 단계로", use_container_width=True):
        st.session_state.stage = 3
        st.rerun()

# ── 3단계: 여행 정보 입력 ─────────────────────────────────────────────────────
elif st.session_state.stage == 3:
    st.subheader("3단계  여행 정보 입력")
    st.write(f"분석된 성향: **{st.session_state.personality}**")

    with st.form("travel_info_form"):
        region = st.text_input(
            "여행 지역",
            value=st.session_state.region,
            placeholder="예: 제주도, 도쿄, 파리",
        )
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "여행 시작일",
                value=st.session_state.start_date or date.today(),
                min_value=date.today(),
            )
        with col2:
            end_date = st.date_input(
                "여행 종료일",
                value=st.session_state.end_date or (date.today() + timedelta(days=2)),
                min_value=date.today(),
            )

        st.markdown("**여행 조건 (선택)**")
        cond1, cond2 = st.columns(2)
        with cond1:
            weather = st.selectbox(
                "날씨 조건",
                ["자동", "맑음", "비", "더위", "추위"],
                index=["자동", "맑음", "비", "더위", "추위"].index(
                    st.session_state.get("weather", "자동")
                ),
            )
            budget = st.selectbox(
                "예산 수준",
                ["상관없음", "저가", "중가", "고가"],
                index=["상관없음", "저가", "중가", "고가"].index(
                    st.session_state.get("budget", "상관없음")
                ),
            )
        with cond2:
            travel_style = st.slider(
                "일정 스타일  (0: 빠르게 ↔ 1: 여유롭게)",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.get("travel_style", 0.5)),
                step=0.1,
            )
            crowd = st.selectbox(
                "혼잡도 선호",
                ["상관없음", "조용", "활기"],
                index=["상관없음", "조용", "활기"].index(
                    st.session_state.get("crowd", "상관없음")
                ),
            )

        submitted = st.form_submit_button("일정 생성하기", use_container_width=True)

    if submitted:
        if not region.strip():
            st.warning("여행 지역을 입력해 주세요.")
        elif end_date < start_date:
            st.warning("종료일은 시작일 이후여야 합니다.")
        else:
            st.session_state.region = region.strip()
            st.session_state.start_date = start_date
            st.session_state.end_date = end_date
            st.session_state.weather = weather
            st.session_state.travel_style = travel_style
            st.session_state.budget = budget
            st.session_state.crowd = crowd
            st.session_state.stage = 4
            st.rerun()

# ── 4단계: 여행 일정 생성 ─────────────────────────────────────────────────────
# [수정 4] RAG 검색 → 일정 생성 → 검증 결과 표시 로직 통합
elif st.session_state.stage == 4:
    days = (st.session_state.end_date - st.session_state.start_date).days + 1

    st.subheader("4단계  맞춤형 여행 일정")

    # ── 입력 요약 카드 ────────────────────────────────────────────────────────
    _ts = float(st.session_state.get("travel_style", 0.5))
    _style_text = "빠르게" if _ts < 0.4 else ("여유롭게" if _ts > 0.6 else "보통")

    _card_items = [
        ("🧭", "성향",   st.session_state.get("personality", "미정")),
        ("📍", "지역",   st.session_state.get("region", "미입력")),
        ("📅", "기간",   f"{days}일"),
        ("🌤️", "날씨",  st.session_state.get("weather", "자동")),
        ("💰", "예산",   st.session_state.get("budget", "상관없음")),
        ("🚶", "스타일", _style_text),
        ("👥", "혼잡도", st.session_state.get("crowd", "상관없음")),
    ]

    st.markdown(
        """
        <div style="
            background:#f0f4ff;
            border:1px solid #c9d6ff;
            border-radius:12px;
            padding:16px 20px;
            margin-bottom:12px;
        ">
        <p style="margin:0 0 10px 0; font-weight:700; font-size:15px; color:#3a3f5c;">
            ✈️ 여행 요약
        </p>
        """,
        unsafe_allow_html=True,
    )
    _cols = st.columns(len(_card_items))
    for col, (emoji, label, value) in zip(_cols, _card_items):
        with col:
            st.markdown(
                f"<div style='text-align:center;'>"
                f"<div style='font-size:22px;'>{emoji}</div>"
                f"<div style='font-size:11px; color:#888; margin:2px 0;'>{label}</div>"
                f"<div style='font-size:13px; font-weight:600; color:#2c2c2c;'>{value}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)
    # ─────────────────────────────────────────────────────────────────────────

    # ── 조건 수정 & 재추천 영역 ──────────────────────────────────────────────
    with st.expander("🔧 조건 수정 후 재추천", expanded=False):
        rc1, rc2 = st.columns(2)
        with rc1:
            _weather_opts = ["자동", "맑음", "비", "더위", "추위"]
            new_weather = st.selectbox(
                "날씨 조건",
                _weather_opts,
                index=_weather_opts.index(st.session_state.get("weather", "자동")),
                key="re_weather",
            )
            _budget_opts = ["상관없음", "저가", "중가", "고가"]
            new_budget = st.selectbox(
                "예산 수준",
                _budget_opts,
                index=_budget_opts.index(st.session_state.get("budget", "상관없음")),
                key="re_budget",
            )
        with rc2:
            new_style = st.slider(
                "일정 스타일 (0: 빠르게 ↔ 1: 여유롭게)",
                min_value=0.0, max_value=1.0,
                value=float(st.session_state.get("travel_style", 0.5)),
                step=0.1,
                key="re_style",
            )
            _crowd_opts = ["상관없음", "조용", "활기"]
            new_crowd = st.selectbox(
                "혼잡도 선호",
                _crowd_opts,
                index=_crowd_opts.index(st.session_state.get("crowd", "상관없음")),
                key="re_crowd",
            )

        if st.button("🔁 조건 수정 후 재추천", use_container_width=True):
            st.session_state["weather"]       = new_weather
            st.session_state["budget"]        = new_budget
            st.session_state["travel_style"]  = new_style
            st.session_state["crowd"]         = new_crowd
            st.session_state.itinerary        = ""

            run_recommendation_pipeline(
                personality=st.session_state.personality,
                region=st.session_state.region,
                days=days,
                weather=new_weather,
                budget=new_budget,
                travel_style=new_style,
                crowd=new_crowd,
            )
            st.rerun()
    # ─────────────────────────────────────────────────────────────────────────

    st.divider()

    if not st.session_state.itinerary:
        run_recommendation_pipeline(
            personality=st.session_state.personality,
            region=st.session_state.region,
            days=days,
            weather=st.session_state.get("weather", "자동"),
            budget=st.session_state.get("budget", "상관없음"),
            travel_style=float(st.session_state.get("travel_style", 0.5)),
            crowd=st.session_state.get("crowd", "상관없음"),
        )
        st.rerun()
    else:
        # ── 저장된 일정 보기 모드 ─────────────────────────────────────────────
        viewing_id = st.session_state.get("viewing_plan_id")
        if viewing_id:
            matched = next(
                (p for p in st.session_state["saved_plans"] if p["id"] == viewing_id),
                None,
            )
            if matched:
                st.info(
                    f"📂 저장된 일정 보기 — {matched['personality']} · "
                    f"{matched['region']} · {matched['days']}일 "
                    f"({matched['created_at']})"
                )
                render_timeline(matched["itinerary"])
                if st.button("← 현재 일정으로 돌아가기"):
                    st.session_state["viewing_plan_id"] = None
                    st.rerun()
                st.stop()

        # ── TL;DR 요약 ──────────────────────────────────────────────────────
        tldr = st.session_state.get("tldr_summary", "")
        if tldr:
            st.markdown("### 📌 한눈에 보는 여행 요약")
            tldr_lines = [l.strip() for l in tldr.splitlines() if l.strip()]
            if tldr_lines:
                items_html = "".join(
                    f"<p style='margin:6px 0; font-size:14px; color:#1a1a2e;'>{l}</p>"
                    for l in tldr_lines
                )
                st.markdown(
                    "<div style='"
                    "background:#fffdf0; border:1px solid #ffe08a;"
                    "border-left:4px solid #f5a623; border-radius:10px;"
                    "padding:14px 18px; margin-bottom:16px;'>"
                    f"<p style='margin:0 0 8px 0; font-size:13px; font-weight:700;"
                    f"color:#b07d00;'>TL;DR — 핵심만 먼저 확인하세요</p>"
                    + items_html
                    + "</div>",
                    unsafe_allow_html=True,
                )

        # ── 추천 근거 요약 ───────────────────────────────────────────────────
        reason_summary = st.session_state.get("reason_summary", "")
        if reason_summary:
            st.markdown("### 📌 추천 기준 요약")
            lines = [l.strip() for l in reason_summary.splitlines() if l.strip().startswith("-")]
            if lines:
                st.markdown(
                    "<div style='"
                    "background:#f8f9ff; border:1px solid #d0d8ff;"
                    "border-left:4px solid #4a7afe; border-radius:10px;"
                    "padding:14px 18px; margin-bottom:16px;'>"
                    + "".join(
                        f"<p style='margin:4px 0; font-size:14px; color:#2c2c4a;'>{l}</p>"
                        for l in lines
                    )
                    + "</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.info(reason_summary)

        # ── 추천 신뢰도 표시 ──────────────────────────────────────────────────
        cr = st.session_state.get("confidence_result")
        if cr:
            d = cr["details"]
            pct = round(cr["score"] * 100)
            level_color = {"높음": "#1a7f37", "보통": "#9a6700", "낮음": "#cf222e"}
            level_bg    = {"높음": "#dafbe1", "보통": "#fff8c5", "낮음": "#ffebe9"}
            color = level_color.get(cr["level"], "#555")
            bg    = level_bg.get(cr["level"], "#f5f5f5")

            hal_line = (
                "<p style='margin:3px 0; font-size:13px; color:#2c2c4a;'>"
                "✅ 할루시네이션 없음 — 모든 장소가 RAG 검색 결과에서 확인됨</p>"
                if d["hallucination_ok"] else ""
            )

            st.markdown(
                f"<div style='background:{bg}; border:1px solid {color}33;"
                f"border-left:4px solid {color}; border-radius:10px;"
                f"padding:14px 18px; margin-bottom:16px;'>"
                f"<p style='margin:0 0 8px 0; font-size:15px; font-weight:700; color:{color};'>"
                f"🔍 추천 신뢰도: {cr['level']} {cr['level_emoji']} ({pct}%)</p>"
                f"<p style='margin:3px 0; font-size:13px; color:#2c2c4a;'>"
                f"📦 RAG 커버리지: {d['matched_count']}/{d['total_places']}개 일치 "
                f"({d['coverage']}%)</p>"
                f"<p style='margin:3px 0; font-size:13px; color:#2c2c4a;'>"
                f"📍 지역 일치도: {d['region']} ({d['region_score']}%)</p>"
                f"<p style='margin:3px 0; font-size:13px; color:#2c2c4a;'>"
                f"🧭 성향 일치율: {d['personality_pct']}%</p>"
                f"<p style='margin:3px 0; font-size:13px; color:#2c2c4a;'>"
                f"🗂️ 검색 데이터: {d['data_count']}개</p>"
                f"{hal_line}"
                f"</div>",
                unsafe_allow_html=True,
            )

        render_timeline(st.session_state.itinerary, interactive=True)

        # ── 일정 저장 버튼 ───────────────────────────────────────────────────
        already_saved = any(
            p["itinerary"] == st.session_state.itinerary
            for p in st.session_state.get("saved_plans", [])
        )
        st.divider()
        if already_saved:
            st.success("✅ 이미 저장된 일정입니다.")
        else:
            default_title = f"{st.session_state.get('personality', '미정')} · {st.session_state.get('region', '미입력')} {days}일"
            plan_title = st.text_input(
                "📝 일정 제목",
                value=default_title,
                placeholder="저장할 일정의 제목을 입력하세요",
                key="plan_title_input",
            )
            if st.button("💾 일정 저장하기", use_container_width=True):
                save_plan(days, title=plan_title)
                st.success(
                    f"'{plan_title}' 일정이 저장되었습니다! "
                    f"(총 {len(st.session_state['saved_plans'])}개 저장됨)"
                )
                st.rerun()

        # 현재 일정 JSON 내보내기
        _export_title = next(
            (p["title"] for p in st.session_state.get("saved_plans", []) if p["itinerary"] == st.session_state.itinerary),
            st.session_state.get("plan_title_input", f"{st.session_state.get('personality', '미정')} · {st.session_state.get('region', '미입력')} {days}일"),
        )
        current_plan_data = {
            "title": _export_title,
            "personality":  st.session_state.get("personality", ""),
            "region":       st.session_state.get("region", ""),
            "days":         days,
            "weather":      st.session_state.get("weather", "자동"),
            "budget":       st.session_state.get("budget", "상관없음"),
            "travel_style": float(st.session_state.get("travel_style", 0.5)),
            "crowd":        st.session_state.get("crowd", "상관없음"),
            "itinerary":    st.session_state.get("itinerary", ""),
            "tldr_summary": st.session_state.get("tldr_summary", ""),
            "exported_at":  datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        st.download_button(
            label="📤 현재 일정 JSON 내보내기",
            data=json.dumps(current_plan_data, ensure_ascii=False, indent=2),
            file_name=f"travel_plan_{st.session_state.get('region', 'plan')}.json",
            mime="application/json",
            use_container_width=True,
            key="export_current_plan",
        )

    # ── 검증 결과 표시 ───────────────────────────────────────────────────────
    if st.session_state.get("validation"):
        v = st.session_state["validation"]
        st.divider()
        st.caption("Hallucination 검증 결과")
        vcol1, vcol2, vcol3 = st.columns(3)
        vcol1.metric("전체 장소", v["total"])
        vcol2.metric("검증 완료", len(v["verified"]))
        vcol3.metric("미검증 장소", len(v["unverified"]))

        if v["is_clean"]:
            st.success("모든 장소가 RAG 검색 결과 내에서 확인되었습니다.")
        else:
            st.warning(
                "다음 장소는 RAG 데이터에서 확인되지 않았습니다: "
                + ", ".join(v["unverified"])
            )

    # ── 다른 성향으로 추천받기 ────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 🔀 다른 스타일로 추천받기")
    st.caption("현재 조건(지역·날씨·예산·스타일)은 그대로 유지되고, 여행 성향만 변경됩니다.")

    current_personality = st.session_state.get("personality", "")
    other_personalities = [p for p in PERSONALITY_TYPES if p != current_personality]

    alt_cols = st.columns(len(other_personalities))
    for col, alt_p in zip(alt_cols, other_personalities):
        with col:
            if st.button(f"{alt_p}", use_container_width=True, key=f"alt_{alt_p}"):
                prev_personality = current_personality
                st.session_state["personality"] = alt_p
                st.session_state.itinerary = ""
                st.session_state.retrieved_places = []
                st.session_state.candidate_pool = []
                st.session_state.pop("validation", None)

                st.info(f"✨ {prev_personality} → {alt_p}으로 성향 변경 후 재추천 중...")

                with st.spinner("다른 스타일 여행 생성 중..."):
                    api_key = st.secrets.get("OPENAI_API_KEY", "")
                    vectorstore = build_vectorstore(api_key)
                    query = (
                        f"{alt_p} "
                        f"{get_personality_keywords(alt_p)} "
                        f"{st.session_state.region}"
                    )
                    # ✅ Query Rewrite — 다른 성향 추천에도 적용
                    rewrite = rewrite_query(
                        query,
                        personality=alt_p,
                        region=st.session_state.region,
                    )
                    # ✅ Query Optimization — 다른 성향 추천에도 적용
                    alt_opt      = optimize_query(rewrite)
                    alt_strategy = alt_opt["strategy"]

                    alt_retrieve_kwargs = dict(
                        region=st.session_state.region,
                        top_k=alt_strategy["top_k"],
                        personality=alt_p,  # 버튼으로 선택한 성향 유지 (intent 무시)
                        weather=st.session_state.get("weather", "자동"),
                        budget=st.session_state.get("budget", "상관없음"),
                        crowd=st.session_state.get("crowd", "상관없음"),
                        style=_style_label(float(st.session_state.get("travel_style", 0.5))),
                        llm=get_llm(streaming=False),
                        keywords=rewrite["keywords"],
                        faiss_weight=alt_strategy["faiss_weight"],
                        bm25_weight=alt_strategy["bm25_weight"],
                        diversity=alt_strategy["diversity"],
                        query_type=rewrite["query_type"],
                        indoor_outdoor=rewrite.get("filters", {}).get("indoor_outdoor") or "",
                        visual_intent=rewrite.get("visual_intent", False),
                        filter_purposes=rewrite.get("filters", {}).get("purpose", []),
                    )

                    # ✅ Multi-step Retrieval — COMPLEX형일 때만 분해 실행
                    if alt_strategy["use_multi_step"]:
                        # rewrite_query()가 이미 분해한 sub_queries 재사용 (LLM 재호출 없음)
                        alt_sub_queries = rewrite.get("sub_queries") or [rewrite["rewritten_query"]]
                        if len(alt_sub_queries) > 1:
                            rag_result = multi_step_retrieve(vectorstore, alt_sub_queries, **alt_retrieve_kwargs)
                        else:
                            rag_result = retrieve_places(vectorstore, alt_sub_queries[0], **alt_retrieve_kwargs)
                    else:
                        rag_result = retrieve_places(vectorstore, rewrite["rewritten_query"], **alt_retrieve_kwargs)
                    retrieved      = rag_result["main_places"]
                    candidate_pool = rag_result["candidate_pool"]
                    st.session_state.retrieved_places = retrieved
                    st.session_state.candidate_pool   = candidate_pool

                    st.session_state.itinerary = generate_itinerary(
                        alt_p,
                        st.session_state.region,
                        days,
                        retrieved if retrieved else None,
                        travel_style=float(st.session_state.get("travel_style", 0.5)),
                        weather=st.session_state.get("weather", "자동"),
                        budget=st.session_state.get("budget", "상관없음"),
                        crowd=st.session_state.get("crowd", "상관없음"),
                        candidate_pool=candidate_pool if candidate_pool else None,
                    )

                    if retrieved:
                        st.session_state["validation"] = validate_itinerary(
                            st.session_state.itinerary, retrieved
                        )
                        st.session_state["reason_summary"] = generate_reason_summary(
                            personality=alt_p,
                            region=st.session_state.region,
                            weather=st.session_state.get("weather", "자동"),
                            budget=st.session_state.get("budget", "상관없음"),
                            crowd=st.session_state.get("crowd", "상관없음"),
                            travel_style=float(st.session_state.get("travel_style", 0.5)),
                            retrieved_places=retrieved,
                        )
                        st.session_state["confidence_result"] = calculate_confidence(
                            itinerary=st.session_state.itinerary,
                            retrieved_places=retrieved,
                            personality=alt_p,
                            region=st.session_state.region,
                            validation=st.session_state.get("validation"),
                        )
                        st.session_state["tldr_summary"] = generate_tldr_summary(
                            itinerary=st.session_state.itinerary,
                            personality=alt_p,
                            travel_style=float(st.session_state.get("travel_style", 0.5)),
                        )
                    else:
                        st.session_state.pop("validation", None)
                        st.session_state.pop("reason_summary", None)
                        st.session_state["confidence_result"] = None
                        st.session_state["tldr_summary"] = ""
                        st.session_state["tldr_summary"] = ""

                st.rerun()
    # ─────────────────────────────────────────────────────────────────────────

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("일정 다시 생성", use_container_width=True):
            st.session_state.itinerary = ""
            st.session_state.retrieved_places = []
            st.session_state.candidate_pool = []
            st.session_state.pop("validation", None)
            st.session_state.pop("reason_summary", None)
            st.session_state["confidence_result"] = None
            st.session_state["tldr_summary"] = ""
            st.rerun()
    with col_b:
        if st.button("처음부터 다시", use_container_width=True):
            for k, v in defaults.items():
                st.session_state[k] = v
            st.session_state.pop("validation", None)
            st.session_state.pop("reason_summary", None)
            st.session_state["confidence_result"] = None
            st.rerun()
