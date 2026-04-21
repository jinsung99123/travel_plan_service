import re
import uuid
import streamlit as st
from datetime import date, datetime, timedelta
from langchain_openai import ChatOpenAI
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


def render_timeline(itinerary: str) -> None:
    """파싱된 일정을 시간 흐름 기반 타임라인 UI로 렌더링한다."""
    days = _parse_itinerary(itinerary)

    if not days:
        st.code(itinerary, language=None)
        return

    for day_data in days:
        st.markdown(
            f"<h3 style='margin:24px 0 8px 0; color:#3a3f5c;'>📆 Day {day_data['day']}</h3>",
            unsafe_allow_html=True,
        )

        for place in day_data["places"]:
            emoji  = _time_emoji(place["start_time"])
            label  = _time_label(place["start_time"])
            tr     = place["time_range"]
            name   = place["name"]
            desc   = place["description"]
            alts   = place["alternatives"]
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

            st.markdown(
                f"""
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
                """,
                unsafe_allow_html=True,
            )

        st.divider()


def save_plan(days: int) -> None:
    """현재 session_state의 일정을 saved_plans에 저장한다."""
    plan = {
        "id":           str(uuid.uuid4()),
        "personality":  st.session_state.get("personality", "미정"),
        "region":       st.session_state.get("region", "미입력"),
        "days":         days,
        "weather":      st.session_state.get("weather", "자동"),
        "budget":       st.session_state.get("budget", "상관없음"),
        "travel_style": float(st.session_state.get("travel_style", 0.5)),
        "crowd":        st.session_state.get("crowd", "상관없음"),
        "itinerary":    st.session_state.get("itinerary", ""),
        "created_at":   datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    st.session_state["saved_plans"].insert(0, plan)  # 최신순


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

    for plan in plans:
        with st.sidebar.container():
            st.sidebar.markdown(
                f"**{plan['personality']}** · {plan['region']} · {plan['days']}일  \n"
                f"<span style='font-size:11px;color:#888;'>{plan['created_at']}</span>",
                unsafe_allow_html=True,
            )
            btn_col, del_col = st.sidebar.columns([3, 1])
            with btn_col:
                if st.button("보기", key=f"view_{plan['id']}"):
                    st.session_state["viewing_plan_id"] = plan["id"]
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

# 진행 단계 표시
step_labels = ["성향 질문", "성향 분석", "여행 정보", "일정 생성"]
cols = st.columns(4)
for i, (col, label) in enumerate(zip(cols, step_labels), start=1):
    with col:
        if st.session_state.stage > i:
            st.success(f"{i}. {label}")
        elif st.session_state.stage == i:
            st.info(f"{i}. {label}")
        else:
            st.markdown(
                f"<div style='text-align:center; color:gray;'>{i}. {label}</div>",
                unsafe_allow_html=True,
            )

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
        with st.spinner("답변을 분석하고 있습니다..."):
            st.session_state.personality = analyze_personality(st.session_state.answers)

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
            # 1. session_state 갱신
            st.session_state["weather"]       = new_weather
            st.session_state["budget"]        = new_budget
            st.session_state["travel_style"]  = new_style
            st.session_state["crowd"]         = new_crowd

            with st.spinner("새로운 여행 계획 생성 중..."):
                # 2. RAG 재검색
                api_key = st.secrets.get("OPENAI_API_KEY", "")
                vectorstore = build_vectorstore(api_key)
                query = (
                    f"{st.session_state.personality} "
                    f"{get_personality_keywords(st.session_state.personality)} "
                    f"{st.session_state.region}"
                )
                rag_result = retrieve_places(
                    vectorstore, query, st.session_state.region, top_k=15,
                    personality=st.session_state.personality,
                    weather=new_weather, budget=new_budget, crowd=new_crowd,
                )
                retrieved      = rag_result["main_places"]
                candidate_pool = rag_result["candidate_pool"]
                st.session_state.retrieved_places = retrieved
                st.session_state.candidate_pool   = candidate_pool

                # 3. 일정 재생성
                st.session_state.itinerary = generate_itinerary(
                    st.session_state.personality,
                    st.session_state.region,
                    days,
                    retrieved if retrieved else None,
                    travel_style=new_style,
                    weather=new_weather,
                    budget=new_budget,
                    crowd=new_crowd,
                    candidate_pool=candidate_pool if candidate_pool else None,
                )

                # 4. 검증
                if retrieved:
                    st.session_state["validation"] = validate_itinerary(
                        st.session_state.itinerary, retrieved
                    )
                else:
                    st.session_state.pop("validation", None)

            st.success("새로운 여행 일정이 생성되었습니다!")
            st.rerun()
    # ─────────────────────────────────────────────────────────────────────────

    st.divider()

    if not st.session_state.itinerary:
        # ── RAG: 벡터 스토어 빌드 + 장소 검색 ───────────────────────────────
        api_key = st.secrets.get("OPENAI_API_KEY", "")
        vectorstore = build_vectorstore(api_key)

        query = (
            f"{st.session_state.personality} "
            f"{get_personality_keywords(st.session_state.personality)} "
            f"{st.session_state.region}"
        )
        rag_result = retrieve_places(
            vectorstore,
            query,
            st.session_state.region,
            top_k=15,
            personality=st.session_state.personality,
            weather=st.session_state.get("weather", "자동"),
            budget=st.session_state.get("budget", "상관없음"),
            crowd=st.session_state.get("crowd", "상관없음"),
        )
        retrieved      = rag_result["main_places"]
        candidate_pool = rag_result["candidate_pool"]
        st.session_state.retrieved_places  = retrieved
        st.session_state.candidate_pool    = candidate_pool

        # RAG 검색 결과 요약 표시
        if retrieved:
            with st.expander(f"RAG 검색된 후보 장소 {len(retrieved)}개 보기"):
                for p in retrieved:
                    st.markdown(f"- **{p['장소명']}** ({p['카테고리']}) — {p['주소']}")
            with st.expander(f"전체 후보 풀 {len(candidate_pool)}개 보기 (대안용)"):
                for p in candidate_pool:
                    st.markdown(f"- **{p['장소명']}** ({p['카테고리']}) — {p['주소']}")

        # ── 일정 생성 (RAG 컨텍스트 주입) ───────────────────────────────────
        st.session_state.itinerary = generate_itinerary(
            st.session_state.personality,
            st.session_state.region,
            days,
            retrieved if retrieved else None,
            travel_style=float(st.session_state.get("travel_style", 0.5)),
            weather=st.session_state.get("weather", "자동"),
            budget=st.session_state.get("budget", "상관없음"),
            crowd=st.session_state.get("crowd", "상관없음"),
            candidate_pool=candidate_pool if candidate_pool else None,
        )

        # ── 검증 로직 ────────────────────────────────────────────────────────
        if retrieved:
            result = validate_itinerary(
                st.session_state.itinerary,
                st.session_state.retrieved_places,
            )
            st.session_state["validation"] = result

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

        render_timeline(st.session_state.itinerary)

        # ── 일정 저장 버튼 ───────────────────────────────────────────────────
        already_saved = any(
            p["itinerary"] == st.session_state.itinerary
            for p in st.session_state.get("saved_plans", [])
        )
        if already_saved:
            st.success("✅ 이미 저장된 일정입니다.")
        elif st.button("💾 일정 저장하기", use_container_width=True):
            save_plan(days)
            st.success(
                f"일정이 저장되었습니다! "
                f"(총 {len(st.session_state['saved_plans'])}개 저장됨)"
            )
            st.rerun()

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
                    rag_result = retrieve_places(
                        vectorstore, query, st.session_state.region, top_k=15,
                        personality=alt_p,
                        weather=st.session_state.get("weather", "자동"),
                        budget=st.session_state.get("budget", "상관없음"),
                        crowd=st.session_state.get("crowd", "상관없음"),
                    )
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
                    else:
                        st.session_state.pop("validation", None)

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
            st.rerun()
    with col_b:
        if st.button("처음부터 다시", use_container_width=True):
            for k, v in defaults.items():
                st.session_state[k] = v
            st.session_state.pop("validation", None)
            st.rerun()
