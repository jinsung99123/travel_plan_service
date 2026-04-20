import streamlit as st
from datetime import date, timedelta
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── [수정 1] RAG 모듈 임포트 (신규 추가) ──────────────────────────────────────
from rag.loader import get_personality_keywords
from rag.retriever import build_vectorstore, retrieve_places, format_place_context
from rag.validator import validate_itinerary

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
    "retrieved_places": [],  # [수정 2] RAG 검색 결과 저장 (신규 추가)
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
        temperature=0.7,
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
ITINERARY_SYSTEM_RAG = (
    ITINERARY_SYSTEM
    + "\n\n[중요] 아래 제공된 장소 목록에서만 선택하여 일정을 구성하라.\n"
    "목록에 없는 장소는 절대 사용 금지.\n"
    "하루에 장소는 3~4개 배치하라."
)

itinerary_prompt = ChatPromptTemplate.from_messages([
    ("system", ITINERARY_SYSTEM),
    ("human", "여행 성향: {personality}\n여행 지역: {region}\n여행 일수: {days}일"),
])

# [수정 3-b] RAG 전용 프롬프트 템플릿 (신규 추가)
itinerary_prompt_rag = ChatPromptTemplate.from_messages([
    ("system", ITINERARY_SYSTEM_RAG),
    ("human", (
        "여행 성향: {personality}\n"
        "여행 지역: {region}\n"
        "여행 일수: {days}일\n\n"
        "[참고 장소 목록]\n{place_context}"
    )),
])


# [수정 3-c] generate_itinerary에 retrieved_places 파라미터 추가
# 기존 시그니처 유지 (retrieved_places=None → 기존 동작 그대로)
def generate_itinerary(
    personality: str,
    region: str,
    days: int,
    retrieved_places: list[dict] | None = None,
) -> str:
    if retrieved_places:
        # RAG 경로: 검색된 장소를 컨텍스트로 주입
        prompt = itinerary_prompt_rag
        inputs = {
            "personality":   personality,
            "region":        region,
            "days":          days,
            "place_context": format_place_context(retrieved_places),
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


# ── 헤더 ──────────────────────────────────────────────────────────────────────
st.title("AI 여행 플래너")
st.caption("여행 성향을 분석하고 맞춤형 여행 일정을 생성합니다.")

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
            st.session_state.stage = 4
            st.rerun()

# ── 4단계: 여행 일정 생성 ─────────────────────────────────────────────────────
# [수정 4] RAG 검색 → 일정 생성 → 검증 결과 표시 로직 통합
elif st.session_state.stage == 4:
    days = (st.session_state.end_date - st.session_state.start_date).days + 1

    st.subheader("4단계  맞춤형 여행 일정")
    col1, col2, col3 = st.columns(3)
    col1.metric("여행 성향", st.session_state.personality)
    col2.metric("여행 지역", st.session_state.region)
    col3.metric("여행 일수", f"{days}일")

    st.divider()

    if not st.session_state.itinerary:
        # ── [추가] RAG: 벡터 스토어 빌드 + 장소 검색 ────────────────────────
        api_key = st.secrets.get("OPENAI_API_KEY", "")
        vectorstore = build_vectorstore(api_key)

        query = (
            f"{st.session_state.personality} "
            f"{get_personality_keywords(st.session_state.personality)} "
            f"{st.session_state.region}"
        )
        retrieved = retrieve_places(
            vectorstore,
            query,
            st.session_state.region,
            top_k=15,
        )
        st.session_state.retrieved_places = retrieved

        # RAG 검색 결과 요약 표시
        if retrieved:
            with st.expander(f"RAG 검색된 후보 장소 {len(retrieved)}개 보기"):
                for p in retrieved:
                    st.markdown(f"- **{p['장소명']}** ({p['카테고리']}) — {p['주소']}")

        # ── 일정 생성 (RAG 컨텍스트 주입) ───────────────────────────────────
        st.session_state.itinerary = generate_itinerary(
            st.session_state.personality,
            st.session_state.region,
            days,
            retrieved_places=retrieved if retrieved else None,
        )

        # ── [추가] 검증 로직 ─────────────────────────────────────────────────
        if retrieved:
            result = validate_itinerary(
                st.session_state.itinerary,
                st.session_state.retrieved_places,
            )
            st.session_state["validation"] = result
    else:
        st.code(st.session_state.itinerary, language=None)

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

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("일정 다시 생성", use_container_width=True):
            st.session_state.itinerary = ""
            st.session_state.retrieved_places = []
            st.session_state.pop("validation", None)
            st.rerun()
    with col_b:
        if st.button("처음부터 다시", use_container_width=True):
            for k, v in defaults.items():
                st.session_state[k] = v
            st.session_state.pop("validation", None)
            st.rerun()
