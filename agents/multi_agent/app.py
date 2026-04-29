"""
agents/multi_agent/app.py

Multi Agent Travel Plan Service — Streamlit UI
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from orchestrator import run_orchestrator  # type: ignore

st.set_page_config(
    page_title="노원 여행 추천",
    page_icon="📍",
    layout="centered",
)

st.caption("📍 현재 노원구 지역 기반 추천 서비스입니다.")
st.title("노원 여행 추천")

query = st.text_input("어떤 여행을 원하시나요?", placeholder="예: 데이트 코스 추천해줘, 조용한 카페 알려줘")

if st.button("추천받기") and query.strip():
    with st.spinner("추천 장소를 찾는 중..."):
        result = run_orchestrator(query.strip(), region="노원")

    if result["metadata"].get("region_mismatch"):
        st.warning("⚠️ 현재는 노원구 데이터 기반으로 추천됩니다.")

    st.markdown("### 추천 결과")
    st.write(result["reply"])

    if result["course"]:
        st.markdown("### 코스 일정")
        for i, step in enumerate(result["course"], 1):
            names = " → ".join(p["name"] for p in step)
            st.markdown(f"**코스 {i}:** {names}")
    elif result["places"]:
        st.markdown("### 추천 장소")
        for p in result["places"]:
            with st.expander(f"{p['name']} ({p['category']})"):
                st.write(p.get("description", ""))
                st.caption(p.get("address", ""))
