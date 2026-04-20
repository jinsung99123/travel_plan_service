"""
Streamlit 인증 헬퍼 모듈

login()          - POST /auth/login → access_token 저장
logout()         - session_state 인증 정보 초기화
get_auth_headers() - Bearer 토큰 헤더 반환
require_auth()   - 로그인 상태 확인
show_login_ui()  - 로그인 폼 렌더링
"""

import requests
import streamlit as st

# ── 설정 ─────────────────────────────────────────────────────────────────────
# Streamlit Cloud: secrets.toml 또는 시크릿 UI에 AUTH_API_BASE 설정
# 로컬 개발: 기본값 http://localhost:8000 사용
try:
    _API_BASE: str = st.secrets["AUTH_API_BASE"]
except (KeyError, Exception):
    _API_BASE = "http://localhost:8000"
_TIMEOUT = 5  # 초


# ── 인증 상태 키 목록 (로그아웃 시 일괄 제거용) ──────────────────────────────
_AUTH_KEYS = ("access_token", "is_logged_in", "username")


# ── 공개 함수 ─────────────────────────────────────────────────────────────────
def login(username: str, password: str) -> tuple[bool, str]:
    """
    POST /auth/login 호출.
    성공: session_state에 토큰 저장 후 (True, "") 반환
    실패: (False, 에러 메시지) 반환. 비밀번호는 어디에도 저장하지 않음.
    """
    try:
        resp = requests.post(
            f"{_API_BASE}/auth/login",
            json={"username": username, "password": password},
            timeout=_TIMEOUT,
        )
    except requests.exceptions.ConnectionError:
        return False, "인증 서버에 연결할 수 없습니다. 백엔드가 실행 중인지 확인하세요."
    except requests.exceptions.Timeout:
        return False, "요청 시간이 초과되었습니다."

    if resp.status_code == 200:
        data = resp.json()
        st.session_state["access_token"] = data["access_token"]
        st.session_state["is_logged_in"] = True
        st.session_state["username"] = username
        return True, ""

    detail = resp.json().get("detail", "로그인에 실패했습니다.")
    return False, detail


def logout() -> None:
    """session_state에서 인증 정보 전체 제거."""
    for key in _AUTH_KEYS:
        st.session_state.pop(key, None)


def get_auth_headers() -> dict[str, str]:
    """
    API 요청에 포함할 Authorization 헤더 반환.
    토큰이 없으면 빈 dict 반환 → 호출 측에서 토큰 필요 여부 판단.
    """
    token = st.session_state.get("access_token", "")
    if not token:
        return {}
    return {"Authorization": f"Bearer {token}"}


def require_auth() -> bool:
    """로그인 상태면 True, 아니면 False."""
    return bool(st.session_state.get("is_logged_in"))


def register(username: str, password: str) -> tuple[bool, str]:
    """
    POST /auth/register 호출.
    성공: (True, "") 반환
    실패: (False, 에러 메시지) 반환
    """
    try:
        resp = requests.post(
            f"{_API_BASE}/auth/register",
            json={"username": username, "password": password},
            timeout=_TIMEOUT,
        )
    except requests.exceptions.ConnectionError:
        return False, "인증 서버에 연결할 수 없습니다. 백엔드가 실행 중인지 확인하세요."
    except requests.exceptions.Timeout:
        return False, "요청 시간이 초과되었습니다."

    if resp.status_code == 201:
        return True, ""

    detail = resp.json().get("detail", "회원가입에 실패했습니다.")
    return False, detail


def show_login_ui() -> None:
    """
    로그인 / 회원가입 폼 UI 렌더링.
    auth_mode 세션 상태로 화면 전환.
    """
    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = "login"

    st.title("AI 여행 플래너")
    st.caption("로그인 후 여행 일정을 생성할 수 있습니다.")
    st.divider()

    col, _ = st.columns([1.2, 1])
    with col:
        if st.session_state["auth_mode"] == "login":
            st.subheader("로그인")
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("아이디", placeholder="username")
                password = st.text_input("비밀번호", type="password", placeholder="password")
                submitted = st.form_submit_button("로그인", use_container_width=True)

            if submitted:
                if not username.strip() or not password.strip():
                    st.warning("아이디와 비밀번호를 모두 입력하세요.")
                    return
                with st.spinner("인증 중..."):
                    ok, msg = login(username.strip(), password)
                if ok:
                    st.success(f"{username}님, 환영합니다!")
                    st.rerun()
                else:
                    st.error(msg)

            st.caption("계정이 없으신가요?")
            if st.button("회원가입", use_container_width=True):
                st.session_state["auth_mode"] = "register"
                st.rerun()

        else:
            st.subheader("회원가입")
            with st.form("register_form", clear_on_submit=False):
                username = st.text_input("아이디 (공백 없이 3자 이상)", placeholder="username")
                password = st.text_input("비밀번호", type="password", placeholder="password")
                password2 = st.text_input("비밀번호 확인", type="password", placeholder="password")
                submitted = st.form_submit_button("가입하기", use_container_width=True)

            if submitted:
                if not username.strip() or not password.strip():
                    st.warning("아이디와 비밀번호를 모두 입력하세요.")
                    return
                if password != password2:
                    st.error("비밀번호가 일치하지 않습니다.")
                    return
                with st.spinner("가입 처리 중..."):
                    ok, msg = register(username.strip(), password)
                if ok:
                    st.success("회원가입이 완료되었습니다. 로그인해 주세요.")
                    st.session_state["auth_mode"] = "login"
                    st.rerun()
                else:
                    st.error(msg)

            st.caption("이미 계정이 있으신가요?")
            if st.button("로그인으로 돌아가기", use_container_width=True):
                st.session_state["auth_mode"] = "login"
                st.rerun()
