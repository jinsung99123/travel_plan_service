"""
FastAPI 인증 백엔드
- POST /auth/login    : JWT 발급
- POST /auth/register : 회원가입
- GET  /auth/verify   : 토큰 검증
"""

import os
from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from pydantic import BaseModel

# ── 설정 ─────────────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-before-deploy-use-32chars!!")
ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 60
# 쉼표로 구분된 허용 출처 목록 (예: https://myapp.streamlit.app,http://localhost:8501)
_ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")


def _verify_pw(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ── 데모 유저 DB (운영 시 실제 DB로 교체) ────────────────────────────────────
# 비밀번호 해시 사전 계산 → 모듈 로드 시 bcrypt 연산 없음 (이벤트 루프 블로킹 방지)
# 해시 재생성: python -c "import bcrypt; print(bcrypt.hashpw(b'NEW_PW', bcrypt.gensalt(rounds=10)).decode())"
USERS_DB: dict[str, str] = {
    "admin": "$2b$10$UyM41McVPcmTGDonkcluJesOu1ZCwOEVrGEhtClCiJLTvdyACHCzu",
    "user1": "$2b$10$n10d1t8locXv13SxaqqC3eP/Sf3p8tQwApOLFIYCC1QBe9vOVDLRa",
}

# ── FastAPI 앱 ────────────────────────────────────────────────────────────────
app = FastAPI(title="Travel Planner Auth API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 스키마 ────────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ── 내부 유틸 ─────────────────────────────────────────────────────────────────
def _create_access_token(username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": username, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def _decode_token(token: str) -> str:
    """토큰 검증 후 username 반환. 실패 시 HTTPException."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub", "")
        if not username:
            raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")


# ── 엔드포인트 ────────────────────────────────────────────────────────────────
@app.post("/auth/register", status_code=201)
def register(req: RegisterRequest) -> dict:
    """신규 회원 등록. username 중복 시 400 반환."""
    if len(req.username) < 3 or any(c.isspace() for c in req.username):
        raise HTTPException(status_code=400, detail="아이디는 공백 없이 3자 이상이어야 합니다.")
    if req.username in USERS_DB:
        raise HTTPException(status_code=400, detail="이미 사용 중인 아이디입니다.")
    hashed = bcrypt.hashpw(req.password.encode(), bcrypt.gensalt(rounds=10)).decode()
    USERS_DB[req.username] = hashed
    return {"message": "회원가입이 완료되었습니다."}


@app.post("/auth/login", response_model=TokenResponse)
def login(req: LoginRequest) -> TokenResponse:
    """아이디·비밀번호 검증 후 JWT 발급."""
    hashed = USERS_DB.get(req.username)
    if not hashed or not _verify_pw(req.password, hashed):
        raise HTTPException(status_code=401, detail="아이디 또는 비밀번호가 올바르지 않습니다.")
    return TokenResponse(access_token=_create_access_token(req.username))


@app.get("/auth/verify")
def verify(token: str) -> dict:
    """토큰 유효성 확인 (Streamlit 재시작 시 세션 복구용)."""
    username = _decode_token(token)
    return {"username": username, "valid": True}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
