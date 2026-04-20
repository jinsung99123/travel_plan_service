"""
[1단계] CSV 데이터 로드
[2단계] 성향 키워드 매핑

기존 app.py의 PERSONALITY_TYPES / PERSONALITY_DESC 상수와 충돌하지 않도록
CSV 읽기 + 키워드 조회만 담당한다.
"""

import pandas as pd
from pathlib import Path
from functools import lru_cache

DATA_DIR = Path(__file__).parent.parent / "data"


@lru_cache(maxsize=1)
def load_places() -> pd.DataFrame:
    """places.csv 로드. 반복 I/O 방지를 위해 캐시 적용."""
    return pd.read_csv(DATA_DIR / "places.csv", encoding="utf-8")


@lru_cache(maxsize=1)
def load_persona() -> pd.DataFrame:
    """persona.csv 로드. 반복 I/O 방지를 위해 캐시 적용."""
    return pd.read_csv(DATA_DIR / "persona.csv", encoding="utf-8")


def get_personality_keywords(personality: str) -> str:
    """
    [2단계] persona.csv에서 성향에 해당하는 핵심+보조 키워드를 반환한다.
    벡터 검색 쿼리 생성에 사용된다.
    성향이 없으면 성향명 그대로 반환.
    """
    df = load_persona()
    row = df[df["성향"] == personality]
    if row.empty:
        return personality
    core = str(row.iloc[0]["핵심키워드"]).replace(",", " ")
    sub = str(row.iloc[0]["보조키워드"]).replace(",", " ")
    return f"{core} {sub}"
