"""
places_extended.csv + persona.csv → data/places_enriched.json

생성 항목:
  - purpose: 장소별 동적 추론 (카테고리+가격+crowd 기반, 카테고리 고정값 금지)
  - mood:    키워드 + crowd + price 기반 동적 추론 (편안한 금지)
              어휘: 조용한 / 활기찬 / 감성적인 / 힐링되는 / 가성비 / 특별한
  - page_content: FAISS용 자연어 문장 (vocabulary 원형 보존)
  - plain_text:   BM25용 키-값 구조 텍스트 (확장 토큰 포함)
  - metadata:     retriever.py 호환 전체 필드 (personality_tags, is_general, is_visual_spot)
"""

import csv, json, re, sys
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

DATA_DIR = Path(__file__).parent / "data"


# ── 주소 → 지역 전체 분리 ─────────────────────────────────────────────────────
def extract_region_full(address: str) -> dict:
    """주소 → {sido, sigungu, dong, region, short_region} 분리."""
    parts = address.split()
    sido    = parts[0] if len(parts) > 0 else ""
    sigungu = parts[1] if len(parts) > 1 else ""
    dong_match = re.search(r"(\S+동)", address)
    dong = dong_match.group(1) if dong_match else ""
    gu   = next((p for p in parts if p.endswith("구")), None)
    short = f"{gu} {dong}".strip() if gu and dong else (gu or sigungu)
    return {
        "sido":         sido,
        "sigungu":      sigungu,
        "dong":         dong,
        "region":       f"{sido} {sigungu}".strip(),
        "short_region": short,
    }


def extract_region(address: str) -> str:
    """page_content용 짧은 지역 표현."""
    info = extract_region_full(address)
    return info["short_region"]


# ── 카테고리 분류 상수 ────────────────────────────────────────────────────────
_ACTIVITY_CATS = {"헬스", "볼링", "스포츠", "체육관", "테니스", "당구장", "스포츠센터", "댄스"}
_FOOD_CATS     = {
    "냉면", "육류", "한식", "갈비", "돈까스", "샤브샤브", "해산물", "두부",
    "닭요리", "일식", "해물", "양식", "베트남", "삼계탕", "보쌈", "초밥",
    "치킨", "참치", "조개", "칼국수", "태국", "중식", "떡볶이", "장어", "이탈리안", "분식",
}
_CAFE_CATS     = {"카페", "디저트카페", "고양이카페"}
_PARK_CATS     = {"공원", "테마거리"}
_CULTURAL_CATS = {"공연장", "문화시설", "문화센터", "전시", "갤러리", "박물관"}


# ── purpose 동적 생성 ──────────────────────────────────────────────────────────
def infer_purpose(row: dict, kw_set: set) -> list[str]:
    cat   = row["카테고리"]
    crowd = row["crowd_level"]
    price = row.get("price_level", "중가")
    pool: set[str] = set()

    # 카테고리 방향성
    if cat in _ACTIVITY_CATS or kw_set & {"운동", "체험", "스포츠", "건강"}:
        pool.add("활동")

    if cat in _PARK_CATS or kw_set & {"산책", "자연", "힐링"}:
        pool |= {"데이트", "힐링"}

    if cat in _CULTURAL_CATS or kw_set & {"문화", "전시", "전통"}:
        pool |= {"데이트", "친구"}

    if cat in _CAFE_CATS:
        pool.add("데이트")
        if crowd == "낮음":
            pool.add("혼자")

    if cat in _FOOD_CATS:
        if price == "고가":
            pool |= {"데이트", "가족"}
        elif price == "저가":
            pool |= {"친구", "혼자"}
        else:  # 중가
            pool.add("혼자" if crowd == "낮음" else "친구")

    # 키워드 기반 추론
    if kw_set & {"SNS", "감성", "사진"}:
        pool.add("데이트")

    if kw_set & {"핫플"}:
        pool |= {"데이트", "친구"}

    if kw_set & {"가족"}:
        pool.add("가족")

    if kw_set & {"가성비", "맛집"}:
        pool |= {"친구", "혼자"}

    # crowd 보정
    if crowd == "낮음":
        pool |= {"혼자", "힐링"}
    elif crowd == "높음":
        pool.add("친구")

    # 최소 1개 보장
    if not pool:
        pool.add("혼자" if crowd == "낮음" else "친구")

    # 우선순위 정렬 후 최대 3개
    order = ["데이트", "친구", "혼자", "힐링", "활동", "가족"]
    ranked = [p for p in order if p in pool]
    if "가족" in kw_set and "가족" not in ranked[:3]:
        return ranked[:2] + ["가족"]
    return ranked[:3]


# ── mood 동적 생성 ─────────────────────────────────────────────────────────────
# 어휘: 조용한 / 활기찬 / 감성적인 / 힐링되는 / 가성비 / 특별한
def infer_mood(row: dict, kw_set: set) -> list[str]:
    cat   = row["카테고리"]
    crowd = row["crowd_level"]
    price = row.get("price_level", "중가")
    pool: set[str] = set()

    if kw_set & {"SNS", "감성", "사진", "핫플", "거리", "문화", "전통", "전시"}:
        pool.add("감성적인")

    if kw_set & {"산책", "자연", "힐링", "여유"} or cat in _PARK_CATS:
        pool.add("힐링되는")

    if crowd == "낮음" or kw_set & {"여유", "조용"}:
        pool.add("조용한")

    if crowd == "높음" or kw_set & {"핫플", "활동", "활기", "거리"}:
        pool.add("활기찬")

    if cat in _FOOD_CATS and price == "저가":
        pool.add("가성비")

    if cat in _FOOD_CATS and price == "고가":
        pool.add("특별한")

    # fallback (카테고리 기반, "편안한" 금지)
    if not pool:
        if cat in _ACTIVITY_CATS:
            pool.add("활기찬")
        elif cat in _PARK_CATS:
            pool.add("힐링되는")
        elif cat in _FOOD_CATS:
            pool.add("활기찬" if crowd == "높음" else "조용한")
        else:
            pool.add("감성적인" if crowd != "낮음" else "조용한")

    order = ["조용한", "활기찬", "감성적인", "힐링되는", "가성비", "특별한"]
    return [m for m in order if m in pool][:2]


# ── 텍스트 매핑 상수 ──────────────────────────────────────────────────────────
_CROWD_DESC = {
    "낮음": "혼잡도가 낮아 여유롭게 머무를 수 있다",
    "보통": "적당한 사람들과 편안한 분위기를 즐길 수 있다",
    "높음": "활기차고 사람들로 북적이는 분위기다",
}

_WEATHER_INDOOR_SENT = {
    ("비",       "실내"): "비 오는 날에도 방문하기 좋은 실내 공간이며",
    ("비",       "혼합"): "비 오는 날에는 실내 공간을 중심으로 이용하면 좋으며",
    ("비",       "실외"): "비 오는 날에는 야외 공간 이용에 주의가 필요하며",
    ("더위",     "실내"): "더운 날씨에 시원하게 즐길 수 있는 실내 공간이며",
    ("더위",     "실외"): "더운 날씨에도 탁 트인 실외에서 즐길 수 있으며",
    ("더위",     "혼합"): "더운 날씨에 실내외를 모두 이용할 수 있으며",
    ("추위",     "실내"): "추운 날씨에도 따뜻하게 이용할 수 있는 실내 공간이며",
    ("추위",     "실외"): "추운 날씨에도 신선한 실외 공간을 즐길 수 있으며",
    ("추위",     "혼합"): "추운 날씨에 실내외를 모두 이용할 수 있으며",
    ("모든 날씨", "실내"): "날씨에 관계없이 언제든 방문할 수 있는 실내 공간이며",
    ("모든 날씨", "실외"): "날씨에 관계없이 탁 트인 실외에서 즐길 수 있으며",
    ("모든 날씨", "혼합"): "날씨에 관계없이 실내외를 모두 이용할 수 있으며",
}

_PURPOSE_LABEL = {
    "데이트": "데이트",
    "혼자":  "혼자",
    "친구":  "친구",
    "힐링":  "힐링",
    "활동":  "활동",
    "가족":  "가족",
}


# ── page_content 생성 (FAISS용 자연어 문장) ──────────────────────────────────
def build_page_content(row: dict, purpose: list[str], mood: list[str], region: str) -> str:
    cat     = row["카테고리"]
    desc    = row["설명"]
    crowd   = row["crowd_level"]
    indoor  = row["indoor_outdoor"]
    weather = row["weather_fit"]

    # 목적 문구
    p_labels = [_PURPOSE_LABEL.get(p, p) for p in purpose]
    if len(p_labels) == 1:
        purpose_phrase = f"{p_labels[0]} 방문하기 좋은"
    elif len(p_labels) == 2:
        purpose_phrase = f"{p_labels[0]}나 {p_labels[1]} 방문하기 좋은"
    else:
        purpose_phrase = ", ".join(p_labels) + " 방문하기 좋은"

    # 분위기 문구 — 원형 그대로 붙여 vocabulary 매칭 보장
    mood_phrase = " ".join(mood)  # e.g., "조용한 감성적인"

    # 문장 1: 목적 + 분위기 + 지역 + 카테고리
    sent1 = f"{purpose_phrase} {mood_phrase} 분위기의 {region} {cat}."

    # 날씨+실내외 문구
    weather_indoor = _WEATHER_INDOOR_SENT.get(
        (weather, indoor),
        f"{indoor} 공간이며",
    )

    # 혼잡도 문구
    crowd_desc = _CROWD_DESC.get(crowd, "")

    # 최종 조합: "목적+분위기+카테고리. 날씨+실내, 혼잡도. 설명."
    return f"{sent1} {weather_indoor}, {crowd_desc}. {desc}"


# ── plain_text 생성 (BM25용 키-값 구조) ──────────────────────────────────────
def build_plain_text(
    row: dict, purpose: list[str], mood: list[str], region: str, kw_set: set
) -> str:
    name    = row["장소명"]
    cat     = row["카테고리"]
    desc    = row["설명"]
    crowd   = row["crowd_level"]
    indoor  = row["indoor_outdoor"]
    weather = row["weather_fit"]
    price   = row["price_level"]

    # 날씨 확장 토큰 (BM25 쿼리 다양성 대응)
    weather_tokens = weather
    if weather == "비":
        weather_tokens = "비 우천 비오는날"
    elif weather == "더위":
        weather_tokens = "더위 여름 시원"
    elif weather == "추위":
        weather_tokens = "추위 겨울 따뜻"

    # 실내외 확장 토큰
    indoor_tokens = indoor
    if indoor == "혼합":
        indoor_tokens = "실내 실외 혼합"

    # 혼잡도 확장 토큰
    crowd_tokens = crowd
    if crowd == "낮음":
        crowd_tokens = "낮음 조용한 한산"
    elif crowd == "높음":
        crowd_tokens = "높음 활기찬 혼잡"

    kw_str = " ".join(sorted(kw_set))

    lines = [
        f"장소명: {name}",
        f"카테고리: {cat}",
        f"추천 상황: {', '.join(purpose)}",
        f"분위기: {', '.join(mood)}",
        f"날씨: {weather_tokens}",
        f"실내외: {indoor_tokens}",
        f"혼잡도: {crowd_tokens}",
        f"가격대: {price}",
        f"설명: {desc}",
        f"키워드: {kw_str}",
        f"지역: {region}",
    ]
    return "\n".join(lines)


# ── persona.csv 기반 personality_tags 매핑 ───────────────────────────────────
def map_personality_tags(kw_set: set, persona_rows: list[dict]) -> tuple[list[str], bool, bool]:
    """
    장소 키워드 ∩ 성향 키워드 교집합으로 personality_tags 산출.
    균형형 → is_general 플래그 (personality_tags에서 제거).
    감성형 → is_visual_spot 플래그 (SNS/사진/감성 키워드 보유 시).
    personality_tags 최대 2개.
    """
    raw: list[str] = []
    for p in persona_rows:
        p_kws = {k.strip() for k in f"{p['핵심키워드']},{p['보조키워드']}".split(",")}
        if kw_set & p_kws:
            raw.append(p["성향"])
    is_general     = "균형형" in raw
    is_visual_spot = bool(kw_set & {"SNS", "사진", "감성"})
    tags = [t for t in raw if t not in ("균형형", "감성형")][:2]
    return tags, is_general, is_visual_spot


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main() -> None:
    with open(DATA_DIR / "places_extended.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    with open(DATA_DIR / "persona.csv", encoding="utf-8") as f:
        persona_rows = list(csv.DictReader(f))

    results: list[dict] = []
    for row in rows:
        kw_str  = row["키워드"]
        kw_set  = {k.strip() for k in kw_str.split(",")}
        address = row["주소"]
        region_info = extract_region_full(address)
        region      = region_info["short_region"]

        purpose = infer_purpose(row, kw_set)
        mood    = infer_mood(row, kw_set)
        personality_tags, is_general, is_visual_spot = map_personality_tags(kw_set, persona_rows)

        results.append({
            # ── 검색 텍스트 ──────────────────────────────────────────────────
            "id":           row["id"],
            "place_name":   row["장소명"],
            "category":     row["카테고리"],
            "purpose":      purpose,
            "mood":         mood,
            "page_content": build_page_content(row, purpose, mood, region),
            "plain_text":   build_plain_text(row, purpose, mood, region, kw_set),
            # ── retriever.py 호환 메타데이터 (전체 보존) ─────────────────────
            "metadata": {
                # 기존 필드 (하위 호환)
                "id":             int(row["id"]),
                "장소명":          row["장소명"],
                "카테고리":         row["카테고리"],
                "키워드":          kw_str,
                "설명":           row["설명"],
                "주소":           address,
                # 구조화 지역 정보
                "category":       row["카테고리"],
                "keywords":       list(kw_set),
                "region":         region_info["region"],
                "sido":           region_info["sido"],
                "sigungu":        region_info["sigungu"],
                "dong":           region_info["dong"],
                # 성향 태그
                "personality_tags": personality_tags,
                "is_general":       is_general,
                "is_visual_spot":   is_visual_spot,
                # extended 필드
                "stay_time":      row.get("stay_time", ""),
                "crowd_level":    row.get("crowd_level", ""),
                "best_time":      row.get("best_time", ""),
                "price_level":    row.get("price_level", ""),
                "indoor_outdoor": row.get("indoor_outdoor", ""),
                "weather_fit":    row.get("weather_fit", ""),
                # enriched 필드
                "purpose":        purpose,
                "mood":           mood,
            },
        })

    out = DATA_DIR / "places_enriched.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ── 검증 리포트 ────────────────────────────────────────────────────────────
    print(f"생성 완료: {out}  ({len(results)}개 장소)\n")

    all_pc = " ".join(r["page_content"] for r in results)
    all_pt = " ".join(r["plain_text"]   for r in results)

    check_words = [
        "데이트", "혼자", "친구", "가족",
        "조용한", "감성적인", "활기찬", "힐링되는",
        "비", "실내", "실외",
    ]
    print("── vocabulary coverage ──────────────────────")
    for w in check_words:
        pc = all_pc.count(w)
        pt = all_pt.count(w)
        status = "✅" if pc >= 5 and pt >= 5 else "⚠️ "
        print(f"  {status} {w:8s}: page_content={pc:3d}회  plain_text={pt:3d}회")

    print("\n── purpose 분포 ──────────────────────────────")
    all_purposes = [p for r in results for p in r["purpose"]]
    for p, cnt in Counter(all_purposes).most_common():
        bar = "█" * (cnt // 2)
        print(f"  {p:6s}: {cnt:3d}건  {bar}")

    print("\n── mood 분포 ─────────────────────────────────")
    all_moods = [m for r in results for m in r["mood"]]
    for m, cnt in Counter(all_moods).most_common():
        bar = "█" * (cnt // 2)
        print(f"  {m:8s}: {cnt:3d}건  {bar}")

    # 샘플 출력 (카페 1개, 공원 1개, 식당 1개)
    print("\n── 샘플 출력 ─────────────────────────────────")
    sample_cats = ["카페", "공원", "육류"]
    shown: set[str] = set()
    for r in results:
        if r["category"] in sample_cats and r["category"] not in shown:
            shown.add(r["category"])
            print(f"\n[ {r['place_name']} / {r['category']} ]")
            print(f"  purpose: {r['purpose']}")
            print(f"  mood:    {r['mood']}")
            print("  page_content:")
            for line in r["page_content"].split("\n"):
                print(f"    {line}")
            print(f"  plain_text (앞 300자):")
            print(f"    {r['plain_text'][:300]}")


if __name__ == "__main__":
    main()
