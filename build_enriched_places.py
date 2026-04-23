"""
places_extended.csv + persona.csv → data/places_enriched.json

생성 항목:
  - purpose: 장소별 동적 추론 (카테고리 고정값 금지)
  - mood:    키워드 + crowd 기반 동적 추론 (fallback "편안한" 금지)
  - page_content: FAISS용 자연어 문장 (데이트/혼자/비/실내 등 반드시 포함)
  - plain_text:   BM25용 키워드 열거 텍스트
  - metadata:     retriever.py 호환 전체 필드 (personality_tags, is_general 포함)
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
_ACTIVITY_CATS   = {"헬스", "볼링", "스포츠", "체육관", "테니스", "당구장", "스포츠센터", "댄스"}
_FOOD_CATS       = {
    "냉면", "육류", "한식", "갈비", "돈까스", "샤브샤브", "해산물", "두부",
    "닭요리", "일식", "해물", "양식", "베트남", "삼계탕", "보쌈", "초밥",
    "치킨", "참치", "조개", "칼국수", "태국", "중식", "떡볶이", "장어", "이탈리안", "분식",
}
_CAFE_CATS       = {"카페", "디저트카페", "고양이카페"}
_PARK_CATS       = {"공원", "테마거리"}
_CULTURAL_CATS   = {"공연장", "문화시설", "문화센터", "전시", "갤러리", "박물관"}


# ── purpose 동적 생성 ──────────────────────────────────────────────────────────
def infer_purpose(row: dict, kw_set: set) -> list[str]:
    cat   = row["카테고리"]
    crowd = row["crowd_level"]
    pool: set[str] = set()

    # 카테고리 방향성
    if cat in _ACTIVITY_CATS or kw_set & {"스포츠", "건강", "활동", "체험"}:
        pool |= {"친구", "활동"}

    if cat in _PARK_CATS or kw_set & {"산책", "자연", "힐링"}:
        pool |= {"데이트", "힐링"}

    if cat in _CULTURAL_CATS or kw_set & {"문화", "전시", "전통"}:
        pool |= {"데이트", "친구"}

    if cat in _CAFE_CATS:
        pool.add("데이트")
        if crowd == "낮음":
            pool.add("혼자")

    if cat in _FOOD_CATS:
        pool.add("친구" if crowd != "낮음" else "혼자")

    # 키워드 기반 추론
    if kw_set & {"SNS", "감성", "사진", "핫플"}:
        pool |= {"데이트", "친구"}

    if kw_set & {"가족"}:
        pool.add("가족")

    if kw_set & {"가성비", "맛집", "음식"}:
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
    # "가족" 키워드가 명시된 장소는 가족 목적을 반드시 포함 (순위와 무관)
    order = ["데이트", "친구", "혼자", "힐링", "활동", "가족"]
    ranked = [p for p in order if p in pool]
    if "가족" in kw_set and "가족" not in ranked[:3]:
        return ranked[:2] + ["가족"]
    return ranked[:3]


# ── mood 동적 생성 ─────────────────────────────────────────────────────────────
def infer_mood(row: dict, kw_set: set) -> list[str]:
    cat   = row["카테고리"]
    crowd = row["crowd_level"]
    pool: set[str] = set()

    if kw_set & {"SNS", "감성", "사진", "핫플", "거리", "문화", "전통", "전시"}:
        pool.add("감성")

    if kw_set & {"산책", "자연", "힐링", "여유"}:
        pool.add("힐링")

    if crowd == "낮음" or kw_set & {"여유", "조용"}:
        pool.add("조용")

    if crowd == "높음" or kw_set & {"핫플", "활동", "활기", "거리"}:
        pool.add("활기")

    # fallback (카테고리 기반, "편안한" 금지)
    if not pool:
        if cat in _ACTIVITY_CATS:
            pool.add("활기")
        elif cat in _PARK_CATS:
            pool.add("힐링")
        elif cat in _FOOD_CATS:
            pool.add("활기" if crowd == "높음" else "조용")
        else:
            pool.add("감성" if crowd != "낮음" else "조용")

    order = ["감성", "활기", "힐링", "조용"]
    return [m for m in order if m in pool][:2]


# ── 텍스트 매핑 상수 ──────────────────────────────────────────────────────────
_CROWD_LABEL = {"낮음": "한산", "보통": "보통", "높음": "혼잡"}
_CROWD_DESC  = {
    "낮음": "한산하여 조용하게 이용할 수 있다",
    "보통": "적당한 사람들이 있어 편안한 분위기다",
    "높음": "사람이 많아 활기찬 분위기다",
}
_INDOOR_LABEL = {"실내": "실내 공간", "실외": "실외 공간", "혼합": "실내외 모두 이용 가능한 공간"}
_INDOOR_SENT  = {
    "실내": "실내 공간으로 날씨의 영향을 받지 않는다",
    "실외": "실외 공간으로 탁 트인 환경을 즐길 수 있다",
    "혼합": "실내외를 모두 이용할 수 있다",
}
_WEATHER_SENT = {
    "비":      "비 오는 날에도 방문하기 좋다",
    "더위":    "더운 날씨에 시원하게 즐길 수 있다",
    "추위":    "추운 날씨에도 따뜻하게 이용할 수 있다",
    "모든 날씨": "날씨에 관계없이 언제든 방문할 수 있다",
}
_PRICE_LABEL = {"저가": "저렴한 편", "중가": "적당한 가격", "고가": "가격대가 높은 편"}


# ── page_content 생성 (FAISS용) ───────────────────────────────────────────────
def build_page_content(row: dict, purpose: list[str], mood: list[str], region: str) -> str:
    name       = row["장소명"]
    cat        = row["카테고리"]
    desc       = row["설명"]
    crowd      = row["crowd_level"]
    indoor     = row["indoor_outdoor"]
    weather    = row["weather_fit"]
    stay       = row["stay_time"]
    best_time  = row["best_time"]
    price      = row["price_level"]

    purpose_str = ", ".join(purpose)
    mood_str    = ", ".join(mood)

    # 혼자/데이트/친구 가능 여부 문장 (핵심 vocabulary 강제 포함)
    usage: list[str] = []
    if "혼자" in purpose:
        usage.append("혼자 방문하기 좋다")
    if "데이트" in purpose:
        usage.append("데이트 코스로 적합하다")
    if "친구" in purpose:
        usage.append("친구와 함께 즐기기 좋다")
    if "가족" in purpose:
        usage.append("가족 단위 방문에도 좋다")
    if "힐링" in purpose:
        usage.append("힐링이 필요할 때 찾기 좋은 곳이다")
    if "활동" in purpose:
        usage.append("활동적인 여가를 즐기기 좋다")
    usage_sentence = " ".join(usage) + "."

    # 날씨 + 실내외 조합 문장 (핵심 vocabulary 강제 포함)
    if weather == "비" and indoor == "실내":
        weather_indoor = "비 오는 날에도 실내에서 편안하게 이용할 수 있다."
    elif weather == "비" and indoor == "실외":
        weather_indoor = "비가 올 경우 야외 공간 이용에 주의가 필요하다."
    elif weather == "비" and indoor == "혼합":
        weather_indoor = "비 오는 날에는 실내 공간을 중심으로 이용하면 좋다."
    elif indoor == "실내":
        weather_indoor = f"{_WEATHER_SENT.get(weather, weather)} 실내 공간으로 쾌적하다."
    elif indoor == "실외":
        weather_indoor = f"{_WEATHER_SENT.get(weather, weather)} 실외 공간이라 자연을 가까이 느낄 수 있다."
    else:
        weather_indoor = f"{_WEATHER_SENT.get(weather, weather)} {_INDOOR_SENT.get(indoor, '')}."

    lines = [
        f"{purpose_str}에 어울리는 {region}의 {cat}.",
        "",
        f"장소명: {name}",
        f"추천 상황: {purpose_str}",
        f"분위기: {mood_str}",
        f"설명: {desc} {usage_sentence}",
        f"혼잡도: {_CROWD_LABEL.get(crowd, crowd)} ({_CROWD_DESC.get(crowd, '')})",
        f"실내외: {_INDOOR_LABEL.get(indoor, indoor)}",
        f"날씨: {weather_indoor}",
        f"방문 시간대: {best_time} | 체류 시간: {stay}분 | 가격대: {_PRICE_LABEL.get(price, price)}",
        f"지역: {region}",
    ]
    return "\n".join(lines)


# ── plain_text 생성 (BM25용) ──────────────────────────────────────────────────
def build_plain_text(
    row: dict, purpose: list[str], mood: list[str], region: str, kw_set: set
) -> str:
    name      = row["장소명"]
    cat       = row["카테고리"]
    desc      = row["설명"]
    crowd     = row["crowd_level"]
    indoor    = row["indoor_outdoor"]
    weather   = row["weather_fit"]
    stay      = row["stay_time"]
    price     = row["price_level"]
    best_time = row["best_time"]

    # 확장 키워드: weather / indoor / crowd 어휘 명시적 추가
    extended: list[str] = list(kw_set)

    if crowd == "낮음":
        extended += ["조용", "한산"]
    elif crowd == "높음":
        extended += ["활기", "혼잡"]

    if indoor == "실내":
        extended.append("실내")
    elif indoor == "실외":
        extended.append("실외")
    elif indoor == "혼합":
        extended += ["실내", "실외"]

    if weather == "비":
        extended += ["비", "우천", "비오는날"]
    elif weather == "더위":
        extended += ["더위", "여름", "시원"]
    elif weather == "추위":
        extended += ["추위", "겨울", "따뜻"]

    extended += purpose
    extended += mood

    # 순서 보존 중복 제거
    seen: set[str] = set()
    deduped: list[str] = []
    for tok in extended:
        if tok not in seen:
            seen.add(tok)
            deduped.append(tok)

    kw_combined = " ".join(deduped)

    tokens = [
        f"장소명 {name}",
        f"카테고리 {cat}",
        f"추천상황 {' '.join(purpose)}",
        f"분위기 {' '.join(mood)}",
        f"키워드 {kw_combined}",
        f"설명 {desc}",
        f"날씨 {weather}",
        f"실내외 {indoor}",
        f"가격 {price}",
        f"혼잡도 {crowd}",
        f"체류시간 {stay}",
        f"방문시간 {best_time}",
        f"지역 {region}",
    ]
    return " ".join(tokens)


# ── persona.csv 기반 personality_tags 매핑 ───────────────────────────────────
def map_personality_tags(kw_set: set, persona_rows: list[dict]) -> tuple[list[str], bool]:
    """
    장소 키워드 ∩ 성향 키워드 교집합으로 personality_tags 산출.
    균형형은 is_general 플래그로 분리하여 personality_tags에서 제거.
    """
    raw: list[str] = []
    for p in persona_rows:
        p_kws = {k.strip() for k in f"{p['핵심키워드']},{p['보조키워드']}".split(",")}
        if kw_set & p_kws:
            raw.append(p["성향"])
    is_general = "균형형" in raw
    return [t for t in raw if t != "균형형"], is_general


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
        personality_tags, is_general = map_personality_tags(kw_set, persona_rows)

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

    check_words = ["데이트", "혼자", "친구", "가족", "조용", "감성", "활기", "힐링", "비", "실내", "실외"]
    print("── vocabulary coverage ──────────────────────")
    for w in check_words:
        pc = all_pc.count(w)
        pt = all_pt.count(w)
        status = "✅" if pc >= 5 and pt >= 5 else "⚠️ "
        print(f"  {status} {w:6s}: page_content={pc:3d}회  plain_text={pt:3d}회")

    print("\n── purpose 분포 ──────────────────────────────")
    all_purposes = [p for r in results for p in r["purpose"]]
    for p, cnt in Counter(all_purposes).most_common():
        bar = "█" * (cnt // 2)
        print(f"  {p:6s}: {cnt:3d}건  {bar}")

    print("\n── mood 분포 ─────────────────────────────────")
    all_moods = [m for r in results for m in r["mood"]]
    for m, cnt in Counter(all_moods).most_common():
        bar = "█" * (cnt // 2)
        print(f"  {m:6s}: {cnt:3d}건  {bar}")

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
            print(f"  plain_text (앞 200자):")
            print(f"    {r['plain_text'][:200]}")


if __name__ == "__main__":
    main()
