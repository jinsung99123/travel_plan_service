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

# 가족 purpose 직접 확장 대상 카테고리 (성인 전용·야간 시설 제외)
_FAMILY_CAT_DIRECT = {
    "공원", "테마거리", "공연장", "문화시설", "문화센터",
    "전통", "볼링", "스포츠", "체육관",
}
_FAMILY_EXCL_KW  = {"술", "야간", "바", "클럽", "성인"}
_FAMILY_EXCL_CAT = {"당구장", "헬스", "테니스", "댄스"}  # 성인 전용 활동


def _is_family_place(row: dict, kw_set: set) -> bool:
    """가족 purpose 추가 여부 판단 (조건 기반, 무조건 추가 금지)."""
    if kw_set & _FAMILY_EXCL_KW:
        return False
    cat   = row["카테고리"]
    crowd = row["crowd_level"]
    if cat in _FAMILY_EXCL_CAT:
        return False
    if cat in _FAMILY_CAT_DIRECT:
        return True
    if kw_set & {"체험", "전시", "놀이"} and crowd != "높음":
        return True
    if kw_set & {"공원", "산책"} and crowd == "낮음":
        return True
    if "가족" in kw_set:
        return True
    return False


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

    if kw_set & {"가성비", "맛집"}:
        pool |= {"친구", "혼자"}

    # 가족 확장 (카테고리·키워드·조건 기반)
    if _is_family_place(row, kw_set):
        pool.add("가족")

    # crowd 보정
    if crowd == "낮음":
        pool |= {"혼자", "힐링"}
    elif crowd == "높음":
        pool.add("친구")

    # 최소 1개 보장
    if not pool:
        pool.add("혼자" if crowd == "낮음" else "친구")

    # 우선순위 정렬 — 가족 포함 시 최대 4개 (상위 3개 + 가족)
    order  = ["데이트", "친구", "혼자", "힐링", "활동", "가족"]
    ranked = [p for p in order if p in pool]
    if "가족" not in pool:
        return ranked[:3]
    if "가족" in ranked[:3]:
        return ranked[:3]
    return [p for p in ranked if p != "가족"][:3] + ["가족"]


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


# ── 장소별 고유 특징 문장 (FAISS 임베딩 분산용) ─────────────────────────────
# 카페/음식점은 page_content가 거의 동일해 임베딩이 수렴하는 문제를 방지.
# 장소명 기준 룩업 → build_page_content() 마지막에 "특징: ..." 추가.
_UNIQUE_FEATURE: dict[str, str] = {
    # ── 카페 ─────────────────────────────────────────────────────────────────
    "스타벅스 공릉DT점":
        "드라이브 스루 이용이 가능해 차에서 내리지 않고도 커피를 주문할 수 있는 편리한 공간이다.",
    "할리스 노원문화의거리점":
        "노원 문화의 거리 한복판에 자리해 공연·전시 관람 전후로 자연스럽게 들르기 좋은 위치다.",
    "스타벅스 노원역점":
        "노원역과 직결되어 이동 중 빠르게 음료를 픽업하거나 약속 전 대기하기 편리하다.",
    "노원두물마루 커피&스낵":
        "중랑천 자전거길 바로 옆에 위치해 산책이나 라이딩 뒤 잠깐 쉬어가기 좋은 소박한 공간이다.",
    "카페포레스트":
        "실내외를 모두 갖춘 자연 친화적 공간으로, 야외 테라스에서 나무와 풀 내음을 맡으며 커피를 즐길 수 있다.",
    "오피셜커피":
        "트렌디한 인테리어와 깔끔한 동선으로 SNS 사진 촬영에 인기가 높으며, 혼자 노트북 작업하기에도 적합하다.",
    "투썸플레이스 상계점":
        "다양한 케이크와 음료를 한 자리에서 즐길 수 있어 생일 케이크 픽업이나 기념일 방문에도 활용도가 높다.",
    "스타벅스 태릉입구역DT점":
        "드라이브 스루 전용 운영 방식으로, 바쁜 아침 출근길에 빠르게 음료를 받아갈 수 있다.",
    "스타벅스 상계초교사거리점":
        "상계 주거 밀집 지역 한가운데 위치해 동네 카페처럼 편안하게 일상적으로 이용할 수 있는 공간이다.",
    "호이폴로이커피로스터스":
        "직접 로스팅한 스페셜티 원두를 사용하며, 원두 향과 추출 방식에 집중해 커피 본연의 맛을 즐기는 이들에게 적합하다.",
    "에슬로우커피 공릉점":
        "이름처럼 느리고 여유롭게 시간을 보내도 눈치 보이지 않는 분위기로, 긴 독서나 노트북 작업에 안성맞춤이다.",
    "투썸플레이스 태릉입구역점":
        "시즌 한정 케이크 메뉴로 유명하며, 역 인근이라 지인과의 만남 장소로 자주 활용된다.",
    "마카모예 브레드바":
        "매일 직접 구운 빵과 베이커리 디저트를 함께 즐길 수 있어, 아침 식사나 브런치 겸 카페로 이용하기 좋다.",
    "투썸플레이스 석계역점":
        "석계역과 가까워 이동 전후 빠르게 들르거나 모임 장소로 잡기 편리한 프랜차이즈 카페다.",
    "꽁냥꽁냥":
        "다양한 고양이와 직접 교감할 수 있는 체험형 공간으로, 반려묘를 키우지 않아도 힐링이 되는 독특한 카페다.",
    "메가MGC커피 공릉도깨비시장점":
        "인근 도깨비시장 구경 후 저렴하게 음료 한 잔 즐기기 좋으며, 자주 방문하는 단골이 많은 가성비 카페다.",
    "브런힐":
        "브런치와 커피를 한 자리에서 즐길 수 있어 주말 오전 늦게 일어나 여유롭게 식사하고 싶을 때 좋다.",
    "웨이스테이션":
        "넓은 창과 밝은 채광이 특징이며, 테이블 간격이 넓어 옆 테이블 신경 쓰지 않고 대화하기 편하다.",
    "오누이":
        "동네 작은 카페 특유의 아늑함이 있어 혼자 책을 읽거나 조용히 생각을 정리하기 좋은 공간이다.",
    "커피베스코":
        "커피 품질에 집중한 소규모 카페로, 사람이 적어 조용히 한 잔을 천천히 음미하기 적합한 곳이다.",
    "아너카페":
        "골목 안쪽에 위치해 외부 소음이 차단되며, 창가 자리에서 혼자만의 시간을 보내기 안성맞춤이다.",
    "할리스 태릉입구역점":
        "태릉입구역 바로 인근으로 출퇴근 동선에서 벗어나지 않고 이용하기 좋은 역세권 카페다.",
    "감각":
        "공간 곳곳의 소품과 조명이 사진 찍기 좋게 배치되어 있어 인스타그램 감성을 즐기는 방문객에게 인기다.",
    "오르름":
        "2층 높이에서 내려다보이는 골목 뷰와 차분한 내부 구성으로, 일상의 소음에서 벗어나 머리를 식히기 좋다.",
    "시즌":
        "계절마다 새롭게 바뀌는 한정 디저트 메뉴가 강점으로, 재방문할 때마다 다른 경험을 할 수 있는 아늑한 카페다.",
    "씨즌 서울 바이 홍신애":
        "요리연구가 홍신애가 운영하는 브랜드 카페로, 메뉴 선택에 있어 셰프의 손맛이 느껴지는 퀄리티 있는 공간이다.",
    "무이로커피":
        "작은 규모에서 오는 주인장의 온기가 느껴지는 곳으로, 조용히 단둘이 대화를 나누기에 이상적인 카페다.",
    "스타쿠빙":
        "독특한 빙수와 디저트 메뉴가 SNS에서 화제가 된 카페로, 시각적으로 매력적인 메뉴 구성이 특징이다.",
    "탐앤탐스 하계점":
        "넓은 테이블과 여유로운 좌석 배치로 그룹 방문이나 오랜 대화, 스터디 모임에 적합한 카페다.",
    "헤미스":
        "세련된 톤의 공간 구성과 낮은 조명이 어우러져, 커플이나 친구들과 분위기 있는 시간을 보내기 좋다.",
    "아웃캐스트":
        "개성 있는 콘셉트의 인테리어와 독특한 메뉴 조합으로 젊은 층 사이에서 알려진 핫플레이스 카페다.",
    "니토":
        "외부에서 잘 보이지 않는 조용한 골목 안 카페로, 아는 사람만 찾는 숨은 공간 특유의 분위기가 매력이다.",
    "우지커피 중계은행사거리점":
        "주변 카페 대비 합리적인 가격에 퀄리티 있는 커피를 제공해 근처 직장인과 학생의 단골이 많다.",
    "후아나 석계점":
        "아기자기한 소품과 따뜻한 조명으로 꾸며진 아늑한 인테리어로, 단둘이 오래 머물며 이야기 나누기 좋다.",
    # ── 음식점 ───────────────────────────────────────────────────────────────
    "강강술래 상계지점":
        "한우 부위별 프리미엄 코스로 제공하며, 특별한 날의 기념 식사나 비즈니스 자리에 어울리는 고급 고깃집이다.",
    "경복식당":
        "가정집 분위기에서 차려지는 저렴한 한식으로, 부담 없이 혼자 든든한 한 끼를 해결하기 좋다.",
    "크래버대게나라 노원점":
        "킹크랩과 대게 요리를 메인으로 활기찬 분위기 속에서 즐기며, 가족 모임이나 특별 외식에 어울린다.",
    "감동식당":
        "푸짐한 반찬과 국물 요리가 가정식 느낌을 주며, 혼밥으로도 부담 없이 이용할 수 있는 저렴한 한식당이다.",
    "제일콩집":
        "직접 만든 두부와 콩 기반 메뉴로 구성되어, 기름진 음식 대신 담백하고 건강한 한 끼를 원할 때 좋다.",
    "닭한마리 공릉본점":
        "공릉동 오랜 단골 맛집으로, 진한 닭 국물이 우러나기까지 기다림이 있지만 그만한 가치가 있다.",
    "전통평양냉면 제형면옥":
        "화학 조미료 없이 고기 육수만으로 끓인 정통 평양냉면으로, 냉면 마니아들이 즐겨 찾는 전문점이다.",
    "노원돈부리":
        "두툼한 토핑과 적당한 양으로 가볍고 든든하게 한 끼를 해결할 수 있는 일식 덮밥 전문 식당이다.",
    "이오냉면 노원점":
        "탄력 있는 면발과 깔끔하게 정돈된 양념이 특징으로, 담백하고 시원하게 즐기기 좋은 냉면 전문점이다.",
    "참만나":
        "숯불에 직접 구워 제공하는 프리미엄 갈비로, 품질 높은 고기와 반찬 구성이 중요한 자리에 어울린다.",
    "털보고된이 본점":
        "문화의 거리 인근에 자리한 해물 전문점으로, 다양한 해물 요리를 한 자리에서 골라 먹을 수 있다.",
    "예향정 노원점":
        "한정식 스타일의 반찬 구성이 특징으로, 제대로 차려진 한식 한 끼가 먹고 싶을 때 찾기 좋다.",
    "로니로티 노원점":
        "파스타와 리소토 등 다양한 양식 메뉴를 즐길 수 있으며, 편안한 분위기에서 식사와 대화를 함께 하기 좋다.",
    "삼덕식당":
        "기본에 충실한 두꺼운 고기와 정갈한 반찬이 강점으로, 여럿이 모여 부담 없이 고기를 구워 먹기 좋다.",
    "온달왕돈까스 상계점":
        "바삭한 튀김 옷과 두꺼운 고기 두께로 가성비 높은 돈까스를 제공하는 상계동 동네 맛집이다.",
    "돈까스먹는용만이":
        "소스 종류가 다양하고 가격이 저렴해 학생과 직장인 사이에서 자주 선택받는 돈까스 식당이다.",
    "하노이별 공릉점":
        "쌀국수와 분짜 등 정통 베트남 스타일을 저렴한 가격에 즐길 수 있어 이국적인 식사를 원할 때 좋다.",
    "소담촌 노원역점":
        "신선한 채소와 담백한 육수가 기본인 깔끔한 샤브샤브로, 가볍고 건강한 식사를 원할 때 어울린다.",
    "영양센타 노원점":
        "통닭에 인삼과 황기를 넣어 오래 푹 고아낸 삼계탕으로, 체력 보충이나 건강을 챙기고 싶을 때 추천한다.",
    "엄마마늘보쌈":
        "마늘을 듬뿍 넣어 삶은 보쌈 수육과 아삭한 겉절이의 조합이 일품이며, 술자리나 모임 식사에 잘 어울린다.",
    "나승준함흥냉면":
        "함흥 스타일의 쫄깃한 면발과 매콤한 비빔 소스가 특징으로, 평양냉면과 다른 강렬한 맛을 원하는 이들에게 적합하다.",
    "마포생고기":
        "두껍게 썬 생고기를 직접 구워 먹는 방식으로, 신선한 고기 그대로의 맛을 즐기는 이들에게 인기다.",
    "샤브20 노원상계점":
        "무한리필 방식의 샤브샤브로, 소량씩 다양한 재료를 골라 즐길 수 있어 부담 없는 가성비 식사가 가능하다.",
    "피노키오냉면":
        "새콤달콤한 양념이 배어든 냉면이 대표 메뉴로, 여름철뿐 아니라 사시사철 단골이 많은 냉면집이다.",
    "경성초밥":
        "신선한 재료로 만든 코스 초밥을 합리적인 가격으로 즐길 수 있어, 특별하지만 부담 없는 일식 식사를 원할 때 적합하다.",
    "진미치킨숯불바베큐":
        "숯불에 직접 구워낸 바비큐 치킨으로, 연기 향이 배어 일반 치킨과 차별화된 깊은 맛이 특징이다.",
    "다미참치":
        "참치 부위별 다양한 회를 세트로 구성하여 제공하며, 프리미엄 저녁 식사나 기념일 방문에 어울리는 곳이다.",
    "왕십리조개창고 노원점":
        "여러 종류의 조개를 직접 구워 먹는 방식으로, 여럿이 함께 왁자지껄하게 즐기기 좋은 모임 식당이다.",
    "엄마손칼국수":
        "손으로 직접 밀어 만든 면과 사골 베이스 진한 국물이 특징으로, 속이 든든하게 채워지는 한식 식당이다.",
    "썸머타이":
        "팟타이, 그린커리 등 현지에 가까운 태국 음식을 즐길 수 있어 이국적인 미식 경험을 원할 때 좋다.",
    "중원":
        "짜장면과 짬뽕 등 기본 메뉴에 충실하면서 코스 요리도 가능한 중식당으로, 단체 모임에도 이용할 수 있다.",
    "쪼매매운떡볶이 공릉점":
        "강렬한 매운 맛이 특징인 떡볶이 전문점으로, 매운 음식 도전을 즐기는 이들에게 인기 있는 분식집이다.",
    "해품장 팔팔장어 본점":
        "양념장어구이와 소금구이를 모두 즐길 수 있으며, 여름 보양식 또는 스태미나 보충 식사로 추천하는 전문점이다.",
    "페페그라노":
        "화덕 피자와 생면 파스타가 메인인 이탈리안 레스토랑으로, 식사와 와인을 곁들이기 좋은 분위기다.",
    "동선식당":
        "시원한 갈비탕과 찜갈비가 함께 인기 있는 식당으로, 지역 단골들이 오래 믿고 찾는 노원구의 터줏대감이다.",
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

    # 최종 조합: "목적+분위기+카테고리. 날씨+실내, 혼잡도. 설명. 특징."
    base = f"{sent1} {weather_indoor}, {crowd_desc}. {desc}"
    feature = _UNIQUE_FEATURE.get(row.get("장소명", ""), "")
    result = f"{base} 특징: {feature}" if feature else base

    if "활동" in purpose:
        result += " 이 장소는 운동, 체험, 스포츠 활동을 즐기기 좋은 장소입니다."

    # 메타데이터 기반 특징 문장 (최대 2개 — 문서 간 임베딩 차별화)
    price   = row.get("price_level", "중가")
    kw_set  = {k.strip() for k in row.get("키워드", "").split(",")}
    feature_lines: list[str] = []

    if crowd == "낮음":
        feature_lines.append("조용하고 한산한 분위기로 혼자 방문하기 좋습니다.")
    elif crowd == "높음":
        feature_lines.append("활기찬 분위기로 사람들과 함께 즐기기 좋습니다.")

    if indoor == "실내":
        feature_lines.append("날씨와 관계없이 편안하게 이용할 수 있는 실내 공간입니다.")
    elif indoor == "실외":
        feature_lines.append("야외 공간에서 자연을 느끼며 시간을 보낼 수 있습니다.")

    if price == "저가":
        feature_lines.append("가성비 좋은 가격대로 부담 없이 방문할 수 있습니다.")
    elif price == "고가":
        feature_lines.append("특별한 날 방문하기 좋은 분위기와 가격대를 갖추고 있습니다.")

    if kw_set & {"SNS", "사진"}:
        feature_lines.append("사진 촬영이나 SNS 공유에 적합한 공간입니다.")
    if kw_set & {"뷰", "루프탑"}:
        feature_lines.append("전망이 좋아 여유롭게 시간을 보내기 좋은 장소입니다.")
    if "데이트" in kw_set:
        feature_lines.append("연인과 함께 방문하기 좋은 분위기를 제공합니다.")

    if feature_lines:
        result += " " + " ".join(feature_lines[:2])
    return result


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
    plain = "\n".join(lines)

    if "활동" in purpose:
        extra = "활동 활동적인 운동 체험 스포츠 레저 액티비티 야외활동 코스 데이트코스"
        plain += " " + extra
    return plain


# ── persona.csv 기반 personality_tags 매핑 ───────────────────────────────────
def map_personality_tags(kw_set: set, persona_rows: list[dict]) -> tuple[list[str], bool, bool]:
    """
    장소 키워드 ∩ 성향 키워드 교집합으로 personality_tags 산출.

    is_general: 3가지 이상 성향에 해당 → 범용 장소 (중립 처리)
    is_visual_spot: SNS/사진/감성 키워드 보유 시 True
    균형형 포함 / 감성형 제외 (is_visual_spot으로 별도 처리) / cap 제거
    """
    raw: list[str] = []
    for p in persona_rows:
        p_kws = {k.strip() for k in f"{p['핵심키워드']},{p['보조키워드']}".split(",")}
        if kw_set & p_kws:
            raw.append(p["성향"])
    is_visual_spot = bool(kw_set & {"SNS", "사진", "감성"})
    tags = [t for t in raw if t != "감성형"]  # 균형형 포함, 감성형만 제외
    is_general = len(tags) >= 3
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
