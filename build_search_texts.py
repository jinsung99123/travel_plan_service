"""
검색 최적화 텍스트 생성 스크립트
places_expanded.json + places_extended.csv → JSON 출력
page_content (FAISS용) + plain_text (BM25용)
"""
import csv, json, re, sys
sys.stdout.reconfigure(encoding='utf-8')

# ── 분위기 키워드 매핑 ─────────────────────────────────────────────────────────
_MOOD_TOKENS = {
    '조용': '조용한',
    '여유': '여유',
    '감성': '감성',
    '힐링': '힐링',
    '활기': '활기',
    '핫플': '활기',
    'SNS': '감성',
}

def extract_mood(keywords: str, description: str) -> str:
    seen, moods = set(), []
    for token, label in _MOOD_TOKENS.items():
        if (token in keywords or token in description) and label not in seen:
            seen.add(label)
            moods.append(label)
    return ','.join(moods) if moods else '편안한'


# ── 주소 → 지역 축약 ──────────────────────────────────────────────────────────
def extract_region(address: str) -> str:
    """'서울 노원구 공릉동 ...' → '노원구 공릉동'"""
    parts = address.split()
    # 구 단위 찾기
    gu = next((p for p in parts if p.endswith('구')), None)
    # 동/로/길 단위 찾기 (동 우선)
    dong = next((p for p in parts if p.endswith('동')), None)
    if gu and dong:
        return f"{gu} {dong}"
    if gu:
        return gu
    return parts[1] if len(parts) > 1 else address


# ── 레이블 변환 ───────────────────────────────────────────────────────────────
_CROWD_LABEL = {'높음': '혼잡', '보통': '보통', '낮음': '한산'}
_PRICE_LABEL = {'고가': '가격대 높음', '중가': '가격대 보통', '저가': '가격대 낮음'}
_INDOOR_LABEL = {'실내': '실내', '실외': '실외', '실내외': '실내외 모두'}


def _l(mapping: dict, val: str, default=None) -> str:
    return mapping.get(val, default or val)


# ── page_content 생성 (FAISS용) ───────────────────────────────────────────────
def build_page_content(row: dict, expanded: dict, mood: str, region: str) -> str:
    name      = expanded['name']
    category  = row['카테고리']
    purpose   = expanded['purpose']
    desc      = expanded['description']
    best_time = row['best_time']
    stay      = row['stay_time']
    crowd     = _l(_CROWD_LABEL, row['crowd_level'])
    price     = _l(_PRICE_LABEL, row['price_level'])
    indoor    = _l(_INDOOR_LABEL, row['indoor_outdoor'])
    weather   = row['weather_fit']

    lines = [
        f"{purpose} 여행자가 {region}에서 방문하기 좋은 {category}.",
        "",
        f"장소명: {name}",
        f"추천 상황: {purpose}",
        f"분위기: {mood}",
        f"설명: {desc}",
        "",
        f"방문 시간대: {best_time} | 체류 시간: {stay} | 혼잡도: {crowd} | {price}",
        f"실내외: {indoor} | 날씨: {weather}",
    ]
    return '\n'.join(lines)


# ── plain_text 생성 (BM25용) ──────────────────────────────────────────────────
def build_plain_text(row: dict, expanded: dict, mood: str, region: str) -> str:
    name     = expanded['name']
    category = row['카테고리']
    keywords = expanded['keywords']
    purpose  = expanded['purpose']
    desc     = expanded['description']
    weather  = row['weather_fit']
    indoor   = row['indoor_outdoor']
    price    = row['price_level']
    crowd    = row['crowd_level']
    stay     = row['stay_time']
    best     = row['best_time']

    lines = [
        f"장소명 {name}",
        f"카테고리 {category}",
        f"키워드 {keywords}",
        f"추천상황 {purpose}",
        f"분위기 {mood}",
        f"설명 {desc}",
        f"날씨 {weather}",
        f"실내외 {indoor}",
        f"가격 {price}",
        f"혼잡도 {crowd}",
        f"체류시간 {stay}",
        f"방문시간 {best}",
        f"지역 {region}",
    ]
    return ' '.join(lines)


# ── 메인 처리 ─────────────────────────────────────────────────────────────────
# CSV 로드 (key: 장소명)
csv_map = {}
with open('data/places_extended.csv', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        csv_map[row['장소명']] = row

# 확장 JSON 로드
with open('data/places_expanded.json', encoding='utf-8') as f:
    expanded_list = json.load(f)

results = []
for exp in expanded_list:
    name = exp['name']
    row  = csv_map.get(name)
    if row is None:
        continue  # 매칭 실패 스킵

    mood   = extract_mood(exp['keywords'], exp['description'])
    region = extract_region(row['주소'])

    results.append({
        "name":         name,
        "page_content": build_page_content(row, exp, mood, region),
        "plain_text":   build_plain_text(row, exp, mood, region),
    })

print(json.dumps(results, ensure_ascii=False, indent=2))
