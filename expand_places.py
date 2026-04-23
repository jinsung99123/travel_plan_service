"""
장소 데이터 확장 스크립트
purpose / description 강화 / keywords 확장 → JSON 출력
"""
import csv, json, re, sys
sys.stdout.reconfigure(encoding='utf-8')

# ── 카테고리 분류 ─────────────────────────────────────────────────────────────
CAFE_CATS   = {'카페','디저트카페','고양이카페'}
FOOD_CATS   = {'육류','한식','해산물','두부','닭요리','냉면','일식','갈비','해물',
               '베트남','샤브샤브','삼계탕','보쌈','돈까스','초밥','치킨','참치',
               '조개','장어','칼국수','떡볶이','태국','중식','이탈리안','양식'}
PARK_CATS   = {'공원'}
SPORT_CATS  = {'스포츠','헬스','테니스','볼링','체육관','스포츠센터','당구장'}
CULTURE_CATS= {'공연장','문화시설','문화센터','테마거리','전통'}


# ── purpose 결정 ──────────────────────────────────────────────────────────────
def get_purpose(cat, desc, kws_raw):
    kws = kws_raw.lower()
    purposes = []

    if cat in CAFE_CATS:
        purposes = ['데이트', '혼자', '친구']
    elif cat in FOOD_CATS:
        purposes = ['데이트', '친구', '가족']
    elif cat in PARK_CATS:
        # 가족 체험형 공원
        if '가족' in kws or '어린이' in desc or '체험' in kws:
            purposes = ['가족', '활동', '힐링']
        else:
            purposes = ['힐링', '데이트', '활동']
    elif cat in SPORT_CATS:
        purposes = ['활동']
    elif cat in CULTURE_CATS:
        if '전통' in cat or '떡' in desc:
            purposes = ['가족', '친구']
        else:
            purposes = ['데이트', '친구']
    else:
        purposes = ['혼자', '친구']

    # 조용 → 혼자·힐링 보강 (카페·공원에서 특히 유효)
    if ('조용' in desc or '조용' in kws) and cat in CAFE_CATS:
        if '혼자' not in purposes:
            purposes.insert(1, '혼자')
        if '힐링' not in purposes:
            purposes.append('힐링')

    return ','.join(purposes[:3])


# ── description 강화 ──────────────────────────────────────────────────────────
_ATMO = {
    '조용': '조용한', '차분': '차분한', '여유': '여유로운',
    '힐링': '힐링 분위기의', '감성': '감성적인', '핫플': '활기찬',
    '활기': '활기찬', '사진': '사진 찍기 좋은', 'SNS': '감성적인',
    '체험': '체험 중심의', '문화': '문화적인', '전통': '전통 있는',
}

def _pick_atmo(kws, desc):
    for key, label in _ATMO.items():
        if key in kws or key in desc:
            return label
    return '편안한'

_SITUATION = {
    frozenset(['데이트','혼자','친구']): '혼자 방문하거나 연인·친구와',
    frozenset(['데이트','친구']): '친구나 연인과 함께',
    frozenset(['데이트','친구','가족']): '친구나 가족과 함께',
    frozenset(['가족','활동','힐링']): '가족과 함께',
    frozenset(['힐링','데이트','활동']): '연인이나 친구와',
    frozenset(['활동']): '운동을 즐기기 위해',
}

def _pick_situation(purpose_str):
    p = frozenset(purpose_str.split(','))
    return _SITUATION.get(p, '함께')

_ACTION = {
    '카페':        '커피와 디저트를 즐기며 대화하거나 여유롭게 쉬기',
    '디저트카페':  '디저트와 커피를 즐기며 사진을 찍거나 대화하기',
    '고양이카페':  '고양이와 교감하며 힐링하기',
    '공원':        '산책하거나 자연 속에서 휴식 취하기',
    '스포츠':      '다양한 운동을 즐기며 활동하기',
    '헬스':        '운동 기구를 이용해 체력 관리하기',
    '테니스':      '테니스를 즐기며 활동하기',
    '볼링':        '볼링을 즐기며 신나는 시간 보내기',
    '체육관':      '체육 시설을 이용해 운동하기',
    '스포츠센터':  '실내 운동 시설을 활용해 활동하기',
    '당구장':      '당구를 즐기며 여가 시간 보내기',
    '공연장':      '공연과 전시를 감상하며 문화를 누리기',
    '문화시설':    '다양한 전시와 체험 활동 즐기기',
    '문화센터':    '문화 프로그램에 참여하거나 체험하기',
    '테마거리':    '거리를 거닐며 상점과 분위기를 즐기기',
    '전통':        '전통 체험을 통해 문화를 배우고 즐기기',
}

_FOOD_ACTION = {
    '고가': '맛있는 식사를 즐기며 특별한 시간 보내기',
    '중가': '풍성한 식사를 즐기며 대화하기',
    '저가': '가성비 좋은 식사를 편하게 즐기기',
}

def enhance_desc(row, purpose_str):
    cat   = row['카테고리']
    desc  = row['설명']
    kws   = row['키워드']
    price = row.get('price_level','중가')

    atmo      = _pick_atmo(kws, desc)
    situation = _pick_situation(purpose_str)

    if cat in FOOD_CATS:
        action = _FOOD_ACTION.get(price, '식사를 즐기며 대화하기')
    else:
        action = _ACTION.get(cat, '여유로운 시간 보내기')

    # 원본 설명에서 핵심 정보(장소 특성) 보존
    core = desc.rstrip('.')
    return f"{situation} 방문하기 좋은 {atmo} 곳으로, {action}에 적합하다. {core}."


# ── keywords 확장 ─────────────────────────────────────────────────────────────
_PURPOSE_KW = {
    '데이트': '데이트', '혼자': '혼자', '친구': '친구',
    '가족': '가족', '활동': '활동', '힐링': '힐링',
}

_CAT_EXTRA = {
    '카페':        ['커피','여유','대화'],
    '디저트카페':  ['디저트','사진','대화'],
    '고양이카페':  ['체험','감성','힐링'],
    '공원':        ['산책','자연','휴식'],
    '스포츠':      ['운동','활동'],
    '헬스':        ['운동','건강'],
    '볼링':        ['체험','실내'],
    '당구장':      ['실내','체험'],
    '공연장':      ['공연','문화'],
    '문화시설':    ['전시','체험'],
    '테마거리':    ['거리','쇼핑'],
    '전통':        ['체험','문화'],
}

_ATMO_KW_MAP = {
    '조용': '조용한', '여유': '여유', '감성': '감성', '힐링': '힐링',
    '사진': '사진', '활기': '활기',
}

def expand_kws(row, purpose_str):
    cat      = row['카테고리']
    kws_raw  = row['키워드']
    desc     = row['설명']
    existing = [k.strip() for k in kws_raw.split(',') if k.strip()]

    additions = []

    # 1. purpose 기반
    for p in purpose_str.split(','):
        additions.append(p)

    # 2. 분위기 기반
    for key, kw in _ATMO_KW_MAP.items():
        if key in kws_raw or key in desc:
            additions.append(kw)

    # 3. 카테고리 기반 행동/추가 키워드
    for kw in _CAT_EXTRA.get(cat, []):
        additions.append(kw)

    # 4. 음식 카테고리 공통
    if cat in FOOD_CATS:
        additions += ['식사', '맛집']

    # 중복 제거 (기존 순서 우선) + 최대 8개
    seen  = set(existing)
    final = existing[:]
    for kw in additions:
        if kw not in seen and len(final) < 8:
            seen.add(kw)
            final.append(kw)

    return ','.join(final[:8])


# ── 메인 처리 ─────────────────────────────────────────────────────────────────
results = []

with open('data/places_extended.csv', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        cat   = row['카테고리']
        desc  = row['설명']
        kws   = row['키워드']

        purpose  = get_purpose(cat, desc, kws)
        new_desc = enhance_desc(row, purpose)
        new_kws  = expand_kws(row, purpose)

        results.append({
            "name":        row['장소명'],
            "purpose":     purpose,
            "description": new_desc,
            "keywords":    new_kws,
        })

print(json.dumps(results, ensure_ascii=False, indent=2))
