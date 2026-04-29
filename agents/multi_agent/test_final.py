"""
agents/multi_agent/test_final.py

최종 종합 검증 스크립트 — 8개 시나리오
"""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from orchestrator import run_orchestrator  # type: ignore

CASES = [
    # (label, query, region)
    ("TC1_basic_course",  "노원 데이트 코스 추천해줘",                   "노원"),
    ("TC2a_conditional",  "비 오는 날 실내 데이트 코스 추천해줘",         "노원"),
    ("TC2b_single_cafe",  "조용한 카페 추천해줘",                        "노원"),
    ("TC3_multistep",     "카페 갔다가 식사하고 산책까지 할 수 있는 코스", "노원"),
    ("TC4a_desert",       "사막에서 카페 추천해줘",                       "사막"),
    ("TC4b_vague",        "아무거나 추천해줘",                            "노원"),
    ("TC5a_edge_short",   "추천",                                        "노원"),
    ("TC5b_edge_keyword", "카페",                                        "노원"),
]

SEP  = "=" * 70
SEP2 = "-" * 70

for label, query, region in CASES:
    print(f"\n{SEP}")
    print(f"[TEST] {label}")
    print(f"  query  : {query!r}")
    print(f"  region : {region}")
    print(SEP)

    try:
        result = run_orchestrator(query, region=region)
        meta   = result["metadata"]

        print(f"\n{SEP2}")
        print(f"  agents_used     : {meta['agents_used']}")
        print(f"  trace           : {meta['trace']}")
        print(f"  validation      : {meta['validation']}")
        print(f"  region_mismatch : {meta['region_mismatch']}")
        print(f"  complete        : {result['complete']}")
        print(f"  reply           :\n    {result['reply'][:300]}")

        if result["course"]:
            for i, step in enumerate(result["course"], 1):
                print(f"  코스{i}: {' → '.join(p['name'] for p in step)}")
        elif result["places"]:
            for p in result["places"]:
                print(f"  장소: {p['name']} ({p['category']})")
        print("  [OK] 예외 없이 완료")
    except Exception as e:
        print(f"  [ERROR] {type(e).__name__}: {e}")

print(f"\n{SEP}")
print("[DONE] 모든 테스트 케이스 완료")
print(SEP)
