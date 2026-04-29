"""
agents/multi_agent/test_multiagent.py

Multi Agent 시스템 테스트 스크립트
"""

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from orchestrator import run_orchestrator  # type: ignore

CASES = [
    # (label, query, region)
    ("TC1_course_normal",   "노원 데이트 코스 추천해줘",           "노원"),
    ("TC2_course_indoor",   "비 오는 날 실내 데이트 코스 추천해줘", "노원"),
    ("TC3_single_cafe",     "조용한 카페 추천해줘",                "노원"),
    ("TC4_fallback_desert", "사막에서 카페 추천해줘",              "사막"),
    ("TC5_fallback_weird",  "화성에서 코스 추천해줘",              "화성"),
]

SEP = "=" * 70

for label, query, region in CASES:
    print(f"\n{SEP}")
    print(f"[TEST] {label}")
    print(f"  query  : {query}")
    print(f"  region : {region}")
    print(SEP)

    result = run_orchestrator(query, region=region)
    meta   = result["metadata"]

    print("\n--- 결과 ---")
    print(f"  agents_used     : {meta['agents_used']}")
    print(f"  trace           : {meta['trace']}")
    print(f"  validation      : {meta['validation']}")
    print(f"  region_mismatch : {meta['region_mismatch']}")
    print(f"  complete        : {result['complete']}")
    print(f"  reply       :\n    {result['reply'][:200]}")

    if result["course"]:
        for i, step in enumerate(result["course"], 1):
            print(f"  코스{i}: {' → '.join(p['name'] for p in step)}")
    elif result["places"]:
        for p in result["places"]:
            print(f"  장소: {p['name']} ({p['category']})")

print(f"\n{SEP}")
print("[DONE] 모든 테스트 케이스 완료")
print(SEP)
