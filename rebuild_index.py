"""
FAISS 인덱스 재빌드 스크립트

CSV 기반 구인덱스를 삭제하고 places_enriched.json으로 재생성한다.
검증 쿼리 3개로 semantic 검색 결과를 출력한다.

사용법:
    python rebuild_index.py                         # OPENAI_API_KEY 환경변수 사용
    python rebuild_index.py --api-key sk-...        # 직접 지정
    python rebuild_index.py --mock                  # FakeEmbeddings (구조 검증용, 의미 검색 불가)
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE = Path(__file__).parent
INDEX_DIR        = BASE / "index"
PLACES_INDEX_DIR = INDEX_DIR / "places"
IDS_FILE         = INDEX_DIR / "indexed_ids.json"

_TEST_QUERIES = [
    "비 오는 날 실내 카페",
    "혼자 조용히 갈 곳",
    "데이트 장소 추천",
]


# ── 인수 파싱 ─────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""), help="OpenAI API key")
    p.add_argument("--mock", action="store_true", help="FakeEmbeddings 사용 (구조 검증용)")
    return p.parse_args()


# ── 구인덱스 삭제 ─────────────────────────────────────────────────────────────
def purge_old_index() -> None:
    removed = []
    if PLACES_INDEX_DIR.exists():
        shutil.rmtree(PLACES_INDEX_DIR)
        removed.append(str(PLACES_INDEX_DIR))
    if IDS_FILE.exists():
        IDS_FILE.unlink()
        removed.append(str(IDS_FILE))

    if removed:
        for p in removed:
            print(f"  [삭제] {p}")
    else:
        print("  (기존 인덱스 없음)")


# ── 임베딩 모델 ───────────────────────────────────────────────────────────────
def make_embeddings(api_key: str, mock: bool):
    if mock:
        from langchain_community.embeddings import FakeEmbeddings
        print("  [경고] FakeEmbeddings 사용 — 의미 검색 결과는 무의미함 (구조 검증 전용)")
        return FakeEmbeddings(size=1536)

    if not api_key:
        sys.exit(
            "OPENAI_API_KEY가 없습니다.\n"
            "  환경변수 설정:  export OPENAI_API_KEY=sk-...\n"
            "  또는 인수 사용: python rebuild_index.py --api-key sk-..."
        )
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(openai_api_key=api_key)


# ── 인덱스 빌드 ───────────────────────────────────────────────────────────────
def build_index(embeddings) -> "FAISS":
    import json
    from langchain_community.vectorstores import FAISS

    sys.path.insert(0, str(BASE))
    from rag.loader import build_place_documents

    docs = build_place_documents()
    print(f"  로드된 문서 수: {len(docs)}")

    if not docs:
        sys.exit("[오류] places_enriched.json에서 문서를 로드하지 못했습니다.")

    # page_content 기반 임베딩 (metadata 자동 보존)
    print("  임베딩 중... (OpenAI API 호출)")
    vectorstore = FAISS.from_documents(docs, embeddings)

    PLACES_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(PLACES_INDEX_DIR))

    # indexed_ids 저장
    ids = sorted({int(d.metadata["id"]) for d in docs})
    IDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(ids, f)

    print(f"  저장 완료: {PLACES_INDEX_DIR}")
    print(f"  인덱싱된 ID 수: {len(ids)}")
    return vectorstore


# ── 검증 쿼리 ─────────────────────────────────────────────────────────────────
def validate(vectorstore, mock: bool) -> None:
    if mock:
        print("\n  [FakeEmbeddings] 검증 쿼리 결과는 무의미합니다 — 구조 정상 여부만 확인")

    for query in _TEST_QUERIES:
        print(f"\n  쿼리: \"{query}\"")
        results = vectorstore.similarity_search_with_score(query, k=3)
        for rank, (doc, score) in enumerate(results, 1):
            meta = doc.metadata
            purpose_str = "/".join(meta.get("purpose", []))
            mood_str    = "/".join(meta.get("mood", []))
            indoor      = meta.get("indoor_outdoor", "-")
            weather     = meta.get("weather_fit", "-")
            print(
                f"    {rank}. [{score:.4f}] {meta.get('장소명', '?')} "
                f"({meta.get('카테고리', '?')}) | {purpose_str} | {mood_str} | "
                f"실내외:{indoor} | 날씨:{weather}"
            )


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  FAISS 인덱스 재빌드 — places_enriched.json 기반")
    print("=" * 60)

    print("\n[1] 구인덱스 삭제")
    purge_old_index()

    print("\n[2] 임베딩 모델 준비")
    embeddings = make_embeddings(args.api_key, args.mock)

    print("\n[3] 인덱스 빌드")
    vectorstore = build_index(embeddings)

    print("\n[4] 검증 쿼리")
    validate(vectorstore, args.mock)

    print("\n" + "=" * 60)
    print("  완료. app.py 재시작 시 새 인덱스가 로드됩니다.")
    print("=" * 60)


if __name__ == "__main__":
    main()
