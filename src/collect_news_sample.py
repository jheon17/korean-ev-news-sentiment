import csv
import json
import re
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError

from check_naver_api import load_env_file, search_news


SEARCH_QUERIES = [
    "전기차",
    "전기차 보조금",
    "전기차 판매",
    "전기차 충전",
    "배터리 화재",
    "전기차 가격",
]

DISPLAY_COUNT = 10
SORT = "date"


def make_safe_filename(text: str) -> str:
    safe_text = re.sub(r"[^0-9A-Za-z가-힣]+", "_", text).strip("_")
    return safe_text.lower()


def load_or_fetch_raw_response(raw_path: Path, query: str) -> dict:
    if raw_path.exists():
        return json.loads(raw_path.read_text(encoding="utf-8"))

    data = search_news(query=query, display=DISPLAY_COUNT, start=1, sort=SORT)
    raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def build_rows(data: dict, query: str, collected_at: str) -> list[dict]:
    rows = []

    for item in data.get("items", []):
        rows.append(
            {
                "search_query": query,
                "collected_at": collected_at,
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "originallink": item.get("originallink", ""),
                "link": item.get("link", ""),
                "pubDate": item.get("pubDate", ""),
            }
        )

    return rows


def save_csv(csv_path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "search_query",
        "collected_at",
        "title",
        "description",
        "originallink",
        "link",
        "pubDate",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    load_env_file(project_root / ".env")

    collected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    collection_date = datetime.now().strftime("%Y%m%d")

    all_rows = []
    print("뉴스 샘플 수집 시작")
    print(f"검색어 수: {len(SEARCH_QUERIES)}")
    print(f"검색어별 요청 건수: {DISPLAY_COUNT}")
    print(f"정렬 방식: {SORT}")
    print()

    try:
        for index, query in enumerate(SEARCH_QUERIES, start=1):
            query_filename = make_safe_filename(query)
            raw_path = raw_dir / f"naver_news_{collection_date}_{index:02d}_{query_filename}.json"

            data = load_or_fetch_raw_response(raw_path, query)
            rows = build_rows(data, query, collected_at)
            all_rows.extend(rows)

            print(f"- {query}: {len(rows)}건, 원본 파일: {raw_path.name}")

    except RuntimeError as error:
        print("환경변수 설정 오류")
        print(error)
        return
    except HTTPError as error:
        print("API 요청 실패")
        print(f"상태 코드: {error.code}")
        print(error.read().decode("utf-8", errors="replace"))
        return
    except URLError as error:
        print("네트워크 연결 오류")
        print(error.reason)
        return

    csv_path = raw_dir / f"naver_news_sample_{collection_date}.csv"
    save_csv(csv_path, all_rows)

    print()
    print("뉴스 샘플 수집 완료")
    print(f"전체 저장 행 수: {len(all_rows)}")
    print(f"CSV 파일: {csv_path}")


if __name__ == "__main__":
    main()
