import csv
import html
import re
from collections import Counter, defaultdict
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path


INPUT_FILENAME = "naver_news_sample_20260803.csv"
OUTPUT_FILENAME = "naver_news_sample_20260803_preprocessed.csv"
REPORT_FILENAME = "preprocessing_report_20260803.txt"


def clean_html_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_pub_date(pub_date: str) -> str:
    if not pub_date:
        return ""

    try:
        parsed = parsedate_to_datetime(pub_date)
    except (TypeError, ValueError):
        return ""

    return parsed.isoformat()


def normalize_title_for_check(title: str) -> str:
    title = clean_html_text(title)
    title = title.lower()
    title = re.sub(r"[^0-9a-z가-힣]+", "", title)
    return title


def make_dedup_key(row: dict) -> str:
    originallink = (row.get("originallink") or "").strip()
    link = (row.get("link") or "").strip()
    return originallink or link


def read_csv(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def save_csv(csv_path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "search_query",
        "matched_queries",
        "duplicate_count",
        "collected_at",
        "title",
        "description",
        "originallink",
        "link",
        "pubDate",
        "pubDate_iso",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def preprocess_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    grouped_by_link = defaultdict(list)

    for row in rows:
        grouped_by_link[make_dedup_key(row)].append(row)

    processed_rows = []

    for dedup_key, duplicate_rows in grouped_by_link.items():
        base_row = duplicate_rows[0]
        matched_queries = sorted({row.get("search_query", "") for row in duplicate_rows})

        processed_rows.append(
            {
                "search_query": base_row.get("search_query", ""),
                "matched_queries": ", ".join(matched_queries),
                "duplicate_count": len(duplicate_rows),
                "collected_at": base_row.get("collected_at", ""),
                "title": clean_html_text(base_row.get("title", "")),
                "description": clean_html_text(base_row.get("description", "")),
                "originallink": base_row.get("originallink", ""),
                "link": base_row.get("link", ""),
                "pubDate": base_row.get("pubDate", ""),
                "pubDate_iso": parse_pub_date(base_row.get("pubDate", "")),
            }
        )

    title_counter = Counter(normalize_title_for_check(row.get("title", "")) for row in processed_rows)
    repeated_titles = {
        title: count
        for title, count in title_counter.items()
        if title and count > 1
    }

    search_query_counter = Counter(row.get("search_query", "") for row in rows)
    duplicate_link_count = sum(len(duplicates) - 1 for duplicates in grouped_by_link.values())

    report = {
        "raw_count": len(rows),
        "processed_count": len(processed_rows),
        "duplicate_link_count": duplicate_link_count,
        "missing_title_count": sum(1 for row in processed_rows if not row.get("title")),
        "missing_description_count": sum(1 for row in processed_rows if not row.get("description")),
        "invalid_pub_date_count": sum(1 for row in processed_rows if not row.get("pubDate_iso")),
        "search_query_counter": search_query_counter,
        "repeated_title_count": len(repeated_titles),
        "repeated_titles": repeated_titles,
    }

    return processed_rows, report


def save_report(report_path: Path, report: dict) -> None:
    lines = [
        "4단계 전처리 확인 리포트",
        f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"원본 행 수: {report['raw_count']}",
        f"전처리 후 행 수: {report['processed_count']}",
        f"원문 링크 기준 제거된 중복 행 수: {report['duplicate_link_count']}",
        f"결측 제목 수: {report['missing_title_count']}",
        f"결측 요약문 수: {report['missing_description_count']}",
        f"날짜 변환 실패 수: {report['invalid_pub_date_count']}",
        "",
        "검색어별 원본 수집 행 수:",
    ]

    for query, count in report["search_query_counter"].items():
        lines.append(f"- {query}: {count}")

    lines.append("")
    lines.append(f"정규화 후 동일한 제목 종류 수: {report['repeated_title_count']}")

    if report["repeated_titles"]:
        lines.append("정규화 후 동일한 제목 예시:")
        for title, count in list(report["repeated_titles"].items())[:10]:
            lines.append(f"- {title}: {count}")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_path = project_root / "data" / "raw" / INPUT_FILENAME
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        print(f"원본 CSV 파일을 찾을 수 없습니다: {raw_path}")
        return

    rows = read_csv(raw_path)
    processed_rows, report = preprocess_rows(rows)

    output_path = processed_dir / OUTPUT_FILENAME
    report_path = processed_dir / REPORT_FILENAME

    save_csv(output_path, processed_rows)
    save_report(report_path, report)

    print("뉴스 전처리 완료")
    print(f"원본 행 수: {report['raw_count']}")
    print(f"전처리 후 행 수: {report['processed_count']}")
    print(f"원문 링크 기준 제거된 중복 행 수: {report['duplicate_link_count']}")
    print(f"결측 제목 수: {report['missing_title_count']}")
    print(f"결측 요약문 수: {report['missing_description_count']}")
    print(f"날짜 변환 실패 수: {report['invalid_pub_date_count']}")
    print(f"전처리 CSV 파일: {output_path}")
    print(f"전처리 리포트 파일: {report_path}")


if __name__ == "__main__":
    main()
