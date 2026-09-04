import csv
from pathlib import Path


INPUT_FILENAME = "naver_news_sample_20260803_preprocessed.csv"
OUTPUT_FILENAME = "news_labeling_sample_20260803.csv"
TEST_RATIO = 0.2


def read_csv(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def save_csv(csv_path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "article_id",
        "split",
        "gold_label",
        "label_reason",
        "ambiguous_note",
        "search_query",
        "matched_queries",
        "duplicate_count",
        "title",
        "description",
        "originallink",
        "link",
        "pubDate_iso",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def assign_split(index: int, total_count: int) -> str:
    test_count = max(1, round(total_count * TEST_RATIO))
    return "final_test" if index > total_count - test_count else "dev"


def build_labeling_rows(rows: list[dict]) -> list[dict]:
    total_count = len(rows)
    labeling_rows = []

    for index, row in enumerate(rows, start=1):
        labeling_rows.append(
            {
                "article_id": f"NEWS-{index:04d}",
                "split": assign_split(index, total_count),
                "gold_label": "",
                "label_reason": "",
                "ambiguous_note": "",
                "search_query": row.get("search_query", ""),
                "matched_queries": row.get("matched_queries", ""),
                "duplicate_count": row.get("duplicate_count", ""),
                "title": row.get("title", ""),
                "description": row.get("description", ""),
                "originallink": row.get("originallink", ""),
                "link": row.get("link", ""),
                "pubDate_iso": row.get("pubDate_iso", ""),
            }
        )

    return labeling_rows


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "processed" / INPUT_FILENAME
    output_dir = project_root / "data" / "labels"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"전처리 CSV 파일을 찾을 수 없습니다: {input_path}")
        return

    rows = read_csv(input_path)
    labeling_rows = build_labeling_rows(rows)

    output_path = output_dir / OUTPUT_FILENAME
    save_csv(output_path, labeling_rows)

    split_counts = {}
    for row in labeling_rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1

    print("라벨링 데이터셋 생성 완료")
    print(f"입력 기사 수: {len(rows)}")
    print(f"출력 기사 수: {len(labeling_rows)}")
    print(f"개발용(dev) 기사 수: {split_counts.get('dev', 0)}")
    print(f"최종 테스트용(final_test) 기사 수: {split_counts.get('final_test', 0)}")
    print(f"라벨링 CSV 파일: {output_path}")
    print()
    print("gold_label 열에는 긍정, 중립, 부정 중 하나를 직접 입력하세요.")


if __name__ == "__main__":
    main()
