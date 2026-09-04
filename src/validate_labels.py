import csv
from collections import Counter
from pathlib import Path


LABEL_FILENAME = "news_labeling_sample_20260803.csv"
ALLOWED_LABELS = {"긍정", "중립", "부정"}


def read_csv(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    label_path = project_root / "data" / "labels" / LABEL_FILENAME

    if not label_path.exists():
        print(f"라벨링 CSV 파일을 찾을 수 없습니다: {label_path}")
        return

    rows = read_csv(label_path)
    invalid_rows = []
    empty_label_rows = []
    empty_reason_rows = []
    split_counter = Counter(row.get("split", "") for row in rows)
    dev_label_counter = Counter()

    for row in rows:
        article_id = row.get("article_id", "")
        split = row.get("split", "")
        label = row.get("gold_label", "").strip()
        reason = row.get("label_reason", "").strip()

        if not label:
            empty_label_rows.append(article_id)
        elif label not in ALLOWED_LABELS:
            invalid_rows.append((article_id, label))

        if not reason:
            empty_reason_rows.append(article_id)

        if split == "dev" and label in ALLOWED_LABELS:
            dev_label_counter[label] += 1

    print("라벨링 데이터 검증 결과")
    print(f"전체 행 수: {len(rows)}")
    print(f"개발용(dev) 행 수: {split_counter.get('dev', 0)}")
    print(f"최종 테스트용(final_test) 행 수: {split_counter.get('final_test', 0)}")
    print(f"빈 gold_label 수: {len(empty_label_rows)}")
    print(f"허용되지 않은 gold_label 수: {len(invalid_rows)}")
    print(f"빈 label_reason 수: {len(empty_reason_rows)}")
    print()

    print("개발용(dev) 라벨 분포")
    for label in ["긍정", "중립", "부정"]:
        print(f"- {label}: {dev_label_counter.get(label, 0)}")

    if empty_label_rows:
        print()
        print("gold_label이 비어 있는 article_id")
        print(", ".join(empty_label_rows))

    if invalid_rows:
        print()
        print("허용되지 않은 gold_label")
        for article_id, label in invalid_rows:
            print(f"- {article_id}: {label}")


if __name__ == "__main__":
    main()
