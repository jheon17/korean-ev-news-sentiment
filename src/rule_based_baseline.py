import csv
import re
from collections import Counter
from pathlib import Path


LABEL_FILENAME = "news_labeling_sample_20260803.csv"
PREDICTION_FILENAME = "rule_based_predictions_20260803.csv"
REPORT_FILENAME = "rule_based_baseline_report_20260803.txt"
LABELS = ["긍정", "중립", "부정"]

POSITIVE_KEYWORDS = [
    "보조금 확대",
    "지원 확대",
    "보급 확대",
    "충전소 확대",
    "충전 인프라 확충",
    "인프라 확대",
    "가격 인하",
    "판매 증가",
    "수요 증가",
    "성장",
    "확대",
    "개선",
    "편의",
]

NEGATIVE_KEYWORDS = [
    "화재",
    "폭발",
    "리콜",
    "결함",
    "보조금 축소",
    "보조금 삭감",
    "지원 축소",
    "가격 인상",
    "판매 감소",
    "판매 부진",
    "수요 감소",
    "충전난",
    "충전 불편",
    "인프라 부족",
    "보험료",
    "수리비",
    "불안",
    "우려",
    "침체",
]


def read_csv(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def save_csv(csv_path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "article_id",
        "split",
        "gold_label",
        "pred_label",
        "positive_score",
        "negative_score",
        "matched_positive_keywords",
        "matched_negative_keywords",
        "title",
        "description",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_keywords(text: str, keywords: list[str]) -> list[str]:
    return [keyword for keyword in keywords if keyword in text]


def predict_label(title: str, description: str) -> dict:
    text = normalize_text(f"{title} {description}")
    matched_positive = find_keywords(text, POSITIVE_KEYWORDS)
    matched_negative = find_keywords(text, NEGATIVE_KEYWORDS)

    positive_score = len(matched_positive)
    negative_score = len(matched_negative)

    if positive_score > negative_score:
        pred_label = "긍정"
    elif negative_score > positive_score:
        pred_label = "부정"
    else:
        pred_label = "중립"

    return {
        "pred_label": pred_label,
        "positive_score": positive_score,
        "negative_score": negative_score,
        "matched_positive_keywords": ", ".join(matched_positive),
        "matched_negative_keywords": ", ".join(matched_negative),
    }


def calculate_metrics(rows: list[dict]) -> dict:
    valid_rows = [
        row
        for row in rows
        if row.get("split") == "dev" and row.get("gold_label") in LABELS
    ]

    confusion = {
        gold_label: {pred_label: 0 for pred_label in LABELS}
        for gold_label in LABELS
    }

    for row in valid_rows:
        confusion[row["gold_label"]][row["pred_label"]] += 1

    class_metrics = {}
    f1_values = []

    for label in LABELS:
        true_positive = confusion[label][label]
        false_positive = sum(confusion[other_label][label] for other_label in LABELS if other_label != label)
        false_negative = sum(confusion[label][other_label] for other_label in LABELS if other_label != label)

        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

        class_metrics[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1_values.append(f1)

    correct_count = sum(1 for row in valid_rows if row["gold_label"] == row["pred_label"])
    accuracy = correct_count / len(valid_rows) if valid_rows else 0
    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0
    prediction_counter = Counter(row["pred_label"] for row in valid_rows)

    return {
        "evaluated_count": len(valid_rows),
        "correct_count": correct_count,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "class_metrics": class_metrics,
        "confusion": confusion,
        "prediction_counter": prediction_counter,
    }


def save_report(report_path: Path, metrics: dict) -> None:
    lines = [
        "6단계 기준 모델 평가 리포트",
        "",
        "평가 범위: dev 데이터만 사용",
        f"평가 기사 수: {metrics['evaluated_count']}",
        f"정답 수: {metrics['correct_count']}",
        f"Accuracy: {metrics['accuracy']:.3f}",
        f"Macro-F1: {metrics['macro_f1']:.3f}",
        "",
        "클래스별 지표:",
    ]

    for label in LABELS:
        class_metric = metrics["class_metrics"][label]
        lines.append(
            f"- {label}: Precision {class_metric['precision']:.3f}, "
            f"Recall {class_metric['recall']:.3f}, F1 {class_metric['f1']:.3f}"
        )

    lines.extend(["", "혼동행렬:", "gold_label \\ pred_label,긍정,중립,부정"])
    for gold_label in LABELS:
        row = [gold_label] + [str(metrics["confusion"][gold_label][pred_label]) for pred_label in LABELS]
        lines.append(",".join(row))

    lines.extend(["", "예측 라벨 분포:"])
    for label in LABELS:
        lines.append(f"- {label}: {metrics['prediction_counter'].get(label, 0)}")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    label_path = project_root / "data" / "labels" / LABEL_FILENAME
    prediction_dir = project_root / "data" / "predictions"
    report_dir = project_root / "reports"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    if not label_path.exists():
        print(f"라벨링 CSV 파일을 찾을 수 없습니다: {label_path}")
        return

    rows = [
        row
        for row in read_csv(label_path)
        if row.get("split") == "dev"
    ]
    prediction_rows = []

    for row in rows:
        prediction = predict_label(row.get("title", ""), row.get("description", ""))
        prediction_rows.append(
            {
                "article_id": row.get("article_id", ""),
                "split": row.get("split", ""),
                "gold_label": row.get("gold_label", ""),
                "pred_label": prediction["pred_label"],
                "positive_score": prediction["positive_score"],
                "negative_score": prediction["negative_score"],
                "matched_positive_keywords": prediction["matched_positive_keywords"],
                "matched_negative_keywords": prediction["matched_negative_keywords"],
                "title": row.get("title", ""),
                "description": row.get("description", ""),
            }
        )

    metrics = calculate_metrics(prediction_rows)
    prediction_path = prediction_dir / PREDICTION_FILENAME
    report_path = report_dir / REPORT_FILENAME

    save_csv(prediction_path, prediction_rows)
    save_report(report_path, metrics)

    print("기준 모델 평가 완료")
    print("평가 범위: dev 데이터만 사용")
    print(f"평가 기사 수: {metrics['evaluated_count']}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro-F1: {metrics['macro_f1']:.3f}")
    print(f"예측 CSV 파일: {prediction_path}")
    print(f"평가 리포트 파일: {report_path}")


if __name__ == "__main__":
    main()
