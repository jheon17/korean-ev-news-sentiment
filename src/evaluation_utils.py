import csv
from collections import Counter
from pathlib import Path


LABELS = ["긍정", "중립", "부정"]


def read_csv(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def save_csv(csv_path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def calculate_metrics(rows: list[dict], split: str = "dev") -> dict:
    selected_rows = rows if split == "all" else [row for row in rows if row.get("split") == split]

    valid_rows = [
        row
        for row in selected_rows
        if row.get("gold_label") in LABELS
        and row.get("pred_label") in LABELS
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
    failed_count = sum(1 for row in selected_rows if row.get("pred_label") not in LABELS)
    accuracy = correct_count / len(valid_rows) if valid_rows else 0
    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0
    prediction_counter = Counter(row["pred_label"] for row in valid_rows)

    return {
        "split": split,
        "evaluated_count": len(valid_rows),
        "correct_count": correct_count,
        "failed_count": failed_count,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "class_metrics": class_metrics,
        "confusion": confusion,
        "prediction_counter": prediction_counter,
    }


def save_report(report_path: Path, title: str, metrics: dict, extra_lines: list[str] | None = None) -> None:
    lines = [
        title,
        "",
        f"평가 범위: {metrics['split']} 데이터만 사용",
        f"평가 기사 수: {metrics['evaluated_count']}",
        f"정답 수: {metrics['correct_count']}",
        f"분류 실패 수: {metrics['failed_count']}",
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

    if extra_lines:
        lines.extend(["", *extra_lines])

    report_path.write_text("\n".join(lines), encoding="utf-8")
