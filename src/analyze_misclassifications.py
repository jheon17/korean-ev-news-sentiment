import csv
from collections import Counter
from pathlib import Path


PREDICTION_FILENAME = "openai_gpt_predictions_20260803.csv"
REPORT_FILENAME = "error_analysis_openai_gpt_20260803.md"
CSV_FILENAME = "error_analysis_openai_gpt_20260803.csv"


def read_csv(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def save_csv(csv_path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "article_id",
        "gold_label",
        "pred_label",
        "error_type",
        "label_review_note",
        "title",
        "model_reason",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def classify_error_type(row: dict) -> str:
    gold_label = row.get("gold_label", "")
    pred_label = row.get("pred_label", "")
    text = f"{row.get('title', '')} {row.get('description', '')}"

    if pred_label == "분류실패":
        return "분류 실패"

    if gold_label == "긍정" and pred_label == "중립":
        if any(keyword in text for keyword in ["안전", "화재대응", "열폭주", "배터리", "충전", "지원", "확대"]):
            return "긍정 개선 요소를 중립으로 과소평가"
        if any(keyword in text for keyword in ["유럽", "미국", "해외"]):
            return "국내 기준과 해외 시장 영향 혼재"
        return "긍정 신호의 시장 영향 범위 판단 차이"

    if gold_label == "부정" and pred_label == "중립":
        return "부정 위험을 중립으로 과소평가"

    if gold_label == "중립" and pred_label in {"긍정", "부정"}:
        return "중립 기사를 방향성 있게 과대평가"

    return "기타 오분류"


def make_label_review_note(row: dict, error_type: str) -> str:
    if error_type == "국내 기준과 해외 시장 영향 혼재":
        return "국내 시장 기준을 엄격히 적용하면 중립일 수 있으므로 gold_label 재검토 후보"

    if error_type == "긍정 개선 요소를 중립으로 과소평가":
        return "안전·충전·지원 개선을 간접 긍정으로 볼지 라벨 가이드에 명확히 적을 필요가 있음"

    if error_type == "부정 위험을 중립으로 과소평가":
        return "비용·안전·정책 위험은 소비자 신뢰 하락 가능성을 더 강하게 반영할지 검토"

    if error_type == "중립 기사를 방향성 있게 과대평가":
        return "특정 기업·사실 전달 뉴스의 시장 전체 영향 기준을 더 엄격히 유지할지 검토"

    return "dev 데이터 기준으로 라벨 근거와 프롬프트 기준을 함께 검토"


def build_error_rows(rows: list[dict]) -> list[dict]:
    error_rows = []

    for row in rows:
        if row.get("split") != "dev":
            continue

        if row.get("gold_label") == row.get("pred_label"):
            continue

        error_type = classify_error_type(row)
        error_rows.append(
            {
                "article_id": row.get("article_id", ""),
                "gold_label": row.get("gold_label", ""),
                "pred_label": row.get("pred_label", ""),
                "error_type": error_type,
                "label_review_note": make_label_review_note(row, error_type),
                "title": row.get("title", ""),
                "model_reason": row.get("reason", ""),
            }
        )

    return error_rows


def save_report(report_path: Path, error_rows: list[dict]) -> None:
    counter = Counter(row["error_type"] for row in error_rows)
    lines = [
        "# 9단계 GPT API 오분류 분석",
        "",
        "평가 범위는 dev 데이터만 사용했다.",
        "최종 테스트 데이터는 오류 분석과 프롬프트 수정에 사용하지 않았다.",
        "",
        f"오분류 건수: {len(error_rows)}",
        "",
        "## 오류 유형별 건수",
        "",
        "| 오류 유형 | 건수 |",
        "| --- | ---: |",
    ]

    for error_type, count in counter.most_common():
        lines.append(f"| {error_type} | {count} |")

    lines.extend(
        [
            "",
            "## 주요 관찰",
            "",
            "- GPT API 모델은 긍정 라벨을 중립으로 판단하는 오류가 많았다.",
            "- 안전, 충전, 지원 확대처럼 간접적으로 시장 환경을 개선하는 뉴스를 어느 정도까지 긍정으로 볼지 기준을 더 분명히 해야 한다.",
            "- 해외 판매, 해외 공장, 특정 기업 실적 뉴스는 국내 시장 기준과 충돌할 수 있으므로 gold_label 재검토 후보로 둔다.",
            "",
            "## 프롬프트 개선 방향",
            "",
            "- 직접적인 등록대수 증가가 없어도 구매·이용 조건 개선이면 긍정 후보로 보도록 명시한다.",
            "- 해외 시장 중심 뉴스는 국내 시장 영향이 제목·요약문에 드러나지 않으면 중립으로 보도록 명시한다.",
            "- 연구 초기 단계 기술 뉴스는 상용화나 국내 적용 근거가 없으면 중립으로 보도록 명시한다.",
            "",
            "## 오분류 목록",
            "",
            "| article_id | gold_label | pred_label | 오류 유형 | 제목 |",
            "| --- | --- | --- | --- | --- |",
        ]
    )

    for row in error_rows:
        title = row["title"].replace("|", " ")
        lines.append(
            f"| {row['article_id']} | {row['gold_label']} | {row['pred_label']} | "
            f"{row['error_type']} | {title} |"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    prediction_path = project_root / "data" / "predictions" / PREDICTION_FILENAME
    report_dir = project_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    if not prediction_path.exists():
        print(f"예측 CSV 파일을 찾을 수 없습니다: {prediction_path}")
        return

    rows = read_csv(prediction_path)
    error_rows = build_error_rows(rows)

    csv_path = report_dir / CSV_FILENAME
    report_path = report_dir / REPORT_FILENAME
    save_csv(csv_path, error_rows)
    save_report(report_path, error_rows)

    print("GPT API 오분류 분석 완료")
    print(f"오분류 건수: {len(error_rows)}")
    print(f"오분류 CSV 파일: {csv_path}")
    print(f"오분류 리포트 파일: {report_path}")


if __name__ == "__main__":
    main()
