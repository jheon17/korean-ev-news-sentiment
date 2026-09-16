import argparse
from pathlib import Path

from evaluation_utils import LABELS, calculate_metrics, read_csv, save_csv, save_report


LABEL_FILENAME = "news_labeling_sample_20260803.csv"
PREDICTION_FILENAME = "local_oss_predictions_20260803.csv"
REPORT_FILENAME = "local_oss_report_20260803.txt"
DEFAULT_MODEL = "joeddav/xlm-roberta-large-xnli"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="로컬 오픈소스 모델로 전기차 뉴스 감성을 분류한다.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face 모델 이름")
    parser.add_argument("--split", default="dev", choices=["dev", "final_test", "all"], help="분류할 데이터 범위")
    parser.add_argument("--limit", type=int, default=0, help="테스트용 최대 처리 건수. 0이면 전체 처리")
    return parser


def make_input_text(row: dict) -> str:
    return (
        "다음 뉴스가 국내 전기차 시장 전체의 수요, 보급, 등록대수에 미칠 영향을 분류하세요.\n"
        f"제목: {row.get('title', '')}\n"
        f"요약문: {row.get('description', '')}"
    )


def select_rows(rows: list[dict], split: str, limit: int) -> list[dict]:
    if split == "all":
        selected = rows
    else:
        selected = [row for row in rows if row.get("split") == split]

    if limit > 0:
        return selected[:limit]

    return selected


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    label_path = project_root / "data" / "labels" / LABEL_FILENAME
    prediction_dir = project_root / "data" / "predictions"
    report_dir = project_root / "reports"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    if not label_path.exists():
        print(f"라벨링 CSV 파일을 찾을 수 없습니다: {label_path}")
        return

    try:
        from transformers import pipeline
    except ImportError:
        print("transformers 패키지가 설치되어 있지 않습니다.")
        print("먼저 다음 명령을 실행하세요.")
        print("uv sync --extra local-models")
        return

    rows = read_csv(label_path)
    selected_rows = select_rows(rows, args.split, args.limit)

    print("로컬 오픈소스 모델 분류 시작")
    print(f"모델: {args.model}")
    print(f"분류 대상: {args.split}")
    print(f"처리 건수: {len(selected_rows)}")
    print("처음 실행하면 모델 다운로드 때문에 시간이 오래 걸릴 수 있습니다.")

    classifier = pipeline("zero-shot-classification", model=args.model)
    prediction_rows = []

    for index, row in enumerate(selected_rows, start=1):
        result = classifier(
            make_input_text(row),
            candidate_labels=LABELS,
            hypothesis_template="이 뉴스가 국내 전기차 시장 전체에 미치는 영향은 {}이다.",
            multi_label=False,
        )

        prediction_rows.append(
            {
                "article_id": row.get("article_id", ""),
                "split": row.get("split", ""),
                "gold_label": row.get("gold_label", ""),
                "pred_label": result["labels"][0],
                "confidence": f"{result['scores'][0]:.4f}",
                "model": args.model,
                "title": row.get("title", ""),
                "description": row.get("description", ""),
            }
        )

        print(f"- {index}/{len(selected_rows)} {row.get('article_id', '')}: {result['labels'][0]}")

    prediction_path = prediction_dir / PREDICTION_FILENAME
    report_path = report_dir / REPORT_FILENAME
    save_csv(
        prediction_path,
        prediction_rows,
        ["article_id", "split", "gold_label", "pred_label", "confidence", "model", "title", "description"],
    )

    metrics = calculate_metrics(prediction_rows, split=args.split)
    save_report(
        report_path,
        "7단계 로컬 오픈소스 모델 평가 리포트",
        metrics,
        [f"모델: {args.model}", "방식: zero-shot-classification"],
    )

    print()
    print("로컬 오픈소스 모델 분류 완료")
    print(f"예측 CSV 파일: {prediction_path}")
    print(f"평가 리포트 파일: {report_path}")


if __name__ == "__main__":
    main()
