import argparse
import json
import os
import time
from pathlib import Path

from check_naver_api import load_env_file
from evaluation_utils import LABELS, calculate_metrics, read_csv, save_csv, save_report


LABEL_FILENAME = "news_labeling_sample_20260803.csv"
PREDICTION_FILENAME = "openai_gpt_predictions_20260803.csv"
REPORT_FILENAME = "openai_gpt_report_20260803.txt"
DEFAULT_MODEL = "gpt-5.6-luna"
USD_TO_KRW = 1400
LUNA_INPUT_USD_PER_1M = 0.20
LUNA_OUTPUT_USD_PER_1M = 0.90
CONSERVATIVE_INPUT_TOKENS_PER_ARTICLE = 2000
CONSERVATIVE_OUTPUT_TOKENS_PER_ARTICLE = 300


JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": LABELS},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
        "evidence": {"type": "string"},
    },
    "required": ["label", "confidence", "reason", "evidence"],
    "additionalProperties": False,
}


SYSTEM_INSTRUCTIONS_V1 = """
너는 전기차 뉴스 감성 분류 보조자다.
감성은 문장의 일반적인 분위기가 아니라 국내 전기차 시장 전체의 수요, 보급, 등록대수에 미칠 예상 영향이다.

라벨 기준:
- 긍정: 수요, 보급, 등록 증가 가능성을 높이는 내용
- 중립: 사실 전달, 영향이 불명확하거나 긍정과 부정이 섞인 내용
- 부정: 수요, 보급, 등록 감소 또는 위험 증가 가능성을 높이는 내용

판단 근거가 부족하면 긍정이나 부정으로 억지 분류하지 말고 중립을 선택한다.
특정 기업에만 유리한 내용은 국내 시장 전체 영향이 명확할 때만 긍정으로 본다.
""".strip()


SYSTEM_INSTRUCTIONS_V2 = """
너는 전기차 뉴스 감성 분류 보조자다.
감성은 문장의 일반적인 분위기가 아니라 국내 전기차 시장 전체의 수요, 보급, 소비자 신뢰, 등록대수에 미칠 예상 영향이다.

라벨 기준:
- 긍정: 전기차 구매 장벽을 낮추거나, 선택지를 늘리거나, 충전·정비·안전·배터리 기술·고객 지원을 개선해 국내 전기차 수요·보급에 유리한 조건을 만드는 내용
- 중립: 단순 사실 전달, 특정 기업·해외시장·주가·행사 중심이라 국내 시장 전체 영향이 불명확한 내용, 긍정과 부정 요소가 비슷하게 섞인 내용
- 부정: 화재·결함·리콜·비용 상승·보조금 축소·충전 불편·판매 부진처럼 소비자 신뢰나 구매 의향을 낮출 가능성이 큰 내용

판단 원칙:
- 수치로 등록대수 증가가 직접 제시되지 않아도, 국내 소비자의 전기차 구매·이용 조건이 좋아지는 뉴스면 긍정을 고려한다.
- 안전 대응 강화, 배터리 안전성 개선, 충전 인프라 개선, 전기차 라인업 확대, 구매·정비 지원 강화는 긍정 후보로 본다.
- 해외 판매, 해외 공장, 특정 기업 주가, 연구 초기 단계 뉴스는 국내 시장 적용 가능성이 제목·요약문에 드러나지 않으면 중립으로 본다.
- 판단 근거가 부족하면 중립을 선택한다.
""".strip()


PROMPT_VERSIONS = {
    "v1": SYSTEM_INSTRUCTIONS_V1,
    "v2": SYSTEM_INSTRUCTIONS_V2,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenAI GPT API로 전기차 뉴스 감성을 분류한다.")
    parser.add_argument("--run", action="store_true", help="실제 API 호출을 실행한다. 비용이 발생할 수 있다.")
    parser.add_argument("--model", default="", help="사용할 OpenAI 모델. 비우면 OPENAI_MODEL 또는 기본값을 사용한다.")
    parser.add_argument("--split", default="dev", choices=["dev", "final_test", "all"], help="분류할 데이터 범위")
    parser.add_argument("--limit", type=int, default=3, help="최대 처리 건수. 비용 확인 전에는 작은 값 권장")
    parser.add_argument("--max-krw", type=int, default=10000, help="실행 전 보수적 추정 비용 한도")
    parser.add_argument("--request-delay", type=float, default=0, help="요청 사이에 기다릴 초")
    parser.add_argument("--max-retries", type=int, default=2, help="rate limit 오류 재시도 횟수")
    parser.add_argument("--retry-failed-only", action="store_true", help="기존 예측 파일에서 분류실패 행만 다시 실행한다.")
    parser.add_argument("--prompt-version", default="v1", choices=sorted(PROMPT_VERSIONS), help="사용할 프롬프트 버전")
    return parser


def make_prompt(row: dict) -> str:
    return (
        "아래 뉴스가 국내 전기차 시장 전체에 미칠 영향을 분류하세요.\n\n"
        f"기사 ID: {row.get('article_id', '')}\n"
        f"제목: {row.get('title', '')}\n"
        f"요약문: {row.get('description', '')}\n\n"
        "반드시 JSON 형식으로만 답하세요."
    )


def select_rows(rows: list[dict], split: str, limit: int) -> list[dict]:
    if split == "all":
        selected = rows
    else:
        selected = [row for row in rows if row.get("split") == split]

    if limit > 0:
        return selected[:limit]

    return selected


def get_model(args_model: str) -> str:
    return args_model or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL


def validate_result(result: dict) -> None:
    if result.get("label") not in LABELS:
        raise ValueError(f"허용되지 않은 라벨입니다: {result.get('label')}")

    confidence = result.get("confidence")
    if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
        raise ValueError(f"confidence 값이 0~1 범위가 아닙니다: {confidence}")

    for key in ["reason", "evidence"]:
        if not isinstance(result.get(key), str) or not result.get(key).strip():
            raise ValueError(f"{key} 값이 비어 있습니다.")


def estimate_cost_krw(article_count: int) -> float:
    input_cost = article_count * CONSERVATIVE_INPUT_TOKENS_PER_ARTICLE * LUNA_INPUT_USD_PER_1M / 1_000_000
    output_cost = article_count * CONSERVATIVE_OUTPUT_TOKENS_PER_ARTICLE * LUNA_OUTPUT_USD_PER_1M / 1_000_000
    return (input_cost + output_cost) * USD_TO_KRW


def get_usage_value(usage, *names: str) -> int:
    for name in names:
        if isinstance(usage, dict) and name in usage:
            return usage[name] or 0

        value = getattr(usage, name, None)
        if value is not None:
            return value

    return 0


def estimate_actual_cost_krw(usage) -> float:
    input_tokens = get_usage_value(usage, "input_tokens", "prompt_tokens")
    output_tokens = get_usage_value(usage, "output_tokens", "completion_tokens")
    input_cost = input_tokens * LUNA_INPUT_USD_PER_1M / 1_000_000
    output_cost = output_tokens * LUNA_OUTPUT_USD_PER_1M / 1_000_000
    return (input_cost + output_cost) * USD_TO_KRW


def classify_with_openai(client, model: str, row: dict, prompt_version: str) -> dict:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "developer", "content": PROMPT_VERSIONS[prompt_version]},
            {"role": "user", "content": make_prompt(row)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "ev_news_sentiment",
                "schema": JSON_SCHEMA,
                "strict": True,
            }
        },
        store=False,
    )

    result = json.loads(response.output_text)
    validate_result(result)
    return result, response.usage


def is_insufficient_quota_error(error: Exception) -> bool:
    error_text = str(error)
    return "insufficient_quota" in error_text or "exceeded your current quota" in error_text


def is_rate_limit_error(error: Exception) -> bool:
    error_text = str(error)
    return "rate_limit_exceeded" in error_text or "Rate limit reached" in error_text


def classify_with_retry(client, model: str, row: dict, prompt_version: str, max_retries: int) -> tuple[dict, object]:
    for retry_count in range(max_retries + 1):
        try:
            return classify_with_openai(client, model, row, prompt_version)
        except Exception as exception:
            if not is_rate_limit_error(exception) or retry_count == max_retries:
                raise

            wait_seconds = 7 * (retry_count + 1)
            print(f"  rate limit 발생: {wait_seconds}초 뒤 재시도합니다.")
            time.sleep(wait_seconds)

    raise RuntimeError("재시도 후에도 분류에 실패했습니다.")


def merge_predictions(existing_rows: list[dict], new_rows: list[dict]) -> list[dict]:
    merged_by_id = {row.get("article_id", ""): row for row in existing_rows}

    for row in new_rows:
        merged_by_id[row.get("article_id", "")] = row

    return list(merged_by_id.values())


def sum_estimated_cost_krw(rows: list[dict]) -> float:
    total = 0

    for row in rows:
        try:
            total += float(row.get("estimated_cost_krw") or 0)
        except ValueError:
            continue

    return total


def get_output_paths(project_root: Path, prompt_version: str) -> tuple[Path, Path]:
    prediction_dir = project_root / "data" / "predictions"
    report_dir = project_root / "reports"

    if prompt_version == "v1":
        prediction_filename = PREDICTION_FILENAME
        report_filename = REPORT_FILENAME
    else:
        prediction_filename = f"openai_gpt_{prompt_version}_predictions_20260803.csv"
        report_filename = f"openai_gpt_{prompt_version}_report_20260803.txt"

    return prediction_dir / prediction_filename, report_dir / report_filename


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    load_env_file(project_root / ".env")

    label_path = project_root / "data" / "labels" / LABEL_FILENAME
    prediction_dir = project_root / "data" / "predictions"
    report_dir = project_root / "reports"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    prediction_path, report_path = get_output_paths(project_root, args.prompt_version)

    if not label_path.exists():
        print(f"라벨링 CSV 파일을 찾을 수 없습니다: {label_path}")
        return

    rows = read_csv(label_path)
    existing_prediction_rows = read_csv(prediction_path) if prediction_path.exists() else []

    if args.retry_failed_only:
        failed_article_ids = {
            row.get("article_id", "")
            for row in existing_prediction_rows
            if row.get("pred_label") == "분류실패"
        }
        selected_rows = [row for row in rows if row.get("article_id", "") in failed_article_ids]
    else:
        selected_rows = select_rows(rows, args.split, args.limit)

    model = get_model(args.model)
    estimated_krw = estimate_cost_krw(len(selected_rows))

    print("OpenAI GPT API 분류 설정")
    print(f"모델: {model}")
    print(f"분류 대상: {args.split}")
    print(f"프롬프트 버전: {args.prompt_version}")
    print(f"처리 예정 건수: {len(selected_rows)}")
    print(f"실패 행만 재시도: {args.retry_failed_only}")
    print(f"보수적 추정 비용: 약 {estimated_krw:.2f}원")
    print(f"예산 한도: {args.max_krw}원")
    print("주의: --run 옵션을 붙이면 실제 API 호출이 실행되며 비용이 발생할 수 있습니다.")
    print()

    if not args.run:
        print("현재는 dry-run입니다. 실제 API 호출은 하지 않았습니다.")
        print("실행하려면 예를 들어 다음 명령을 사용하세요.")
        print("uv run python src/openai_gpt_classifier.py --run --split dev --limit 3")
        return

    if estimated_krw > args.max_krw:
        print("보수적 추정 비용이 예산 한도를 넘어서 실행하지 않습니다.")
        print("처리 건수나 --max-krw 값을 조정하세요.")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY 환경변수가 없습니다.")
        print(".env 파일에 OPENAI_API_KEY를 추가한 뒤 다시 실행하세요.")
        return

    try:
        from openai import OpenAI
    except ImportError:
        print("openai 패키지가 설치되어 있지 않습니다.")
        print("먼저 다음 명령을 실행하세요.")
        print("uv sync")
        return

    client = OpenAI()
    prediction_rows = []
    total_actual_krw = 0
    should_stop = False

    for index, row in enumerate(selected_rows, start=1):
        try:
            result, usage = classify_with_retry(client, model, row, args.prompt_version, args.max_retries)
            pred_label = result["label"]
            confidence = result["confidence"]
            reason = result["reason"]
            evidence = result["evidence"]
            actual_krw = estimate_actual_cost_krw(usage)
            total_actual_krw += actual_krw
            error = ""
        except Exception as exception:
            pred_label = "분류실패"
            confidence = ""
            reason = ""
            evidence = ""
            actual_krw = 0
            error = str(exception)
            should_stop = is_insufficient_quota_error(exception)

        prediction_rows.append(
            {
                "article_id": row.get("article_id", ""),
                "split": row.get("split", ""),
                "gold_label": row.get("gold_label", ""),
                "pred_label": pred_label,
                "confidence": confidence,
                "reason": reason,
                "evidence": evidence,
                "estimated_cost_krw": f"{actual_krw:.4f}",
                "error": error,
                "model": model,
                "prompt_version": args.prompt_version,
                "title": row.get("title", ""),
                "description": row.get("description", ""),
            }
        )
        print(f"- {index}/{len(selected_rows)} {row.get('article_id', '')}: {pred_label}")

        if should_stop:
            print("OpenAI API quota 오류가 발생해 추가 요청을 중단합니다.")
            break

        if args.request_delay > 0 and index < len(selected_rows):
            time.sleep(args.request_delay)

    if args.retry_failed_only and existing_prediction_rows:
        rows_for_output = merge_predictions(existing_prediction_rows, prediction_rows)
    else:
        rows_for_output = prediction_rows

    cumulative_estimated_krw = sum_estimated_cost_krw(rows_for_output)

    save_csv(
        prediction_path,
        rows_for_output,
        [
            "article_id",
            "split",
            "gold_label",
            "pred_label",
            "confidence",
            "reason",
            "evidence",
            "estimated_cost_krw",
            "error",
            "model",
            "prompt_version",
            "title",
            "description",
        ],
    )

    metrics = calculate_metrics(rows_for_output, split="dev")
    save_report(
        report_path,
        "7단계 OpenAI GPT API 평가 리포트",
        metrics,
        [
            f"모델: {model}",
            f"프롬프트 버전: {args.prompt_version}",
            "출력 형식: Structured Outputs JSON Schema",
            f"이번 실행 사용량 기반 추정 비용: 약 {total_actual_krw:.4f}원",
            f"예측 파일 누적 추정 비용: 약 {cumulative_estimated_krw:.4f}원",
        ],
    )

    print()
    print("OpenAI GPT API 분류 완료")
    print(f"이번 실행 사용량 기반 추정 비용: 약 {total_actual_krw:.4f}원")
    print(f"예측 파일 누적 추정 비용: 약 {cumulative_estimated_krw:.4f}원")
    print(f"예측 CSV 파일: {prediction_path}")
    print(f"평가 리포트 파일: {report_path}")


if __name__ == "__main__":
    main()
