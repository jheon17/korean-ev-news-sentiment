import argparse
import json
import os
import re
from pathlib import Path

from evaluation_utils import LABELS, calculate_metrics, read_csv, save_csv, save_report


LABEL_FILENAME = "news_labeling_sample_20260803.csv"
PREDICTION_FILENAME = "qwen_local_predictions_20260803.csv"
REPORT_FILENAME = "qwen_local_report_20260803.txt"
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


SYSTEM_PROMPT = """
너는 전기차 뉴스 감성 분류 보조자다.
감성은 문장의 일반적인 분위기가 아니라 국내 전기차 시장 전체의 수요, 보급, 등록대수에 미칠 예상 영향이다.

라벨 기준:
- 긍정: 수요, 보급, 등록 증가 가능성을 높이는 내용
- 중립: 사실 전달, 영향이 불명확하거나 긍정과 부정이 섞인 내용
- 부정: 수요, 보급, 등록 감소 또는 위험 증가 가능성을 높이는 내용

판단 근거가 부족하면 긍정이나 부정으로 억지 분류하지 말고 중립을 선택한다.
반드시 JSON만 출력한다.
""".strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen 로컬 모델 하나로 전기차 뉴스 감성을 분류한다.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face Qwen 모델 이름")
    parser.add_argument("--split", default="dev", choices=["dev", "final_test", "all"], help="분류할 데이터 범위")
    parser.add_argument("--limit", type=int, default=3, help="최대 처리 건수. 0이면 전체 처리")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="모델이 새로 생성할 최대 토큰 수")
    parser.add_argument(
        "--allow-omp-duplicate",
        action="store_true",
        help="Anaconda base 환경의 OpenMP 중복 오류를 임시 우회한다. 권장 해법은 새 가상환경 사용이다.",
    )
    return parser


def select_rows(rows: list[dict], split: str, limit: int) -> list[dict]:
    if split == "all":
        selected = rows
    else:
        selected = [row for row in rows if row.get("split") == split]

    if limit > 0:
        return selected[:limit]

    return selected


def make_user_prompt(row: dict) -> str:
    return f"""
아래 뉴스가 국내 전기차 시장 전체에 미칠 영향을 분류하세요.

기사 ID: {row.get("article_id", "")}
제목: {row.get("title", "")}
요약문: {row.get("description", "")}

출력 형식:
{{
  "label": "긍정 또는 중립 또는 부정",
  "confidence": 0.0,
  "reason": "라벨 판단 이유",
  "evidence": "제목이나 요약문에서 근거가 된 표현"
}}
""".strip()


def extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("모델 출력에서 JSON 객체를 찾지 못했습니다.")

    return json.loads(match.group(0))


def validate_result(result: dict) -> None:
    if result.get("label") not in LABELS:
        raise ValueError(f"허용되지 않은 라벨입니다: {result.get('label')}")

    confidence = result.get("confidence")
    if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
        raise ValueError(f"confidence 값이 0~1 범위가 아닙니다: {confidence}")

    for key in ["reason", "evidence"]:
        if not isinstance(result.get(key), str) or not result.get(key).strip():
            raise ValueError(f"{key} 값이 비어 있습니다.")


def load_qwen_model(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
    )
    model.to(device)
    model.eval()

    return tokenizer, model, device


def classify_with_qwen(tokenizer, model, device: str, row: dict, max_new_tokens: int) -> tuple[dict, str]:
    import torch

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": make_user_prompt(row)},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    result = extract_json(output_text)
    validate_result(result)
    return result, output_text


def main() -> None:
    args = build_parser().parse_args()

    if args.allow_omp_duplicate:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
        tokenizer, model, device = load_qwen_model(args.model)
    except ImportError:
        print("transformers 또는 torch 패키지가 설치되어 있지 않습니다.")
        print("먼저 다음 명령을 실행하세요.")
        print("uv sync --extra local-models")
        return

    rows = read_csv(label_path)
    selected_rows = select_rows(rows, args.split, args.limit)

    print("Qwen 로컬 모델 분류 시작")
    print(f"모델: {args.model}")
    print(f"실행 장치: {device}")
    print(f"분류 대상: {args.split}")
    print(f"처리 건수: {len(selected_rows)}")
    print()

    prediction_rows = []

    for index, row in enumerate(selected_rows, start=1):
        try:
            result, raw_output = classify_with_qwen(tokenizer, model, device, row, args.max_new_tokens)
            pred_label = result["label"]
            confidence = result["confidence"]
            reason = result["reason"]
            evidence = result["evidence"]
            error = ""
        except Exception as exception:
            pred_label = "분류실패"
            confidence = ""
            reason = ""
            evidence = ""
            raw_output = locals().get("raw_output", "")
            error = str(exception)

        prediction_rows.append(
            {
                "article_id": row.get("article_id", ""),
                "split": row.get("split", ""),
                "gold_label": row.get("gold_label", ""),
                "pred_label": pred_label,
                "confidence": confidence,
                "reason": reason,
                "evidence": evidence,
                "error": error,
                "raw_output": raw_output,
                "model": args.model,
                "device": device,
                "title": row.get("title", ""),
                "description": row.get("description", ""),
            }
        )
        print(f"- {index}/{len(selected_rows)} {row.get('article_id', '')}: {pred_label}")

    prediction_path = prediction_dir / PREDICTION_FILENAME
    report_path = report_dir / REPORT_FILENAME
    save_csv(
        prediction_path,
        prediction_rows,
        [
            "article_id",
            "split",
            "gold_label",
            "pred_label",
            "confidence",
            "reason",
            "evidence",
            "error",
            "raw_output",
            "model",
            "device",
            "title",
            "description",
        ],
    )

    metrics = calculate_metrics(prediction_rows, split="dev")
    save_report(
        report_path,
        "7단계 Qwen 로컬 모델 평가 리포트",
        metrics,
        [f"모델: {args.model}", f"실행 장치: {device}", "방식: 로컬 생성형 모델 JSON 출력"],
    )

    print()
    print("Qwen 로컬 모델 분류 완료")
    print(f"예측 CSV 파일: {prediction_path}")
    print(f"평가 리포트 파일: {report_path}")


if __name__ == "__main__":
    main()
