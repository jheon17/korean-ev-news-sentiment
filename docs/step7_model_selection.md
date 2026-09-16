# 7단계: 분류 방식 선택

## 1. 목적

키워드 기준 모델 이후 사용할 분류 방식을 선택하고 실행 준비를 한다.

이번 프로젝트에서는 다음 두 방식을 준비한다.

- 로컬 오픈소스 언어모델
- OpenAI GPT API

> **현재 상태 안내:** 아래 성능은 `gold_label` 재검토 전 초기 실험 기록이다.
> 이후 35건의 `gold_label` 재검토를 완료했으므로 기록된 수치를 현재 성능으로 해석하지 않으며,
> 새 평가 데이터를 구성한 뒤 다시 평가할 예정이다.
> 당시 final_test는 블라인드 데이터로 설계했지만 이미 수동 검토했으므로 최종 블라인드 테스트로 사용하지 않는다.

## 2. 쉬운 개념 설명

로컬 오픈소스 모델은 내 컴퓨터에서 모델을 내려받아 실행하는 방식이다.
API 비용은 들지 않지만 처음 다운로드가 크고 실행 속도가 느릴 수 있다.

GPT API는 OpenAI 서버에 제목과 요약문을 보내 분류 결과를 받는 방식이다.
설치와 실행은 비교적 간단하지만 API 사용량에 따라 비용이 발생할 수 있다.

## 3. 사용자가 먼저 생각할 질문

1. 내 컴퓨터에서 큰 모델을 다운로드하고 실행할 수 있는가?
2. GPT API 요청 비용이 발생할 수 있음을 이해했는가?
3. 최종 테스트용 데이터는 마지막 평가 전까지 모델 수정에 사용하지 않을 수 있는가?

## 4. 설치

필요한 패키지는 다음 명령으로 설치한다.

```powershell
uv sync --extra local-models
```

로컬 오픈소스 모델은 처음 실행할 때 Hugging Face 모델 파일을 다운로드할 수 있다.

GPT API를 실행하려면 `.env` 파일에 다음 값을 추가한다.

```text
OPENAI_API_KEY=발급받은_OpenAI_API_Key
OPENAI_MODEL=gpt-5.6-luna
```

실제 API 키는 코드, README, 노트북, Git에 저장하지 않는다.

## 5. 로컬 오픈소스 모델 실행

처음 만든 로컬 모델 스크립트는 zero-shot 분류 모델을 사용한다.

```powershell
uv run python src/local_oss_classifier.py --split dev
```

Qwen 하나만 사용해 보려면 아래 스크립트를 사용한다.
기본 모델은 `Qwen/Qwen2.5-1.5B-Instruct`다.

```powershell
uv run python src/qwen_local_classifier.py --split dev --limit 3
```

Anaconda `base` 환경에서 OpenMP 중복 오류가 나면, 권장 방법은 새 가상환경을 만드는 것이다.
학습용으로 빠르게 테스트만 해보려면 아래처럼 임시 우회 옵션을 붙일 수 있다.

```powershell
uv run python src/qwen_local_classifier.py --split dev --limit 3 --allow-omp-duplicate
```

문제가 없으면 개발용 데이터 전체를 분류한다.

```powershell
uv run python src/qwen_local_classifier.py --split dev --limit 0
```

처음 실행하면 모델 다운로드 때문에 시간이 오래 걸릴 수 있다.
GPU는 `torch.cuda.is_available()`로 확인하고, CUDA 사용이 가능하면 GPU를 사용한다.
CUDA 사용이 불가능하면 CPU로 실행한다.

출력 파일은 다음 위치에 생성된다.

```text
data/predictions/local_oss_predictions_20260803.csv
reports/local_oss_report_20260803.txt
data/predictions/qwen_local_predictions_20260803.csv
reports/qwen_local_report_20260803.txt
```

## 6. GPT API 실행

기본 실행은 dry-run이라 실제 API 호출을 하지 않는다.

```powershell
uv run python src/openai_gpt_classifier.py
```

실제 API 호출은 `--run` 옵션을 붙여야 실행된다.
비용 확인 전에는 작은 건수로 먼저 테스트한다.

```powershell
uv run python src/openai_gpt_classifier.py --run --split dev --limit 3 --max-krw 10000
```

문제가 없으면 개발용 데이터 전체를 분류한다.

```powershell
uv run python src/openai_gpt_classifier.py --run --split dev --limit 0 --max-krw 10000
```

`gpt-5.6-luna` 기준으로 현재 샘플 규모는 1만 원 한도보다 훨씬 작을 가능성이 높다.
그래도 스크립트는 실행 전에 보수적 추정 비용을 출력하고,
추정 비용이 `--max-krw` 값을 넘으면 실행하지 않는다.

출력 파일은 다음 위치에 생성된다.

```text
data/predictions/openai_gpt_predictions_20260803.csv
reports/openai_gpt_report_20260803.txt
```

## 7. GPT API 출력 형식

GPT API는 다음 JSON 형식으로만 답하도록 설계한다.

```json
{
  "label": "긍정",
  "confidence": 0.8,
  "reason": "충전 인프라 확대 내용이라 전기차 이용 편의성이 좋아질 가능성이 있음",
  "evidence": "요약문에 충전소 확대 내용이 언급됨"
}
```

허용 라벨은 다음 세 개뿐이다.

- 긍정
- 중립
- 부정

잘못된 라벨, 잘못된 JSON, 빈 근거는 오류로 기록한다.

## 8. 결과 해석 방법

평가는 우선 dev 데이터만 사용한다.

비교 대상은 다음 세 가지다.

- 키워드 규칙 기준 모델
- 로컬 오픈소스 모델
- GPT API 모델

각 모델은 Accuracy만 보지 않고 Macro-F1, 클래스별 Precision, Recall, F1, 혼동행렬을 함께 본다.

## 9. 오류, 편향, 데이터 누수 가능성

로컬 오픈소스 모델은 한국어 전기차 뉴스에 특화된 모델이 아니므로 문맥을 잘못 이해할 수 있다.

GPT API는 더 자연스러운 판단을 할 수 있지만, 비용이 발생할 수 있고 항상 정답은 아니다.

최종 테스트용 데이터를 보면서 프롬프트나 키워드를 수정하면 데이터 누수가 생긴다.
따라서 모델 수정은 dev 데이터만 보고 진행한다.

## 10. 완료 조건

7단계는 다음 조건을 만족하면 완료된다.

- 로컬 오픈소스 모델 실행 스크립트가 준비되어 있다.
- GPT API 실행 스크립트가 준비되어 있다.
- GPT API는 실제 호출 전에 `--run` 확인을 요구한다.
- GPT API 출력 형식이 고정 JSON으로 설계되어 있다.
- 예측 결과와 평가 리포트 저장 위치가 정해져 있다.

## 11. Qwen 실행 확인 결과

사용자가 2026년 8월 3일에 Qwen 로컬 모델을 dev 데이터 전체 28건에 실행했다.

- 모델: `Qwen/Qwen2.5-1.5B-Instruct`
- 실행 장치: CPU
- 평가 범위: dev
- 처리 건수: 28
- 분류 성공: 24
- 분류 실패: 4

평가 결과는 다음과 같다.

| 지표 | 값 |
| --- | ---: |
| Accuracy | 0.292 |
| Macro-F1 | 0.381 |

클래스별 지표는 다음과 같다.

| 라벨 | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| 긍정 | 0.750 | 0.176 | 0.286 |
| 중립 | 0.111 | 0.667 | 0.190 |
| 부정 | 1.000 | 0.500 | 0.667 |

혼동행렬은 다음과 같다.

| 실제 \ 예측 | 긍정 | 중립 | 부정 |
| --- | ---: | ---: | ---: |
| 긍정 | 3 | 14 | 0 |
| 중립 | 1 | 2 | 0 |
| 부정 | 0 | 2 | 2 |

실패 4건은 모두 `evidence` 값이 비어 있어서 검증에서 탈락했다.
다음 실행부터는 원인 확인을 위해 `raw_output` 열에 모델 원본 출력을 함께 저장한다.

## 12. GPT API 3건 테스트 결과

사용자가 2026년 8월 3일에 GPT API를 dev 데이터 3건으로 테스트했다.

실행 명령은 다음과 같다.

```powershell
uv run python src/openai_gpt_classifier.py --run --split dev --limit 3 --max-krw 10000
```

처음 실행에서는 OpenAI API quota 부족으로 429 `insufficient_quota` 오류가 발생했다.
이후 quota 문제가 해결된 뒤 다시 실행했고, 3건 모두 정상 분류되었다.

- 처리 건수: 3
- 분류 실패 수: 0
- 실제 사용량 기반 추정 비용: 약 0.8736원
- 예측 라벨 분포: 중립 3건

평가 결과는 다음과 같다.

| 지표 | 값 |
| --- | ---: |
| Accuracy | 0.333 |
| Macro-F1 | 0.167 |

3건 테스트에서는 GPT가 모두 `중립`으로 분류했다.
근거가 부족하면 중립을 선택하라는 지침을 강하게 따른 결과로 볼 수 있다.
다만 3건만으로 모델 성능을 판단하면 안 된다.

dev 전체 28건의 보수적 추정 비용은 약 26.26원으로,
1만 원 한도보다 충분히 낮게 추정되었다.
