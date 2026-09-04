import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://openapi.naver.com/v1/search/news.json"


def load_env_file(env_path: Path) -> None:
    """간단한 .env 파일을 읽어 환경변수로 등록한다."""
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def get_required_env(name: str) -> str:
    value = os.environ.get(name)

    if not value:
        raise RuntimeError(f"환경변수 {name} 값이 없습니다.")

    return value


def search_news(query: str, display: int = 10, start: int = 1, sort: str = "date") -> dict:
    client_id = get_required_env("NAVER_CLIENT_ID")
    client_secret = get_required_env("NAVER_CLIENT_SECRET")

    params = {
        "query": query,
        "display": display,
        "start": start,
        "sort": sort,
    }
    url = f"{API_URL}?{urlencode(params)}"

    request = Request(
        url,
        headers={
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret,
        },
        method="GET",
    )

    with urlopen(request, timeout=10) as response:
        status_code = response.status
        body = response.read().decode("utf-8")

    data = json.loads(body)
    data["_status_code"] = status_code

    return data


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_env_file(project_root / ".env")

    try:
        data = search_news(query="전기차", display=10, start=1, sort="date")
    except RuntimeError as error:
        print("환경변수 설정 오류")
        print(error)
        print()
        print("프로젝트 루트에 .env 파일을 만들고 다음 값을 입력하세요.")
        print("NAVER_CLIENT_ID=발급받은_클라이언트_ID")
        print("NAVER_CLIENT_SECRET=발급받은_클라이언트_SECRET")
        return
    except HTTPError as error:
        print("API 요청 실패")
        print(f"상태 코드: {error.code}")
        print(error.read().decode("utf-8", errors="replace"))
        return
    except URLError as error:
        print("네트워크 연결 오류")
        print(error.reason)
        return

    items = data.get("items", [])

    print("API 연결 확인 결과")
    print(f"상태 코드: {data.get('_status_code')}")
    print(f"전체 검색 결과 수(total): {data.get('total')}")
    print(f"요청 시작 위치(start): {data.get('start')}")
    print(f"응답 기사 수(display): {data.get('display')}")
    print(f"items 타입: {type(items).__name__}")
    print(f"items 개수: {len(items)}")
    print()

    if items:
        first_item = items[0]
        print("첫 번째 기사 데이터 구조")
        print(f"필드 목록: {list(first_item.keys())}")
        print(f"제목 예시: {first_item.get('title')}")
        print(f"발행일 예시: {first_item.get('pubDate')}")


if __name__ == "__main__":
    main()
