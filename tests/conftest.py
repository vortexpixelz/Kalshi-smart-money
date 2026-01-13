import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from _pytest.config.argparsing import Parser

from smart_money_detection.kalshi_client import KalshiClient



def pytest_addoption(parser: Parser) -> None:
    parser.addoption(
        "--live-sandbox",
        action="store_true",
        default=False,
        help="Run Kalshi integration tests against the sandbox environment.",
    )


def _collect_sandbox_credentials() -> Dict[str, Optional[str]]:
    return {
        "api_key": os.getenv("KALSHI_SANDBOX_API_KEY"),
        "email": os.getenv("KALSHI_SANDBOX_EMAIL"),
        "password": os.getenv("KALSHI_SANDBOX_PASSWORD"),
        "api_base": os.getenv("KALSHI_SANDBOX_API_BASE"),
    }


@pytest.fixture(scope="session")
def sandbox_mode(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--live-sandbox"))


@pytest.fixture(scope="session")
def sandbox_credentials() -> Dict[str, Optional[str]]:
    return _collect_sandbox_credentials()


@pytest.fixture(scope="session")
def has_sandbox_credentials(sandbox_credentials: Dict[str, Optional[str]]) -> bool:
    return bool(sandbox_credentials.get("api_key"))


@pytest.fixture(scope="session")
def live_kalshi_client(
    sandbox_mode: bool, sandbox_credentials: Dict[str, Optional[str]]
) -> KalshiClient:
    if not sandbox_mode:
        pytest.skip("Sandbox mode disabled. Pass --live-sandbox to enable live tests.")

    api_key = sandbox_credentials.get("api_key")
    if not api_key:
        pytest.skip("Sandbox credentials missing. Set KALSHI_SANDBOX_API_KEY to run live tests.")

    api_base = sandbox_credentials.get("api_base") or "https://demo-api.kalshi.com"

    client = KalshiClient(api_key=api_key, api_base=api_base)
    yield client
    client.close()


@pytest.fixture()
def sandbox_retry() -> Callable[[Callable[[], object], int, float], object]:
    def _retry(fn: Callable[[], object], attempts: int = 3, delay: float = 1.0) -> object:
        last_error: Optional[BaseException] = None
        for attempt in range(1, attempts + 1):
            try:
                return fn()
            except BaseException as exc:  # pragma: no cover - defensive, logged in caller
                last_error = exc
                if attempt == attempts:
                    raise
                time.sleep(delay)
        if last_error is not None:
            raise last_error
        return None

    return _retry
