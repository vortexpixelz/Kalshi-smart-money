import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
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


def _load_snapshot_trades() -> pd.DataFrame:
    snapshot_path = ROOT / "tests" / "data" / "sandbox_trades_snapshot.json"
    df = pd.read_json(snapshot_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _build_mock_snapshot_payloads() -> Tuple[list, list]:
    trades = _load_snapshot_trades()
    markets = [
        {
            "ticker": "DEMO-TEST",
            "title": "Demo Market",
            "status": "active",
            "volume": float(trades["volume"].sum()),
            "yes_price": 50,
            "open_interest": float(trades["volume"].mean()),
        }
    ]
    return markets, trades.to_dict(orient="records")


class _MockResponse:
    def __init__(self, payload: Dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> Dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"Mocked request failed with {self.status_code}")


class _MockSession:
    def __init__(self, markets: list, trades: list) -> None:
        self.headers: Dict[str, str] = {}
        self._markets = markets
        self._trades = trades

    def request(self, method: str, url: str, **_kwargs) -> _MockResponse:
        if url.endswith("/trade-api/v2/login"):
            return _MockResponse({"token": "mock-token"})
        if url.endswith("/trade-api/v2/markets"):
            return _MockResponse({"markets": self._markets})
        if "/trade-api/v2/markets/" in url and url.endswith("/trades"):
            return _MockResponse({"trades": self._trades})
        return _MockResponse({}, status_code=404)

    def close(self) -> None:
        return None


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
def sandbox_login_credentials(
    sandbox_credentials: Dict[str, Optional[str]],
) -> Tuple[Optional[str], Optional[str]]:
    return sandbox_credentials.get("email"), sandbox_credentials.get("password")


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


@pytest.fixture(scope="session")
def mocked_kalshi_client() -> KalshiClient:
    markets, trades = _build_mock_snapshot_payloads()
    session = _MockSession(markets=markets, trades=trades)
    client = KalshiClient(api_key="mock-token", api_base="https://mock-api", session=session)
    yield client
    client.close()


@pytest.fixture(scope="session")
def kalshi_client(
    sandbox_mode: bool,
    mocked_kalshi_client: KalshiClient,
    request: pytest.FixtureRequest,
) -> KalshiClient:
    if sandbox_mode:
        return request.getfixturevalue("live_kalshi_client")
    return mocked_kalshi_client


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
