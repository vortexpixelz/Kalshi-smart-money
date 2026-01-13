import logging
import pandas as pd
import pytest
import requests

from smart_money_detection.kalshi_client import KalshiApiError, KalshiClient


logger = logging.getLogger(__name__)


@pytest.mark.integration
@pytest.mark.sandbox
@pytest.mark.usefixtures("sandbox_retry")
def test_get_markets_returns_data(
    live_kalshi_client: KalshiClient, sandbox_retry
) -> None:
    markets = sandbox_retry(lambda: live_kalshi_client.get_markets(limit=5))
    assert isinstance(markets, list)
    assert markets, "Expected at least one market in sandbox"
    assert all("ticker" in market for market in markets)
    logger.info("Fetched %d sandbox markets", len(markets))


@pytest.mark.integration
@pytest.mark.sandbox
@pytest.mark.usefixtures("sandbox_retry")
def test_get_trades_for_market(
    live_kalshi_client: KalshiClient, sandbox_retry
) -> None:
    markets = sandbox_retry(lambda: live_kalshi_client.get_markets(limit=1))
    if not markets:
        pytest.skip("Sandbox returned no markets to query trades for.")
    ticker = markets[0]["ticker"]
    trades = sandbox_retry(lambda: live_kalshi_client.get_trades(ticker, limit=100))
    assert isinstance(trades, pd.DataFrame)
    assert not trades.empty, "Expected sandbox trade history"
    expected_columns = {"timestamp", "volume"}
    assert expected_columns.issubset(trades.columns)
    logger.info("Retrieved %d sandbox trades for %s", len(trades), ticker)


def test_get_markets_handles_request_errors(monkeypatch, caplog) -> None:
    client = KalshiClient(api_key="bad-key", api_base="https://api.example.com")

    def _raise(*_args, **_kwargs):
        raise requests.exceptions.HTTPError("boom")

    monkeypatch.setattr(client.session, "request", _raise)

    with caplog.at_level("ERROR"):
        markets = client.get_markets()

    assert markets == []
    assert "Failed to fetch markets" in caplog.text


def test_get_trades_handles_request_errors(monkeypatch, caplog) -> None:
    client = KalshiClient(api_key="bad-key", api_base="https://api.example.com")

    def _raise(*_args, **_kwargs):
        raise requests.exceptions.HTTPError("boom")

    monkeypatch.setattr(client.session, "request", _raise)

    with caplog.at_level("ERROR"):
        trades = client.get_trades("TEST", limit=10)

    assert isinstance(trades, pd.DataFrame)
    assert trades.empty
    assert "Failed to fetch trades" in caplog.text


class _FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or ""

    def json(self):
        return self._payload


def test_request_retries_on_retryable_status(monkeypatch) -> None:
    client = KalshiClient(
        api_key="test-key",
        api_base="https://api.example.com",
        max_retries=2,
        backoff_factor=0.0,
    )
    responses = [
        _FakeResponse(500, payload={"error": "server"}),
        _FakeResponse(200, payload={"markets": [{"ticker": "TEST"}]}),
    ]

    def _request(*_args, **_kwargs):
        return responses.pop(0)

    monkeypatch.setattr(client.session, "request", _request)

    response = client._request("GET", "/trade-api/v2/markets")

    assert response.status_code == 200
    assert response.data["markets"][0]["ticker"] == "TEST"
    assert response.metrics.retries_attempted == 1


def test_request_raises_on_non_retryable_status(monkeypatch) -> None:
    client = KalshiClient(
        api_key="test-key",
        api_base="https://api.example.com",
        max_retries=2,
        backoff_factor=0.0,
    )

    def _request(*_args, **_kwargs):
        return _FakeResponse(404, payload={"error": "missing"}, text="Not Found")

    monkeypatch.setattr(client.session, "request", _request)

    with pytest.raises(KalshiApiError) as exc_info:
        client._request("GET", "/trade-api/v2/markets/missing")

    exc = exc_info.value
    assert exc.endpoint == "/trade-api/v2/markets/missing"
    assert exc.status_code == 404
    assert "Not Found" in (exc.response_body or "")
    assert exc.retries_attempted == 0


def test_request_retries_on_timeout_exception(monkeypatch) -> None:
    client = KalshiClient(
        api_key="test-key",
        api_base="https://api.example.com",
        max_retries=1,
        backoff_factor=0.0,
    )

    def _raise(*_args, **_kwargs):
        raise requests.exceptions.Timeout("too slow")

    monkeypatch.setattr(client.session, "request", _raise)

    with pytest.raises(KalshiApiError) as exc_info:
        client._request("GET", "/trade-api/v2/markets")

    exc = exc_info.value
    assert exc.endpoint == "/trade-api/v2/markets"
    assert exc.status_code is None
    assert "too slow" in (exc.response_body or "")
    assert exc.retries_attempted == 1
