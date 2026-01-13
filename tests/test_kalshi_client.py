import logging
from datetime import datetime, timezone
from typing import Optional, Tuple
import pandas as pd
import pytest
import requests

from smart_money_detection.kalshi_client import KalshiClient


logger = logging.getLogger(__name__)


def _log_timestamp(label: str) -> None:
    logger.info("%s at %s", label, datetime.now(timezone.utc).isoformat())


@pytest.mark.integration
@pytest.mark.sandbox
@pytest.mark.usefixtures("sandbox_retry")
def test_sandbox_login(
    live_kalshi_client: KalshiClient,
    sandbox_login_credentials: Tuple[Optional[str], Optional[str]],
    sandbox_retry,
) -> None:
    email, password = sandbox_login_credentials
    if not email or not password:
        pytest.skip("Sandbox email/password not set; skipping login integration test.")

    _log_timestamp("sandbox login start")
    response = sandbox_retry(lambda: live_kalshi_client.login(email, password))
    assert isinstance(response, dict)
    assert response, "Expected login response payload from sandbox"
    logger.info("Sandbox login response keys: %s", sorted(response.keys()))
    _log_timestamp("sandbox login end")


@pytest.mark.integration
@pytest.mark.sandbox
@pytest.mark.usefixtures("sandbox_retry")
def test_get_markets_returns_data(
    live_kalshi_client: KalshiClient, sandbox_retry
) -> None:
    _log_timestamp("sandbox markets start")
    markets = sandbox_retry(lambda: live_kalshi_client.get_markets(limit=5))
    assert isinstance(markets, list)
    assert markets, "Expected at least one market in sandbox"
    assert all("ticker" in market for market in markets)
    logger.info("Fetched %d sandbox markets", len(markets))
    _log_timestamp("sandbox markets end")


@pytest.mark.integration
@pytest.mark.sandbox
@pytest.mark.usefixtures("sandbox_retry")
def test_get_trades_for_market(
    live_kalshi_client: KalshiClient, sandbox_retry
) -> None:
    _log_timestamp("sandbox trades start")
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
    _log_timestamp("sandbox trades end")


def test_mocked_kalshi_client_happy_path(kalshi_client: KalshiClient) -> None:
    markets = kalshi_client.get_markets(limit=1)
    assert markets
    ticker = markets[0]["ticker"]
    trades = kalshi_client.get_trades(ticker, limit=5)
    assert isinstance(trades, pd.DataFrame)
    assert not trades.empty


def test_get_markets_handles_request_errors(monkeypatch, caplog) -> None:
    client = KalshiClient(api_key="bad-key", api_base="https://api.example.com")

    def _raise(*_args, **_kwargs):
        raise requests.exceptions.HTTPError("boom")

    monkeypatch.setattr(client.session, "request", _raise)

    with caplog.at_level("ERROR"):
        markets = client.get_markets()

    assert markets == []
    assert "Failed to fetch markets" in caplog.text


def test_login_handles_request_errors(monkeypatch, caplog) -> None:
    client = KalshiClient(api_key="bad-key", api_base="https://api.example.com")

    def _raise(*_args, **_kwargs):
        raise requests.exceptions.HTTPError("boom")

    monkeypatch.setattr(client.session, "request", _raise)

    with caplog.at_level("ERROR"):
        response = client.login("test@example.com", "password")

    assert response == {}
    assert "Failed to login" in caplog.text


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
