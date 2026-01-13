import logging
import pandas as pd
import pytest
import requests

from smart_money_detection.kalshi_client import KalshiClient


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
