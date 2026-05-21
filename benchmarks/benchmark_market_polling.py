"""Benchmark script to simulate Kalshi market polling workloads."""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from smart_money_detection.kalshi_client import KalshiClient
from smart_money_detection.utils.performance import (
    get_performance_collector,
    get_telemetry_logger,
    reset_performance_collector,
)


@dataclass
class MockResponse:
    payload: Dict[str, Any]

    def json(self) -> Dict[str, Any]:
        return self.payload

    def raise_for_status(self) -> None:
        return None


class MockSession:
    def __init__(
        self,
        markets: List[Dict[str, Any]],
        trades: Dict[str, List[Dict[str, Any]]],
        latency_ms: float,
    ) -> None:
        self._markets = {market["ticker"]: market for market in markets}
        self._market_list = markets
        self._trades = trades
        self._latency_ms = latency_ms
        self.headers: Dict[str, str] = {}

    def request(self, method: str, url: str, **kwargs: Any) -> MockResponse:
        _ = method
        _ = kwargs
        time.sleep(self._latency_ms / 1000)
        if url.endswith("/markets"):
            return MockResponse({"markets": list(self._market_list)})
        if "/trades" in url:
            ticker = url.split("/markets/")[-1].split("/trades")[0]
            return MockResponse({"trades": list(self._trades.get(ticker, []))})
        if "/markets/" in url:
            ticker = url.split("/markets/")[-1]
            return MockResponse({"market": self._markets.get(ticker)})
        return MockResponse({})

    def close(self) -> None:
        return None


def _build_mock_data(num_markets: int, trades_per_market: int) -> tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    markets = []
    trades_by_ticker: Dict[str, List[Dict[str, Any]]] = {}
    now = datetime.now(timezone.utc)

    for idx in range(num_markets):
        ticker = f"MARKET-{idx:03d}"
        market = {
            "ticker": ticker,
            "title": f"Mock Market {idx}",
            "yes_price": round(random.uniform(0.1, 0.9), 2),
            "volume": random.randint(100, 5000),
            "open_interest": random.randint(50, 2000),
            "close_time": (now + timedelta(days=7)).isoformat(),
            "status": "active",
        }
        markets.append(market)

        trades = []
        for trade_idx in range(trades_per_market):
            trade_time = now - timedelta(minutes=trade_idx)
            trades.append(
                {
                    "timestamp": trade_time.isoformat(),
                    "volume": int(np.random.lognormal(mean=3, sigma=0.5)),
                    "price": round(random.uniform(0.1, 0.9), 2),
                }
            )
        trades_by_ticker[ticker] = trades

    return markets, trades_by_ticker


def run_benchmark(
    *,
    cycles: int,
    num_markets: int,
    trades_per_market: int,
    latency_ms: float,
    label: str,
) -> Dict[str, Any]:
    markets, trades = _build_mock_data(num_markets, trades_per_market)
    session = MockSession(markets, trades, latency_ms)

    reset_performance_collector()
    telemetry = get_telemetry_logger()

    with KalshiClient(api_base="http://mock", session=session) as client:
        start = time.perf_counter()
        for _ in range(cycles):
            market_list = client.get_markets(limit=num_markets)
            for market in market_list:
                ticker = market["ticker"]
                client.get_market(ticker)
                client.get_trades(ticker, limit=trades_per_market)
            client.get_market_summary(market_list[0]["ticker"])
        duration = time.perf_counter() - start

    collector = get_performance_collector()
    summary = collector.summary()

    throughput = cycles / duration if duration > 0 else 0
    payload = {
        "label": label,
        "duration_s": round(duration, 4),
        "cycles": cycles,
        "cycles_per_second": round(throughput, 3),
        "market_count": num_markets,
        "trades_per_market": trades_per_market,
        "latency_ms": latency_ms,
        "metrics": summary,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    telemetry.log_event({"event": "benchmark.market_polling", **payload})
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Kalshi market polling.")
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--markets", type=int, default=10)
    parser.add_argument("--trades", type=int, default=100)
    parser.add_argument("--latency-ms", type=float, default=5.0)
    parser.add_argument("--label", type=str, default="baseline")
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    results = run_benchmark(
        cycles=args.cycles,
        num_markets=args.markets,
        trades_per_market=args.trades,
        latency_ms=args.latency_ms,
        label=args.label,
    )

    output = json.dumps(results, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
