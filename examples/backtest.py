#!/usr/bin/env python3
"""
Backtest smart money detection against historical Kalshi trade data.

Strategy:
  - Split trade history into a TRAIN window (fit detector) and a TEST window
    (simulate following detected smart money signals).
  - For every flagged trade in the test window, assume we copy the smart money
    position at that price.  Mark each position to market at the last observed
    price in the window.
  - Report detection stats, simulated P&L, and win-rate.

Usage
-----
    # With live Kalshi API (requires KALSHI_API_KEY env var)
    python examples/backtest.py --ticker SOME-TICKER

    # Offline using the bundled mock snapshot
    python examples/backtest.py --mock
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Make sure the project root is importable when running directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from smart_money_detection import SmartMoneyDetector
from smart_money_detection.config import load_config
from smart_money_detection.kalshi_client import KalshiClient

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MOCK_SNAPSHOT = (
    Path(__file__).parent.parent / "tests" / "data" / "sandbox_trades_snapshot.json"
)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_mock_trades() -> pd.DataFrame:
    """Load the static offline snapshot bundled with the test suite."""
    with open(MOCK_SNAPSHOT) as fh:
        records = json.load(fh)
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if "side" in df.columns and "taker_side" not in df.columns:
        df = df.rename(columns={"side": "taker_side"})
    return df.sort_values("timestamp").reset_index(drop=True)


def _load_live_trades(ticker: str, limit: int = 2000) -> pd.DataFrame:
    """Fetch real trade history from Kalshi."""
    api_key = os.getenv("KALSHI_API_KEY")
    if not api_key:
        raise RuntimeError("KALSHI_API_KEY is not set. Use --mock for offline mode.")
    api_base = os.getenv("KALSHI_API_BASE")
    with KalshiClient(api_key=api_key, api_base=api_base) as client:
        trades = client.get_trades(ticker, limit=limit)
    if trades.empty:
        raise RuntimeError(f"No trades returned for ticker '{ticker}'.")
    return trades.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# P&L simulation helpers
# ---------------------------------------------------------------------------

def _simulate_pnl(
    trades: pd.DataFrame,
    predictions: np.ndarray,
    scores: np.ndarray,
    lookahead: int = 10,
) -> pd.DataFrame:
    """
    For every detected smart money trade, simulate copying the position.

    The exit price is the price ``lookahead`` rows later (or the last row if
    there are fewer rows remaining).  P&L is calculated per 100-cent notional:

        YES side:  pnl = exit_price - entry_price
        NO  side:  pnl = entry_price - exit_price   (betting against YES)

    Parameters
    ----------
    trades : DataFrame
        Test-window trades sorted by timestamp.
    predictions : ndarray
        Binary flags (1 = smart money, 0 = normal).
    scores : ndarray
        Continuous anomaly scores [0, 1].
    lookahead : int
        Number of subsequent trades used to set the exit price.

    Returns
    -------
    DataFrame with one row per detected signal.
    """
    results: List[dict] = []
    detected_idx = np.where(predictions == 1)[0]

    prices = trades["price"].values
    sides = (
        trades["taker_side"].values
        if "taker_side" in trades.columns
        else np.full(len(trades), "yes")
    )

    for idx in detected_idx:
        exit_idx = min(idx + lookahead, len(trades) - 1)
        entry_price = float(prices[idx])
        exit_price = float(prices[exit_idx])
        side = str(sides[idx]).lower()

        if side in ("yes", "buy"):
            pnl = exit_price - entry_price
        else:
            pnl = entry_price - exit_price

        results.append(
            {
                "trade_id": trades.iloc[idx].get("trade_id", idx),
                "timestamp": trades.iloc[idx]["timestamp"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "side": side,
                "score": float(scores[idx]),
                "pnl_cents": pnl,
                "win": pnl > 0,
            }
        )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Backtest core
# ---------------------------------------------------------------------------

def run_backtest(
    trades: pd.DataFrame,
    train_frac: float = 0.70,
    lookahead: int = 10,
    weighting_method: str = "thompson",
) -> None:
    """
    Run a full train/test backtest and print results.

    Parameters
    ----------
    trades : DataFrame
        Full trade history sorted by timestamp.
    train_frac : float
        Fraction of trades used for training the detector.
    lookahead : int
        How many subsequent trades to use as the exit price.
    weighting_method : str
        Ensemble weighting strategy ('thompson', 'ucb', 'mwu', 'uniform').
    """
    n = len(trades)
    split = int(n * train_frac)
    train_df = trades.iloc[:split].copy()
    test_df = trades.iloc[split:].copy().reset_index(drop=True)

    _hr = "=" * 70

    print(_hr)
    print("  Smart Money Detection — Backtest")
    print(_hr)
    print(f"\n  Total trades   : {n:,}")
    print(f"  Train window   : {len(train_df):,} trades  ({train_frac*100:.0f}%)")
    print(f"  Test  window   : {len(test_df):,} trades  ({(1-train_frac)*100:.0f}%)")
    if "timestamp" in trades.columns:
        print(f"  Time range     : {trades['timestamp'].min()} → {trades['timestamp'].max()}")

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    print(f"\n  Fitting detector on {len(train_df):,} training trades …")
    config = load_config()
    config.ensemble.weighting_method = weighting_method
    detector = SmartMoneyDetector(config)
    detector.fit(
        train_df,
        volume_col="volume",
        timestamp_col="timestamp",
        price_col="price",
    )
    print(f"  Detector fitted  ✓  (weighting={weighting_method})")

    # ------------------------------------------------------------------
    # Predict on test window
    # ------------------------------------------------------------------
    print(f"\n  Scoring {len(test_df):,} test trades …")
    predictions = detector.predict(test_df, volume_col="volume", timestamp_col="timestamp")
    scores = detector.score(test_df, volume_col="volume", timestamp_col="timestamp")

    n_signals = int(predictions.sum())
    detection_rate = n_signals / len(test_df) * 100

    print(f"  Smart money signals : {n_signals:,}  ({detection_rate:.1f}% of test trades)")

    # ------------------------------------------------------------------
    # Ensemble weights
    # ------------------------------------------------------------------
    print(f"\n{'─' * 70}")
    print("  Ensemble Detector Weights")
    print(f"{'─' * 70}")
    for name, weight in detector.get_ensemble_weights().items():
        bar = "█" * max(1, int(weight * 40))
        print(f"  {name:<22} {weight:.3f}  {bar}")

    # ------------------------------------------------------------------
    # P&L simulation
    # ------------------------------------------------------------------
    print(f"\n{'─' * 70}")
    print(f"  P&L Simulation  (lookahead={lookahead} trades)")
    print(f"{'─' * 70}")

    if n_signals == 0:
        print("  No signals detected — nothing to simulate.")
        return

    pnl_df = _simulate_pnl(test_df, predictions, scores, lookahead=lookahead)

    total_pnl = pnl_df["pnl_cents"].sum()
    avg_pnl = pnl_df["pnl_cents"].mean()
    win_rate = pnl_df["win"].mean() * 100
    sharpe_proxy = (
        pnl_df["pnl_cents"].mean() / pnl_df["pnl_cents"].std()
        if pnl_df["pnl_cents"].std() > 0
        else float("nan")
    )

    print(f"\n  Signals simulated  : {len(pnl_df):,}")
    print(f"  Total P&L          : {total_pnl:+.2f} ¢")
    print(f"  Avg P&L / signal   : {avg_pnl:+.2f} ¢")
    print(f"  Win rate           : {win_rate:.1f}%")
    print(f"  Sharpe proxy       : {sharpe_proxy:.3f}")

    # Top signals
    top = pnl_df.nlargest(5, "pnl_cents")
    worst = pnl_df.nsmallest(5, "pnl_cents")

    print(f"\n  Top 5 Winning Signals:")
    _print_signal_table(top)

    print(f"\n  Top 5 Losing Signals:")
    _print_signal_table(worst)

    # Distribution
    buckets = {
        "> +5¢":   (pnl_df["pnl_cents"] > 5).sum(),
        "0 to +5¢": ((pnl_df["pnl_cents"] > 0) & (pnl_df["pnl_cents"] <= 5)).sum(),
        "-5 to 0¢": ((pnl_df["pnl_cents"] >= -5) & (pnl_df["pnl_cents"] <= 0)).sum(),
        "< -5¢":   (pnl_df["pnl_cents"] < -5).sum(),
    }
    print(f"\n  P&L Distribution:")
    for label, count in buckets.items():
        bar = "▓" * count
        print(f"  {label:<12}  {count:>4}  {bar}")

    print(f"\n{'=' * 70}")
    verdict = "PROFITABLE" if total_pnl > 0 else "UNPROFITABLE"
    print(f"  Backtest Result: {verdict}  (total P&L = {total_pnl:+.2f} ¢)")
    print(f"{'=' * 70}\n")

    return pnl_df


def _print_signal_table(df: pd.DataFrame) -> None:
    header = f"  {'Trade ID':<15} {'Timestamp':<20} {'Entry':>6} {'Exit':>6} {'Side':<6} {'Score':>6} {'P&L':>7}"
    print(header)
    print(f"  {'─' * (len(header) - 2)}")
    for _, row in df.iterrows():
        ts = row["timestamp"]
        if hasattr(ts, "strftime"):
            ts = ts.strftime("%Y-%m-%d %H:%M")
        print(
            f"  {str(row['trade_id']):<15} {str(ts):<20} "
            f"{row['entry_price']:>6.1f} {row['exit_price']:>6.1f} "
            f"{row['side']:<6} {row['score']:>6.3f} {row['pnl_cents']:>+7.2f}¢"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest smart money detection")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--ticker", default=None, help="Kalshi market ticker (live mode)")
    source.add_argument("--mock", action="store_true", help="Use bundled offline snapshot")
    p.add_argument("--train-frac", type=float, default=0.70, help="Training fraction (default 0.70)")
    p.add_argument("--lookahead", type=int, default=10, help="Exit lookahead in trades (default 10)")
    p.add_argument("--limit", type=int, default=2000, help="Max trades to fetch (live mode)")
    p.add_argument(
        "--weighting",
        default="thompson",
        choices=["uniform", "thompson", "ucb", "mwu"],
        help="Ensemble weighting strategy",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if args.mock or (not args.ticker and not os.getenv("KALSHI_API_KEY")):
        print("Running in OFFLINE mode using bundled mock snapshot.\n")
        trades = _load_mock_trades()
    elif args.ticker:
        print(f"Fetching live data for ticker '{args.ticker}' …\n")
        try:
            trades = _load_live_trades(args.ticker, limit=args.limit)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return 1
    else:
        # No ticker given but API key is present — try first active market
        api_key = os.getenv("KALSHI_API_KEY")
        api_base = os.getenv("KALSHI_API_BASE")
        with KalshiClient(api_key=api_key, api_base=api_base) as client:
            markets = client.get_markets(limit=5)
        if not markets:
            print("ERROR: No active markets found. Use --ticker or --mock.")
            return 1
        ticker = markets[0]["ticker"]
        print(f"No ticker specified — using first active market: {ticker}\n")
        trades = _load_live_trades(ticker, limit=args.limit)

    if len(trades) < 10:
        print(f"ERROR: Only {len(trades)} trades available — need at least 10 to backtest.")
        return 1

    run_backtest(
        trades,
        train_frac=args.train_frac,
        lookahead=args.lookahead,
        weighting_method=args.weighting,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
