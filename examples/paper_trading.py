#!/usr/bin/env python3
"""
Live paper trading monitor for smart money detection.

Polls the Kalshi API on a configurable interval, runs the smart money
detector on each batch of new trades, and tracks a simulated paper
portfolio — no real orders are placed.

Paper P&L logic
---------------
When a smart money signal fires at price P with taker side "yes" (buy):
    * Open a paper YES position at P cents.
When side is "no" (sell):
    * Open a paper NO position at (100 - P) cents.

At each poll cycle all open positions are marked to market using the
latest observed price.  Positions older than ``--max-age-minutes`` are
force-closed at the current price.

Usage
-----
    # Live mode (requires KALSHI_API_KEY):
    python examples/paper_trading.py --ticker SOME-TICKER

    # Offline demo (uses mock snapshot, simulates time advancing):
    python examples/paper_trading.py --mock --interval 2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from smart_money_detection import SmartMoneyDetector
from smart_money_detection.config import load_config
from smart_money_detection.kalshi_client import KalshiClient

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

MOCK_SNAPSHOT = (
    Path(__file__).parent.parent / "tests" / "data" / "sandbox_trades_snapshot.json"
)

# ---------------------------------------------------------------------------
# Portfolio state
# ---------------------------------------------------------------------------

@dataclass
class PaperPosition:
    trade_id: str
    opened_at: datetime
    entry_price: float   # cents
    side: str            # "yes" or "no"
    score: float
    current_price: float = field(init=False)
    closed: bool = False
    close_price: Optional[float] = None

    def __post_init__(self) -> None:
        self.current_price = self.entry_price

    @property
    def pnl_cents(self) -> float:
        p = self.close_price if self.closed else self.current_price
        if self.side == "yes":
            return p - self.entry_price
        else:
            return (100.0 - self.entry_price) - (100.0 - p)

    @property
    def status(self) -> str:
        return "CLOSED" if self.closed else "OPEN"


@dataclass
class PaperPortfolio:
    positions: List[PaperPosition] = field(default_factory=list)
    n_signals_total: int = 0
    n_polls: int = 0

    def open_position(self, trade_id: str, opened_at: datetime, price: float, side: str, score: float) -> None:
        self.positions.append(PaperPosition(
            trade_id=trade_id,
            opened_at=opened_at,
            entry_price=price,
            side=side,
            score=score,
        ))
        self.n_signals_total += 1

    def mark_to_market(self, current_price: float) -> None:
        for pos in self.positions:
            if not pos.closed:
                pos.current_price = current_price

    def close_old(self, current_price: float, max_age: timedelta) -> None:
        now = datetime.now(timezone.utc)
        for pos in self.positions:
            if not pos.closed and (now - pos.opened_at) >= max_age:
                pos.closed = True
                pos.close_price = current_price

    @property
    def open_positions(self) -> List[PaperPosition]:
        return [p for p in self.positions if not p.closed]

    @property
    def closed_positions(self) -> List[PaperPosition]:
        return [p for p in self.positions if p.closed]

    @property
    def total_pnl(self) -> float:
        return sum(p.pnl_cents for p in self.positions)

    @property
    def realised_pnl(self) -> float:
        return sum(p.pnl_cents for p in self.closed_positions)

    @property
    def unrealised_pnl(self) -> float:
        return sum(p.pnl_cents for p in self.open_positions)

    @property
    def win_rate(self) -> float:
        closed = self.closed_positions
        if not closed:
            return float("nan")
        return sum(1 for p in closed if p.pnl_cents > 0) / len(closed) * 100


# ---------------------------------------------------------------------------
# Mock data feed (offline simulation)
# ---------------------------------------------------------------------------

class MockDataFeed:
    """Replays the snapshot, advancing a cursor on each call."""

    def __init__(self, window_size: int = 4) -> None:
        with open(MOCK_SNAPSHOT) as fh:
            records = json.load(fh)
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        if "side" in df.columns and "taker_side" not in df.columns:
            df = df.rename(columns={"side": "taker_side"})
        self.df = df.sort_values("timestamp").reset_index(drop=True)
        self.cursor = 0
        self.window_size = window_size

    def next_batch(self) -> pd.DataFrame:
        start = self.cursor
        end = min(start + self.window_size, len(self.df))
        batch = self.df.iloc[start:end].copy()
        self.cursor = end if end < len(self.df) else 0   # wrap around for demo
        return batch

    def exhausted(self) -> bool:
        return self.cursor >= len(self.df)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _clear_lines(n: int) -> None:
    for _ in range(n):
        sys.stdout.write("\033[F\033[K")


def _print_dashboard(
    portfolio: PaperPortfolio,
    ticker: str,
    current_price: Optional[float],
    poll_interval: int,
    last_signals: List[PaperPosition],
) -> int:
    """Print dashboard and return the number of lines printed."""
    hr = "─" * 70
    lines = []
    lines.append("=" * 70)
    lines.append(f"  Smart Money Paper Trader  |  {ticker}  |  poll every {poll_interval}s")
    lines.append("=" * 70)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    price_str = f"{current_price:.1f}¢" if current_price is not None else "N/A"
    lines.append(f"  Last price : {price_str:<10}  Polls : {portfolio.n_polls}  Time : {ts}")
    lines.append(hr)
    lines.append(
        f"  Signals total : {portfolio.n_signals_total:>5}   "
        f"Open positions : {len(portfolio.open_positions):>4}   "
        f"Closed : {len(portfolio.closed_positions):>4}"
    )
    lines.append(
        f"  Total P&L     : {portfolio.total_pnl:>+8.2f}¢   "
        f"Realised : {portfolio.realised_pnl:>+8.2f}¢   "
        f"Unrealised : {portfolio.unrealised_pnl:>+8.2f}¢"
    )
    win_rate = portfolio.win_rate
    win_str = f"{win_rate:.1f}%" if not np.isnan(win_rate) else "N/A"
    lines.append(f"  Win rate (closed): {win_str}")
    lines.append(hr)

    # Recent signals
    lines.append("  Recent Signals (last 5):")
    if last_signals:
        lines.append(
            f"  {'Trade ID':<16} {'Time':<20} {'Side':<5} "
            f"{'Entry':>6} {'Current':>8} {'P&L':>8} {'Status'}"
        )
        lines.append(f"  {'─' * 66}")
        for pos in last_signals[-5:]:
            ts_str = pos.opened_at.strftime("%Y-%m-%d %H:%M")
            cur = pos.close_price if pos.closed else pos.current_price
            lines.append(
                f"  {str(pos.trade_id):<16} {ts_str:<20} {pos.side:<5} "
                f"{pos.entry_price:>6.1f} {cur:>8.1f} "
                f"{pos.pnl_cents:>+8.2f}¢ {pos.status}"
            )
    else:
        lines.append("  (no signals yet)")

    lines.append("=" * 70)
    lines.append("  Press Ctrl+C to stop")

    for line in lines:
        print(line)
    return len(lines)


# ---------------------------------------------------------------------------
# Core trading loop
# ---------------------------------------------------------------------------

def _run_loop(
    ticker: str,
    detector: SmartMoneyDetector,
    get_new_trades,          # callable() -> pd.DataFrame
    get_current_price,       # callable() -> Optional[float]
    portfolio: PaperPortfolio,
    poll_interval: int,
    max_age: timedelta,
    max_polls: Optional[int] = None,
) -> None:
    """Main polling loop."""
    all_signals: List[PaperPosition] = []
    last_seen_trade_ids: set = set()
    n_dashboard_lines = 0
    current_price: Optional[float] = None

    print(f"\nStarting paper trading loop for '{ticker}'.\n")

    while True:
        try:
            # ---- Fetch new trades ----
            batch = get_new_trades()
            if batch.empty:
                time.sleep(poll_interval)
                portfolio.n_polls += 1
                continue

            # Deduplicate by trade_id
            if "trade_id" in batch.columns:
                new_batch = batch[~batch["trade_id"].isin(last_seen_trade_ids)]
                last_seen_trade_ids.update(batch["trade_id"].tolist())
            else:
                new_batch = batch

            if not new_batch.empty and "price" in new_batch.columns:
                current_price = float(new_batch["price"].iloc[-1])

            # ---- Detect smart money ----
            if not new_batch.empty and len(new_batch) >= 2:
                try:
                    preds = detector.predict(
                        new_batch, volume_col="volume", timestamp_col="timestamp"
                    )
                    scores = detector.score(
                        new_batch, volume_col="volume", timestamp_col="timestamp"
                    )

                    detected_idx = np.where(preds == 1)[0]
                    for idx in detected_idx:
                        row = new_batch.iloc[idx]
                        side_raw = str(row.get("taker_side", "yes")).lower()
                        side = "yes" if side_raw in ("yes", "buy") else "no"
                        portfolio.open_position(
                            trade_id=str(row.get("trade_id", f"t{portfolio.n_signals_total}")),
                            opened_at=row["timestamp"] if hasattr(row["timestamp"], "tzinfo") else datetime.now(timezone.utc),
                            price=float(row["price"]),
                            side=side,
                            score=float(scores[idx]),
                        )
                        all_signals.append(portfolio.open_positions[-1])
                except Exception as exc:
                    logger = logging.getLogger(__name__)
                    logger.debug("Detection skipped this cycle: %s", exc)

            # ---- Mark to market & expire old positions ----
            if current_price is not None:
                portfolio.mark_to_market(current_price)
                portfolio.close_old(current_price, max_age)

            portfolio.n_polls += 1

            # ---- Dashboard ----
            if n_dashboard_lines > 0:
                _clear_lines(n_dashboard_lines)
            n_dashboard_lines = _print_dashboard(
                portfolio, ticker, current_price, poll_interval, all_signals
            )

            if max_polls and portfolio.n_polls >= max_polls:
                print(f"\nReached max polls ({max_polls}). Stopping.")
                break

            time.sleep(poll_interval)

        except KeyboardInterrupt:
            print("\n\nStopped by user.")
            break

    # ---- Final summary ----
    print(f"\n{'=' * 70}")
    print("  Final Paper Trading Summary")
    print(f"{'=' * 70}")
    print(f"  Total polls        : {portfolio.n_polls}")
    print(f"  Total signals      : {portfolio.n_signals_total}")
    print(f"  Closed positions   : {len(portfolio.closed_positions)}")
    print(f"  Realised P&L       : {portfolio.realised_pnl:+.2f} ¢")
    win_rate = portfolio.win_rate
    win_str = f"{win_rate:.1f}%" if not np.isnan(win_rate) else "N/A"
    print(f"  Win rate           : {win_str}")
    print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live paper trading monitor for smart money detection")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--ticker", default=None, help="Kalshi market ticker (live mode)")
    source.add_argument("--mock", action="store_true", help="Use bundled offline snapshot (demo)")
    p.add_argument("--interval", type=int, default=15, help="Poll interval in seconds (default 15)")
    p.add_argument(
        "--max-age-minutes", type=int, default=30,
        help="Auto-close positions older than N minutes (default 30)"
    )
    p.add_argument("--train-limit", type=int, default=500, help="Initial training trade limit (default 500)")
    p.add_argument("--max-polls", type=int, default=None, help="Stop after N polls (useful for testing)")
    p.add_argument(
        "--weighting",
        default="thompson",
        choices=["uniform", "thompson", "ucb", "mwu"],
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    use_mock = args.mock or (not args.ticker and not os.getenv("KALSHI_API_KEY"))

    if use_mock:
        print("Running in OFFLINE DEMO mode (mock snapshot).\n")
        feed = MockDataFeed(window_size=3)
        ticker = "MOCK-MARKET"
        initial_df = feed.df.copy()

        def get_new_trades() -> pd.DataFrame:
            return feed.next_batch()

        def get_current_price() -> Optional[float]:
            return None

    else:
        api_key = os.getenv("KALSHI_API_KEY")
        if not api_key:
            print("ERROR: KALSHI_API_KEY not set. Use --mock for offline demo.")
            return 1
        api_base = os.getenv("KALSHI_API_BASE")

        if args.ticker:
            ticker = args.ticker
        else:
            # Auto-pick first active market
            with KalshiClient(api_key=api_key, api_base=api_base) as client:
                markets = client.get_markets(limit=5)
            if not markets:
                print("ERROR: No active markets found. Specify --ticker.")
                return 1
            ticker = markets[0]["ticker"]
            print(f"No ticker specified — using first active market: {ticker}\n")

        print(f"Fetching initial training data for '{ticker}' …")
        with KalshiClient(api_key=api_key, api_base=api_base) as client:
            initial_df = client.get_trades(ticker, limit=args.train_limit)
        if initial_df.empty:
            print(f"ERROR: No trades returned for '{ticker}'.")
            return 1
        print(f"  Got {len(initial_df):,} trades for initial fit.\n")

        # Track last seen trade time for incremental fetches
        _last_ts: List[Optional[datetime]] = [
            initial_df["timestamp"].max() if not initial_df.empty else None
        ]

        def get_new_trades() -> pd.DataFrame:
            with KalshiClient(api_key=api_key, api_base=api_base) as c:
                df = c.get_trades(ticker, limit=200, min_ts=_last_ts[0])
            if not df.empty and "timestamp" in df.columns:
                _last_ts[0] = df["timestamp"].max()
            return df

        def get_current_price() -> Optional[float]:
            with KalshiClient(api_key=api_key, api_base=api_base) as c:
                market = c.get_market(ticker)
            return float(market.get("yes_price", 0)) if market else None

    # ------------------------------------------------------------------
    # Fit initial detector
    # ------------------------------------------------------------------
    print(f"Fitting initial detector on {len(initial_df):,} trades …")
    config = load_config()
    config.ensemble.weighting_method = args.weighting
    detector = SmartMoneyDetector(config)
    detector.fit(
        initial_df,
        volume_col="volume",
        timestamp_col="timestamp",
        price_col="price",
    )
    print(f"Detector ready  ✓  (weighting={args.weighting})")
    print(f"Starting live paper trading loop (Ctrl+C to stop) …\n")

    portfolio = PaperPortfolio()
    max_age = timedelta(minutes=args.max_age_minutes)

    _run_loop(
        ticker=ticker,
        detector=detector,
        get_new_trades=get_new_trades,
        get_current_price=get_current_price,
        portfolio=portfolio,
        poll_interval=args.interval,
        max_age=max_age,
        max_polls=args.max_polls,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
