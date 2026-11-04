"""Live Multi-Coin 4H Signal Scanner (Top 50 Momentum Universe).

Evaluates 50 vetted high-momentum tokens at candle close:
- VMA(5, 120) Crossover (Long & Short)
- Anti-Chasing Gate: Stretch (|Close - SMA_120| / ATR) <= 1.5
- Volume Surge Confirmation: Volume > 10-bar median
- Volatility Cap: ATR% <= 6.0%
- Max 5 concurrent slots on Binance Testnet with 1% equity risk per slot.
"""
import argparse
import datetime
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import requests

from notifier import send_telegram
from executor_testnet import BinanceTestnet, execute_signal, load_config

SYMBOLS_FILE = Path("data/top50_momentum_symbols.json")
MAX_SLOTS = 5
BASE_URL = "https://data-api.binance.vision"


def load_symbols() -> list[str]:
    if not SYMBOLS_FILE.exists():
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "LINKUSDT"]
    with open(SYMBOLS_FILE, "r") as f:
        return json.load(f)


def fetch_recent_klines(symbol: str, session: requests.Session, limit: int = 140) -> list[dict]:
    url = f"{BASE_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": "4h", "limit": limit}
    try:
        r = session.get(url, params=params, timeout=8)
        if r.status_code != 200:
            return []
        data = r.json()
        if not data or not isinstance(data, list):
            return []
        klines = []
        for b in data:
            klines.append({
                "open_time": int(b[0]),
                "open": float(b[1]),
                "high": float(b[2]),
                "low": float(b[3]),
                "close": float(b[4]),
                "volume": float(b[5]),
                "close_time": int(b[6]),
            })
        return klines
    except Exception:
        return []


def analyze_symbol(symbol: str, klines: list[dict], now_ms: float) -> dict | None:
    if len(klines) < 125:
        return None

    df = pd.DataFrame(klines).sort_values("open_time").reset_index(drop=True)
    close = df["close"]
    df["sma_fast"] = close.rolling(5).mean()
    df["sma_slow"] = close.rolling(120).mean()

    # ATR 14
    hl = df["high"] - df["low"]
    hc = (df["high"] - close.shift()).abs()
    lc = (df["low"] - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr"] / close * 100
    df["stretch"] = (close - df["sma_slow"]).abs() / df["atr"]

    # Volume median 10
    df["vol_med"] = df["volume"].rolling(10).median()
    df["vol_ratio"] = df["volume"] / df["vol_med"].replace(0, np.nan)

    # Crossover detection
    above = df["sma_fast"] > df["sma_slow"]
    cross_up = above & ~above.shift(1, fill_value=False)
    cross_dn = ~above & above.shift(1, fill_value=True)

    df["sig"] = 0
    df.loc[cross_up, "sig"] = 1
    df.loc[cross_dn, "sig"] = -1

    # Only look at the LAST CLOSED candle
    closed = df[df["open_time"] + 4 * 3600 * 1000 <= now_ms]
    if len(closed) == 0:
        return None
    bar = closed.iloc[-1]

    raw_sig = int(bar["sig"])
    if raw_sig == 0:
        return None

    # Apply quality gates
    side = "LONG" if raw_sig == 1 else "SHORT"
    entry_est = float(bar["close"])
    atr = float(bar["atr"])
    atr_pct = float(bar["atr_pct"])
    stretch = float(bar["stretch"])
    vol_ratio = float(bar["vol_ratio"]) if pd.notna(bar["vol_ratio"]) else 1.0

    # Gate 1: Volatility cap
    if atr_pct > 6.0:
        return {"rejected": True, "symbol": symbol, "reason": f"ATR% ({atr_pct:.1f}%) > 6.0%"}

    # Gate 2: Volume confirmation
    if bar["volume"] <= bar["vol_med"]:
        return {"rejected": True, "symbol": symbol, "reason": f"Volume below 10-bar median (ratio {vol_ratio:.2f})"}

    # Gate 3: Anti-Chasing Stretch Gate
    if stretch > 1.5:
        return {"rejected": True, "symbol": symbol, "reason": f"Overextended stretch ({stretch:.2f} ATR > 1.5)"}

    # Passed all filters!
    sl = entry_est - raw_sig * 2.0 * atr
    tp = entry_est + raw_sig * 4.0 * atr

    return {
        "symbol": symbol,
        "side": side,
        "entry": entry_est,
        "sl": sl,
        "tp": tp,
        "atr": atr,
        "atr_pct": atr_pct,
        "stretch": stretch,
        "vol_ratio": vol_ratio,
        "bar_time": datetime.datetime.fromtimestamp(bar["open_time"] / 1000, datetime.UTC).strftime("%Y-%m-%d %H:%M"),
    }


def format_multicoin_alert(sig: dict, exec_res: dict | None = None) -> str:
    emoji = "🟢" if sig["side"] == "LONG" else "🔴"
    exec_line = ""
    if exec_res:
        st = exec_res.get("status", "N/A")
        if st == "EXECUTED":
            det = exec_res.get("details", {})
            size = det.get("size", "")
            exec_line = f"\n🤖 <b>Testnet Execution:</b> EXECUTED ({size} {sig['symbol']})\n"
        elif st == "FAILED":
            err = exec_res.get("error", "Unknown error")
            exec_line = f"\n⚠️ <b>Testnet Error:</b> {err}\n"
        elif st == "SKIPPED_EXISTING_POSITION":
            exec_line = f"\n⚠️ <b>Testnet:</b> Posisi {sig['side']} sudah aktif sebelumnya\n"
        elif st == "ABORTED":
            exec_line = f"\n⚠️ <b>Testnet Aborted:</b> {exec_res.get('message', 'Not executed')}\n"
        else:
            exec_line = f"\n🤖 <b>Testnet Status:</b> {st}\n"

    return (
        f"{emoji} <b>{sig['symbol']} {sig['side']} SIGNAL (Multi-Coin 4H)</b>\n\n"
        f"<b>Waktu Bar:</b> {sig['bar_time']} UTC\n"
        f"<b>Entry:</b> ${sig['entry']:,.4f}\n"
        f"<b>Stop Loss:</b> ${sig['sl']:,.4f}\n"
        f"<b>Take Profit:</b> ${sig['tp']:,.4f}\n"
        f"<b>Volume Surge:</b> {sig['vol_ratio']:.2f}x\n"
        f"<b>Stretch:</b> {sig['stretch']:.2f} ATR\n"
        f"<b>ATR%:</b> {sig['atr_pct']:.2f}%\n"
        f"{exec_line}\n"
        f"<i>Top 50 Momentum Strategy • Risk 1% per slot</i>"
    )


def scan_multicoin(dry_run: bool = False, execute: bool = False, notify: bool = False):
    symbols = load_symbols()
    print(f"Scanning {len(symbols)} Top Momentum coins...")

    session = requests.Session()
    now_ms = datetime.datetime.now(datetime.UTC).timestamp() * 1000
    valid_signals = []
    rejections = []

    for idx, sym in enumerate(symbols, 1):
        klines = fetch_recent_klines(sym, session, limit=140)
        res = analyze_symbol(sym, klines, now_ms)
        if res:
            if res.get("rejected"):
                rejections.append(res)
            else:
                valid_signals.append(res)
        time.sleep(0.03)

    print(f"Scan complete: {len(valid_signals)} signals found, {len(rejections)} rejected by gates.")

    if not valid_signals:
        print(f"No valid signals on last closed 4H candle ({datetime.datetime.now(datetime.UTC):%Y-%m-%d %H:%M} UTC).")
        return

    # Check active positions to manage slots
    cfg = load_config()
    client = BinanceTestnet(cfg.get("api_key", ""), cfg.get("api_secret", ""), proxy=cfg.get("proxy", ""))
    active_positions = []
    try:
        active_positions = client.get_all_open_positions()
    except Exception as e:
        print(f"Warning: Could not fetch active positions: {e}")

    active_symbols = {p["symbol"] for p in active_positions}
    available_slots = max(0, MAX_SLOTS - len(active_positions))
    print(f"Active Positions ({len(active_positions)}/{MAX_SLOTS}): {list(active_symbols)}")
    print(f"Available Slots: {available_slots}")

    # Filter out symbols already holding an open position
    fresh_candidates = [s for s in valid_signals if s["symbol"] not in active_symbols]
    # Tie-breaking by volume expansion ratio
    fresh_candidates.sort(key=lambda x: x["vol_ratio"], reverse=True)

    selected = fresh_candidates[:available_slots]
    queued = fresh_candidates[available_slots:]

    for sig in selected:
        print(f"\n[TRIGGER] {sig['symbol']} {sig['side']} @ ${sig['entry']} (Surge: {sig['vol_ratio']:.2f}x)")
        exec_res = None
        if execute:
            try:
                exec_res = execute_signal(sig, dry_run=dry_run)
                print(f"  -> Execution result: {exec_res.get('status')}")
            except Exception as e:
                print(f"  -> Execution failed: {e}")
                exec_res = {"status": "FAILED", "error": str(e)}

        if notify:
            msg = format_multicoin_alert(sig, exec_res=exec_res)
            send_telegram(msg)

    if queued:
        print(f"\n[QUEUED/SKIPPED] {len(queued)} signals skipped due to max 5 slot cap:")
        for q in queued:
            print(f"  - {q['symbol']} {q['side']} (Surge: {q['vol_ratio']:.2f}x)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Coin 4H Signal Scanner & Executor")
    parser.add_argument("--execute-testnet", action="store_true", help="Execute valid signals to Binance Testnet")
    parser.add_argument("--notify", action="store_true", help="Send alerts to Telegram")
    parser.add_argument("--dry-run", action="store_true", help="Simulate sizing without real API calls")
    args = parser.parse_args()

    scan_multicoin(
        dry_run=args.dry_run,
        execute=args.execute_testnet,
        notify=args.notify,
    )
