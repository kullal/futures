"""Live signal scanner: check last CLOSED 4H bar for VMA cross with RF confidence filter. Run via cron at candle close (:00/04/08/12/16/20 UTC + a minute)."""
import argparse
import datetime
import pandas as pd

from data import load_klines
from notifier import send_telegram
from strategy import signals, to_df

_last_rejection = None
_rejection_data = None


def format_alert(sig: dict) -> str:
    """Format signal alert untuk Telegram."""
    emoji = "🟢" if sig["side"] == "LONG" else "🔴"
    prob_line = f"<b>RF Prob Long:</b> {sig['rf_prob_long']:.4f}\n" if sig.get("rf_prob_long") is not None else ""
    return (
        f"{emoji} <b>BTCUSDT {sig['side']} SIGNAL</b>\n\n"
        f"<b>Waktu Bar:</b> {sig['time']} UTC\n"
        f"<b>Entry:</b> ${sig['entry']:,.2f}\n"
        f"<b>Stop Loss (SL):</b> ${sig['sl']:,.2f}\n"
        f"<b>Take Profit (TP):</b> ${sig['tp']:,.2f}\n"
        f"<b>ATR%:</b> {sig['atr_pct']}%\n"
        f"<b>Max Leverage:</b> {sig['leverage_max']}X\n"
        f"{prob_line}\n"
        f"<i>Disclaimer: Signal riset, bukan ajakan finansial.</i>"
    )


def format_rejection(side: str, prob_str: str, reason: str = "Filter RF tidak mengonfirmasi arah tren") -> str:
    """Format pesan penolakan VMA cross oleh RF untuk Telegram."""
    return (
        f"⚠️ <b>BTCUSDT VMA Cross Ditolak RF</b>\n\n"
        f"<b>Arah VMA:</b> {side}\n"
        f"<b>RF Prob:</b> {prob_str}\n"
        f"<b>Alasan Skip:</b> {reason}\n\n"
        f"<i>Disclaimer: Signal riset, bukan ajakan finansial.</i>"
    )


def scan(threshold: float = 0.5) -> dict | None:
    """Check last CLOSED 4H bar. Release signal only if RF confirms direction."""
    global _last_rejection, _rejection_data
    _last_rejection = None
    _rejection_data = None

    df = signals(to_df(load_klines()), use_rf=True, rf_n=3, rf_thresh=threshold)
    # last row may be the still-open candle; use last closed bar
    now_ms = datetime.datetime.now(datetime.UTC).timestamp() * 1000
    closed = df[df["open_time"] + 4 * 3600 * 1000 <= now_ms]
    bar = closed.iloc[-1]

    prob = bar.get("rf_prob_long", float("nan"))
    vma_sig = bar.get("vma_sig", bar["sig"])

    if bar["sig"] != 0:
        side = "LONG" if bar["sig"] == 1 else "SHORT"
        s = 1 if bar["sig"] == 1 else -1
        entry = bar["close"]
        return {
            "time": str(bar["dt"]),
            "side": side,
            "entry": round(float(entry), 2),
            "sl": round(float(entry - s * 2 * bar["atr"]), 2),
            "tp": round(float(entry + s * 4 * bar["atr"]), 2),
            "atr_pct": round(float(bar["atr_pct"]), 2),
            "rf_prob_long": round(float(prob), 4) if pd.notna(prob) else None,
            "leverage_max": 3,
        }

    # If VMA cross occurred but RF filter suppressed it (and ATR filter was satisfied)
    if vma_sig != 0 and (pd.isna(bar.get("atr_pct")) or bar["atr_pct"] <= 6.0):
        side = "LONG" if vma_sig == 1 else "SHORT"
        prob_str = f"{prob:.4f}" if pd.notna(prob) else "NaN"
        _last_rejection = f"signal VMA {side} ditolak RF (prob={prob_str})"
        _rejection_data = {
            "side": side,
            "prob": prob_str,
            "reason": f"Probabilitas RF ({prob_str}) tidak konfirmasi arah {side}",
            "time": str(bar["dt"]),
        }
        print(_last_rejection)
        return None

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live 4H signal scanner BTCUSDT")
    parser.add_argument("--notify", action="store_true", help="Kirim notifikasi Telegram jika ada signal/penolakan")
    args = parser.parse_args()

    sig = scan()
    if sig:
        print(sig)
        if args.notify:
            send_telegram(format_alert(sig))
    elif _last_rejection:
        if args.notify and _rejection_data:
            send_telegram(
                format_rejection(
                    side=_rejection_data["side"],
                    prob_str=_rejection_data["prob"],
                    reason=_rejection_data["reason"],
                )
            )
    else:
        print(f"No signal on last closed 4H bar ({datetime.datetime.now(datetime.UTC):%Y-%m-%d %H:%M} UTC)")
