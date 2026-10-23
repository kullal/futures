"""Backtest VMA signal strategy on BTC 4H with risk layer (paper 01 constraints).

Rules:
- Enter on signal at next bar open, direction = sig, leverage = LEV (hard cap 3x).
- SL = entry -/+ atr_mult*ATR, TP = entry +/- tp_mult*ATR.
- Risk per trade: RISK_PCT of equity; position size = risk / SL distance, capped by leverage.
- Exit on SL/TP (intrabar via high/low, SL first = conservative) or opposite signal.
- Taker fee per side.
"""
import json

import pandas as pd

from data import load_klines
from strategy import signals, to_df

FEE = 0.0005          # taker 0.05%
LEV = 3.0             # paper 01: <=3X long, <=5X short keeps liquidation prob ~1%
RISK_PCT = 0.02       # risk 2% equity per trade
ATR_SL = 2.0
ATR_TP = 4.0          # RR 1:2


def backtest(df: pd.DataFrame) -> tuple[list[dict], pd.Series]:
    equity = 10_000.0
    eq_curve = []
    trades = []
    pos = None  # dict(side, entry, sl, tp, size, i_entry)

    for i in range(len(df)):
        row = df.iloc[i]
        eq_curve.append(equity)

        if pos is not None:
            hit_sl = (row["low"] <= pos["sl"]) if pos["side"] == 1 else (row["high"] >= pos["sl"])
            hit_tp = (row["high"] >= pos["tp"]) if pos["side"] == 1 else (row["low"] <= pos["tp"])
            exit_px = None
            if hit_sl:
                exit_px, reason = pos["sl"], "SL"
            elif hit_tp:
                exit_px, reason = pos["tp"], "TP"
            elif row["sig"] == -pos["side"]:
                exit_px, reason = row["open"], "REV"
            if exit_px is not None:
                pnl = pos["side"] * (exit_px - pos["entry"]) * pos["size"]
                fees = (pos["entry"] + exit_px) * pos["size"] * FEE
                equity += pnl - fees
                trades.append({
                    "entry_dt": str(pos["dt"]), "exit_dt": str(row["dt"]),
                    "side": "LONG" if pos["side"] == 1 else "SHORT",
                    "entry": pos["entry"], "exit": exit_px, "reason": reason,
                    "pnl": round(pnl - fees, 2),
                    "equity": round(equity, 2),
                })
                pos = None

        if pos is None and row["sig"] != 0 and not pd.isna(row["atr"]) and equity > 0:
            atr = row["atr"]
            side = int(row["sig"])
            entry = df.iloc[i + 1]["open"] if i + 1 < len(df) else row["close"]
            sl = entry - side * ATR_SL * atr
            tp = entry + side * ATR_TP * atr
            risk_dist = ATR_SL * atr
            size = min(equity * RISK_PCT / risk_dist, equity * LEV / entry)
            pos = {"side": side, "entry": entry, "sl": sl, "tp": tp,
                   "size": size, "dt": row["dt"]}

    return trades, pd.Series(eq_curve, index=df["dt"])


def metrics(trades: list[dict], eq: pd.Series) -> dict:
    if not trades:
        return {"n_trades": 0}
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    rets = eq.pct_change().dropna()
    # 4H bars: 6/day -> annualize sqrt(6*365)
    sharpe = rets.mean() / rets.std() * (6 * 365) ** 0.5 if rets.std() > 0 else 0
    dd = (eq / eq.cummax() - 1).min()
    return {
        "n_trades": len(pnls),
        "win_rate": round(len(wins) / len(pnls) * 100, 1),
        "total_pnl": round(sum(pnls), 2),
        "final_equity": round(eq.iloc[-1], 2),
        "return_pct": round((eq.iloc[-1] / eq.iloc[0] - 1) * 100, 1),
        "max_drawdown_pct": round(dd * 100, 1),
        "sharpe": round(sharpe, 2),
        "avg_win": round(sum(wins) / len(wins), 2) if wins else 0,
        "avg_loss": round(sum(p for p in pnls if p <= 0) / max(1, len(pnls) - len(wins)), 2),
        "exit_reasons": {r: sum(1 for t in trades if t["reason"] == r) for r in ("SL", "TP", "REV")},
    }


def run(
    fast: int = 5,
    slow: int = 120,
    atr_cap: float = 6.0,
    use_onchain: bool = False,
    onchain_thresh: float = 0.5,
    use_garch: bool = False,
    garch_pct: float = 95.0,
    use_rf: bool = True,
    rf_thresh: float = 0.5,
    rf_n: int = 3,
    df: pd.DataFrame | None = None,
):
    if df is None:
        df = to_df(load_klines())
    df = signals(
        df,
        fast=fast,
        slow=slow,
        atr_pct_cap=atr_cap,
        use_onchain=use_onchain,
        onchain_thresh=onchain_thresh,
        use_garch=use_garch,
        garch_pct=garch_pct,
        use_rf=use_rf,
        rf_thresh=rf_thresh,
        rf_n=rf_n,
    )
    trades, eq = backtest(df)
    return trades, eq, df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest VMA strategy with on-chain, GARCH, & RF filters.")
    parser.add_argument("--onchain", action="store_true", help="Enable on-chain filter")
    parser.add_argument("--onchain-thresh", type=float, default=0.5, help="On-chain z-score threshold")
    parser.add_argument("--garch", action="store_true", help="Enable GARCH(1,1) volatility filter")
    parser.add_argument("--garch-pct", type=float, default=95.0, help="GARCH volatility percentile cap")
    parser.add_argument("--rf", dest="rf", action="store_true", default=True, help="Enable Random Forest confidence filter (default: True)")
    parser.add_argument("--no-rf", dest="rf", action="store_false", help="Disable Random Forest confidence filter")
    parser.add_argument("--rf-thresh", type=float, default=0.5, help="RF probability threshold")
    parser.add_argument("--rf-n", type=int, default=3, help="RF forward return prediction horizon (bars)")
    parser.add_argument("--both", action="store_true", help="Enable both on-chain and GARCH filters")
    args = parser.parse_args()

    use_onchain = args.onchain or args.both
    use_garch = args.garch or args.both
    use_rf = args.rf

    trades, eq, df = run(
        use_onchain=use_onchain,
        onchain_thresh=args.onchain_thresh,
        use_garch=use_garch,
        garch_pct=args.garch_pct,
        use_rf=use_rf,
        rf_thresh=args.rf_thresh,
        rf_n=args.rf_n,
    )
    m = metrics(trades, eq)
    labels = []
    if use_onchain:
        labels.append("Onchain")
    if use_garch:
        labels.append("GARCH")
    if use_rf:
        labels.append("RF")
    label = "Baseline" + (" + " + " + ".join(labels) if labels else "")

    print(f"[{label}] Period: {df['dt'].iloc[0].date()} -> {df['dt'].iloc[-1].date()} ({len(df)} bars)")
    print(json.dumps(m, indent=2))
    out = "backtest_trades.json"
    with open(out, "w") as f:
        json.dump(trades, f, indent=1)
    print(f"\nTrades saved -> {out}")
    # self-check
    assert all(t["side"] in ("LONG", "SHORT") for t in trades)
    assert eq.iloc[0] > 0
