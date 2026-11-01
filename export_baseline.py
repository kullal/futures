"""Export baseline strategy performance metrics and trade logs for AI quantitative review.

Computes:
- Overall metrics (Win rate, Profit Factor, Expectancy in R, Max DD, Sharpe)
- In-Sample (2019-2023) vs Out-of-Sample (2024-2026) split
- Annual performance breakdown
- Side breakdown (LONG vs SHORT)
- Exit reason breakdown (SL, TP, REV)
- Top 10 worst loss trades
- Balance invariant self-check
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

from backtest import backtest, run, FEE, ATR_SL, ATR_TP, LEV, RISK_PCT
from data import load_klines
from strategy import to_df


def compute_comprehensive_metrics(trades: list[dict], eq: pd.Series, initial_equity: float = 10_000.0) -> dict:
    if not trades:
        return {"n_trades": 0}

    df_t = pd.DataFrame(trades)
    df_t["entry_dt"] = pd.to_datetime(df_t["entry_dt"])
    df_t["exit_dt"] = pd.to_datetime(df_t["exit_dt"])
    df_t["year"] = df_t["entry_dt"].dt.year

    pnls = df_t["pnl"].values
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = abs(float(losses.sum())) if len(losses) > 0 else 0.0
    profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 999.0

    # Calculate R-multiples if available, otherwise derive from initial risk
    if "r_mult" in df_t.columns:
        r_vals = df_t["r_mult"].values
        expectancy_r = round(float(np.mean(r_vals)), 3)
    else:
        expectancy_r = None

    # Equity curve stats
    rets = eq.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * (6 * 365) ** 0.5) if rets.std() > 0 else 0.0
    dd = float((eq / eq.cummax() - 1).min())

    # Split IS (2019-2023) vs OOS (2024-2026)
    is_mask = df_t["entry_dt"] < "2024-01-01"
    oos_mask = ~is_mask

    def sub_metrics(sub_df: pd.DataFrame) -> dict:
        if len(sub_df) == 0:
            return {"n_trades": 0}
        sub_pnl = sub_df["pnl"].values
        w = sub_pnl[sub_pnl > 0]
        l = sub_pnl[sub_pnl <= 0]
        gp = float(w.sum()) if len(w) > 0 else 0.0
        gl = abs(float(l.sum())) if len(l) > 0 else 0.0
        pf = round(gp / gl, 2) if gl > 0 else 999.0
        r_exp = round(float(sub_df["r_mult"].mean()), 3) if "r_mult" in sub_df.columns else None
        return {
            "n_trades": len(sub_df),
            "win_rate": round(len(w) / len(sub_df) * 100, 1),
            "net_pnl": round(float(sub_pnl.sum()), 2),
            "profit_factor": pf,
            "expectancy_r": r_exp,
        }

    # Annual breakdown
    annual = {}
    for yr, group in df_t.groupby("year"):
        annual[int(yr)] = sub_metrics(group)

    # Side breakdown
    by_side = {}
    for s, group in df_t.groupby("side"):
        by_side[s] = sub_metrics(group)

    # Exit reason breakdown
    by_reason = {}
    for r, group in df_t.groupby("reason"):
        by_reason[r] = {
            "count": len(group),
            "net_pnl": round(float(group["pnl"].sum()), 2),
            "avg_r": round(float(group["r_mult"].mean()), 3) if "r_mult" in group.columns else None,
        }

    # Top 10 worst losing trades
    worst_losses = df_t.sort_values(by="pnl").head(10)[[
        "entry_dt", "exit_dt", "side", "entry", "exit", "reason", "pnl", "r_mult"
    ]].to_dict(orient="records")
    for row in worst_losses:
        row["entry_dt"] = str(row["entry_dt"])
        row["exit_dt"] = str(row["exit_dt"])

    report = {
        "overall": {
            "initial_equity": initial_equity,
            "final_equity": round(float(eq.iloc[-1]), 2),
            "return_pct": round(float((eq.iloc[-1] / initial_equity - 1) * 100), 1),
            "net_pnl": round(float(pnls.sum()), 2),
            "n_trades": len(pnls),
            "win_rate": round(len(wins) / len(pnls) * 100, 1),
            "profit_factor": profit_factor,
            "expectancy_r": expectancy_r,
            "max_drawdown_pct": round(dd * 100, 1),
            "sharpe": round(sharpe, 2),
            "avg_win_usdt": round(float(wins.mean()), 2) if len(wins) > 0 else 0.0,
            "avg_loss_usdt": round(float(losses.mean()), 2) if len(losses) > 0 else 0.0,
        },
        "split_sample": {
            "in_sample_2019_2023": sub_metrics(df_t[is_mask]),
            "out_of_sample_2024_2026": sub_metrics(df_t[oos_mask]),
        },
        "annual_breakdown": annual,
        "by_side": by_side,
        "by_exit_reason": by_reason,
        "top_worst_losses": worst_losses,
    }
    return report


def backtest_with_r(df: pd.DataFrame) -> tuple[list[dict], pd.Series]:
    """Backtest engine augmented with explicit 1R risk tracking per trade."""
    equity = 10_000.0
    eq_curve = []
    trades = []
    pos = None

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
                net_pnl = pnl - fees
                equity += net_pnl
                r_mult = net_pnl / pos["risk_usdt"] if pos["risk_usdt"] > 0 else 0.0

                trades.append({
                    "entry_dt": str(pos["dt"]),
                    "exit_dt": str(row["dt"]),
                    "side": "LONG" if pos["side"] == 1 else "SHORT",
                    "entry": round(pos["entry"], 2),
                    "exit": round(exit_px, 2),
                    "reason": reason,
                    "pnl": round(net_pnl, 2),
                    "r_mult": round(r_mult, 3),
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
            risk_usdt = risk_dist * size

            pos = {
                "side": side,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "size": size,
                "risk_usdt": risk_usdt,
                "dt": row["dt"],
            }

    return trades, pd.Series(eq_curve, index=df["dt"])


def main():
    print("Running baseline strategy (VMA + ATR + RF) with 1R tracking...")
    from strategy import signals
    df_raw = to_df(load_klines())
    df_sig = signals(
        df_raw,
        fast=5,
        slow=120,
        atr_pct_cap=6.0,
        use_onchain=False,
        use_garch=False,
        use_rf=True,
        rf_thresh=0.5,
        rf_n=3,
    )

    trades, eq = backtest_with_r(df_sig)
    report = compute_comprehensive_metrics(trades, eq, initial_equity=10_000.0)

    # Invariant self-check
    initial_eq = report["overall"]["initial_equity"]
    final_eq = report["overall"]["final_equity"]
    total_pnl = report["overall"]["net_pnl"]
    assert abs((initial_eq + total_pnl) - final_eq) < 1.0, f"Invariant fail: {initial_eq} + {total_pnl} != {final_eq}"

    out_file = Path("data/baseline_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[OK] Baseline report exported -> {out_file.resolve()}")
    print("\n--- SUMMARY BASELINE METRICS ---")
    print(f"Total Trades     : {report['overall']['n_trades']}")
    print(f"Win Rate         : {report['overall']['win_rate']}%")
    print(f"Profit Factor    : {report['overall']['profit_factor']}")
    print(f"Expectancy (R)   : {report['overall']['expectancy_r']} R")
    print(f"Max Drawdown     : {report['overall']['max_drawdown_pct']}%")
    print(f"Sharpe Ratio     : {report['overall']['sharpe']}")
    print(f"Net Return       : {report['overall']['return_pct']}% (${report['overall']['net_pnl']:,.2f})")

    print("\n--- IN-SAMPLE (2019-2023) vs OUT-OF-SAMPLE (2024-2026) ---")
    is_res = report["split_sample"]["in_sample_2019_2023"]
    oos_res = report["split_sample"]["out_of_sample_2024_2026"]
    print(f"In-Sample  (2019-2023): {is_res['n_trades']} trades | Win: {is_res['win_rate']}% | PF: {is_res['profit_factor']} | Exp: {is_res['expectancy_r']}R | PnL: ${is_res['net_pnl']:,.2f}")
    print(f"Out-Sample (2024-2026): {oos_res['n_trades']} trades | Win: {oos_res['win_rate']}% | PF: {oos_res['profit_factor']} | Exp: {oos_res['expectancy_r']}R | PnL: ${oos_res['net_pnl']:,.2f}")

    print("\n--- ANNUAL BREAKDOWN ---")
    for yr, data in report["annual_breakdown"].items():
        print(f"{yr}: {data['n_trades']:>2} trades | Win: {data['win_rate']:>5.1f}% | PF: {data['profit_factor']:>4.2f} | Exp: {data['expectancy_r']:>6.3f}R | PnL: ${data['net_pnl']:>8.2f}")


if __name__ == "__main__":
    main()
