"""Multi-Coin Portfolio Backtest Engine (Top 50 USDT-M Universe).

Rules:
1. Strategy: VMA(5, 120) 4H + ATR% <= 6.0% + Volume > 10-bar median.
2. Max open concurrent positions: 5 slots.
3. Ranking when slots contested: Highest volume expansion ratio (vol / vol_med_10).
4. Risk budget: 1% equity per position, SL = 2*ATR, TP = 4*ATR (RR 1:2), Max Leverage 3X.
5. Trading fee: 0.05% taker per side.
"""
import glob
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data/universe_4h")
FEE = 0.0005          # 0.05% taker fee per side
MAX_SLOTS = 5         # User requirement A
RISK_PCT_PER_SLOT = 0.01  # 1% equity risk per trade
LEV_CAP = 3.0
ATR_SL = 2.0
ATR_TP = 4.0


def load_universe_data() -> dict[str, pd.DataFrame]:
    """Load and prepare indicators for all symbols."""
    files = list(DATA_DIR.glob("*.json"))
    universe = {}

    for f in files:
        sym = f.stem
        try:
            with open(f, "r") as fp:
                raw = json.load(fp)
            if len(raw) < 150:
                continue
            df = pd.DataFrame(raw)
            df["dt"] = pd.to_datetime(df["open_time"], unit="ms")
            df = df.sort_values("open_time").reset_index(drop=True)

            # Indicators
            df["sma_fast"] = df["close"].rolling(5).mean()
            df["sma_slow"] = df["close"].rolling(120).mean()

            # ATR 14
            hl = df["high"] - df["low"]
            hc = (df["high"] - df["close"].shift()).abs()
            lc = (df["low"] - df["close"].shift()).abs()
            tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            df["atr"] = tr.rolling(14).mean()
            df["atr_pct"] = df["atr"] / df["close"] * 100

            # Volume median 10
            df["vol_med"] = df["volume"].rolling(10).median()
            df["vol_ratio"] = df["volume"] / df["vol_med"]

            # Crossover signals
            above = df["sma_fast"] > df["sma_slow"]
            cross_up = above & ~above.shift(1, fill_value=False)
            cross_dn = ~above & above.shift(1, fill_value=True)

            df["sig"] = 0
            df.loc[cross_up, "sig"] = 1
            df.loc[cross_dn, "sig"] = -1

            # Filters: ATR cap <= 6% and Volume > median(10)
            valid = (df["atr_pct"] <= 6.0) & (df["volume"] > df["vol_med"]) & (df["atr"].notna())
            df.loc[~valid, "sig"] = 0

            universe[sym] = df.set_index("open_time")
        except Exception:
            continue

    return universe


def run_portfolio_backtest(universe: dict[str, pd.DataFrame], initial_equity: float = 10_000.0) -> dict:
    # Find all common timestamps
    all_timestamps = set()
    for df in universe.values():
        all_timestamps.update(df.index)
    sorted_ts = sorted(list(all_timestamps))

    equity = initial_equity
    eq_curve = []
    trades = []
    open_positions = {}  # sym -> dict(side, entry, sl, tp, size, risk_usdt, dt)

    for i, ts in enumerate(sorted_ts):
        eq_curve.append({"open_time": ts, "equity": equity})

        # 1. Check exits for all open positions on current bar
        closed_syms = []
        for sym, pos in open_positions.items():
            df_sym = universe[sym]
            if ts not in df_sym.index:
                continue
            bar = df_sym.loc[ts]

            hit_sl = (bar["low"] <= pos["sl"]) if pos["side"] == 1 else (bar["high"] >= pos["sl"])
            hit_tp = (bar["high"] >= pos["tp"]) if pos["side"] == 1 else (bar["low"] <= pos["tp"])
            exit_px = None
            reason = None

            if hit_sl:
                exit_px, reason = pos["sl"], "SL"
            elif hit_tp:
                exit_px, reason = pos["tp"], "TP"
            elif bar["sig"] == -pos["side"]:
                exit_px, reason = bar["open"], "REV"

            if exit_px is not None:
                pnl = pos["side"] * (exit_px - pos["entry"]) * pos["size"]
                fee_cost = (pos["entry"] + exit_px) * pos["size"] * FEE
                net_pnl = pnl - fee_cost
                equity += net_pnl
                r_mult = net_pnl / pos["risk_usdt"] if pos["risk_usdt"] > 0 else 0.0

                trades.append({
                    "symbol": sym,
                    "entry_dt": str(pos["dt"]),
                    "exit_dt": str(bar["dt"]),
                    "side": "LONG" if pos["side"] == 1 else "SHORT",
                    "entry": round(pos["entry"], 4),
                    "exit": round(exit_px, 4),
                    "reason": reason,
                    "pnl": round(net_pnl, 2),
                    "r_mult": round(r_mult, 3),
                    "equity": round(equity, 2),
                })
                closed_syms.append(sym)

        for sym in closed_syms:
            del open_positions[sym]

        # 2. Check candidate signals triggering on current bar for entry at NEXT bar
        # If we have free slots available
        free_slots = MAX_SLOTS - len(open_positions)
        if free_slots <= 0 or equity <= 0:
            continue

        candidates = []
        for sym, df_sym in universe.items():
            if sym in open_positions:
                continue  # already holding position in this symbol
            if ts not in df_sym.index:
                continue

            bar = df_sym.loc[ts]
            if bar["sig"] != 0:
                candidates.append({
                    "symbol": sym,
                    "side": int(bar["sig"]),
                    "vol_ratio": float(bar["vol_ratio"]),
                    "bar": bar,
                })

        if not candidates:
            continue

        # Rule B: Rank by Volume Expansion Ratio descending
        candidates.sort(key=lambda x: x["vol_ratio"], reverse=True)
        selected = candidates[:free_slots]

        # 3. Enter selected candidates at NEXT bar open
        if i + 1 < len(sorted_ts):
            next_ts = sorted_ts[i + 1]
            for cand in selected:
                sym = cand["symbol"]
                df_sym = universe[sym]
                if next_ts not in df_sym.index:
                    continue

                entry_bar = df_sym.loc[next_ts]
                entry_px = float(entry_bar["open"])
                prev_bar = cand["bar"]
                atr = float(prev_bar["atr"])
                side = cand["side"]

                sl = entry_px - side * ATR_SL * atr
                tp = entry_px + side * ATR_TP * atr
                risk_dist = ATR_SL * atr

                # Risk sizing: 1% equity per slot, leverage capped at 3X
                risk_usdt = equity * RISK_PCT_PER_SLOT
                raw_size = risk_usdt / risk_dist
                max_size = (equity * LEV_CAP / MAX_SLOTS) / entry_px
                size = min(raw_size, max_size)

                if size * entry_px >= 5.0:  # Binance min notional
                    open_positions[sym] = {
                        "side": side,
                        "entry": entry_px,
                        "sl": sl,
                        "tp": tp,
                        "size": size,
                        "risk_usdt": risk_usdt,
                        "dt": entry_bar["dt"],
                    }

    # Wrap metrics
    df_t = pd.DataFrame(trades)
    df_eq = pd.DataFrame(eq_curve)
    df_eq["dt"] = pd.to_datetime(df_eq["open_time"], unit="ms")
    df_eq = df_eq.set_index("dt")["equity"]

    pnls = df_t["pnl"].values if len(df_t) > 0 else np.array([])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gross_win = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = abs(float(losses.sum())) if len(losses) > 0 else 0.0
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else 999.0

    rets = df_eq.pct_change().dropna()
    sharpe = float(rets.mean() / rets.std() * (6 * 365) ** 0.5) if rets.std() > 0 else 0.0
    dd = float((df_eq / df_eq.cummax() - 1).min())

    # Annual breakdown
    annual = {}
    if len(df_t) > 0:
        df_t["year"] = pd.to_datetime(df_t["entry_dt"]).dt.year
        for yr, grp in df_t.groupby("year"):
            w = grp[grp["pnl"] > 0]
            annual[int(yr)] = {
                "trades": len(grp),
                "win_rate": round(len(w) / len(grp) * 100, 1),
                "net_pnl": round(float(grp["pnl"].sum()), 2),
            }

    # Symbol breakdown (top 10 profitable symbols)
    by_sym = {}
    if len(df_t) > 0:
        for s, grp in df_t.groupby("symbol"):
            by_sym[s] = {
                "trades": len(grp),
                "win_rate": round(len(grp[grp["pnl"] > 0]) / len(grp) * 100, 1),
                "net_pnl": round(float(grp["pnl"].sum()), 2),
            }

    return {
        "initial_equity": initial_equity,
        "final_equity": round(equity, 2),
        "return_pct": round((equity / initial_equity - 1) * 100, 1),
        "net_pnl": round(float(pnls.sum()), 2),
        "total_trades": len(trades),
        "win_rate": round(len(wins) / len(pnls) * 100, 1) if len(pnls) > 0 else 0,
        "profit_factor": pf,
        "expectancy_r": round(float(df_t["r_mult"].mean()), 3) if len(df_t) > 0 else 0,
        "max_drawdown_pct": round(dd * 100, 1),
        "sharpe": round(sharpe, 2),
        "annual": annual,
        "by_symbol": by_sym,
        "trades": trades,
    }


def main():
    print("Loading 4H data for all downloaded universe symbols...")
    universe = load_universe_data()
    print(f"Loaded {len(universe)} symbols with indicators.")

    print("\nRunning Multi-Coin Portfolio Simulation (Max 5 Slots, Vol-Surge Ranking)...")
    res = run_portfolio_backtest(universe)

    print("\n" + "=" * 50)
    print("      PORTFOLIO 50-COIN BACKTEST RESULTS")
    print("=" * 50)
    print(f"Initial Capital   : ${res['initial_equity']:,.2f}")
    print(f"Final Capital     : ${res['final_equity']:,.2f}")
    print(f"Total Net Return  : +{res['return_pct']}% (${res['net_pnl']:,.2f})")
    print(f"Total Trades      : {res['total_trades']} trades")
    print(f"Average Trades/Mo : {res['total_trades'] / 65:.1f} trades/month")
    print(f"Win Rate          : {res['win_rate']}%")
    print(f"Profit Factor     : {res['profit_factor']}")
    print(f"Expectancy (R)    : +{res['expectancy_r']} R")
    print(f"Max Drawdown      : {res['max_drawdown_pct']}%")
    print(f"Sharpe Ratio      : {res['sharpe']}")

    print("\n--- ANNUAL BREAKDOWN ---")
    for yr, d in res["annual"].items():
        print(f"{yr}: {d['trades']:>3d} trades | Win: {d['win_rate']:>5.1f}% | Net PnL: ${d['net_pnl']:>9.2f}")

    # Top 10 coins by profit
    sorted_syms = sorted(res["by_symbol"].items(), key=lambda x: x[1]["net_pnl"], reverse=True)
    print("\n--- TOP 10 CONTRIBUTING COINS ---")
    for s, d in sorted_syms[:10]:
        print(f"{s:<10s}: {d['trades']:>2d} trades | Win: {d['win_rate']:>5.1f}% | PnL: ${d['net_pnl']:>8.2f}")

    out_file = Path("data/portfolio_backtest_report.json")
    with open(out_file, "w") as fp:
        json.dump({k: v for k, v in res.items() if k != "trades"}, fp, indent=2)
    print(f"\nDetailed report saved -> {out_file.resolve()}")


if __name__ == "__main__":
    main()
