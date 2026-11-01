"""Screen Top 50 Momentum Edge Universe from 120 tokens and run Portfolio Backtest.

Applies multi-asset-crypto-momentum skill directives:
1. Anti-Chasing Stretch Gate: |Close - SMA_120| / ATR <= 1.5.
2. Volume Surge Confirmation: Volume > 10-bar median.
3. Max 5 concurrent slots with 1% equity risk per slot.
4. Tie-breaking by volume expansion ratio.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data/universe_4h")
FEE = 0.0005
MAX_SLOTS = 5
RISK_PCT_PER_SLOT = 0.01
LEV_CAP = 3.0
ATR_SL = 2.0
ATR_TP = 4.0
MAX_STRETCH = 1.5


def load_all_indicators() -> dict[str, pd.DataFrame]:
    files = list(DATA_DIR.glob("*.json"))
    universe = {}

    for f in files:
        sym = f.stem
        try:
            with open(f, "r") as fp:
                raw = json.load(fp)
            if len(raw) < 200:
                continue
            df = pd.DataFrame(raw)
            df["dt"] = pd.to_datetime(df["open_time"], unit="ms")
            df = df.sort_values("open_time").reset_index(drop=True)

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

            # Stretch (distance to slow SMA in ATR units)
            df["stretch"] = (close - df["sma_slow"]).abs() / df["atr"]

            # Volume median 10
            df["vol_med"] = df["volume"].rolling(10).median()
            df["vol_ratio"] = df["volume"] / df["vol_med"].replace(0, np.nan)

            # Crossover signals
            above = df["sma_fast"] > df["sma_slow"]
            cross_up = above & ~above.shift(1, fill_value=False)
            cross_dn = ~above & above.shift(1, fill_value=True)

            df["sig"] = 0
            df.loc[cross_up, "sig"] = 1
            df.loc[cross_dn, "sig"] = -1

            # Multi-asset gates:
            # 1. ATR <= 6%
            # 2. Volume > median(10)
            # 3. Stretch <= MAX_STRETCH (don't chase extended breakouts)
            valid = (
                (df["atr_pct"] <= 6.0)
                & (df["volume"] > df["vol_med"])
                & (df["stretch"] <= MAX_STRETCH)
                & df["atr"].notna()
            )
            df.loc[~valid, "sig"] = 0

            universe[sym] = df.set_index("open_time")
        except Exception:
            continue

    return universe


def screen_individual_edge(universe: dict[str, pd.DataFrame]) -> list[dict]:
    """Test each coin individually to rank statistical edge."""
    rankings = []
    for sym, df in universe.items():
        equity = 10_000.0
        trades = []
        pos = None

        for idx in range(len(df)):
            row = df.iloc[idx]
            if pos is not None:
                hit_sl = (row["low"] <= pos["sl"]) if pos["side"] == 1 else (row["high"] >= pos["sl"])
                hit_tp = (row["high"] >= pos["tp"]) if pos["side"] == 1 else (row["low"] <= pos["tp"])
                exit_px = None
                if hit_sl:
                    exit_px, r = pos["sl"], "SL"
                elif hit_tp:
                    exit_px, r = pos["tp"], "TP"
                elif row["sig"] == -pos["side"]:
                    exit_px, r = row["open"], "REV"

                if exit_px is not None:
                    pnl = pos["side"] * (exit_px - pos["entry"]) * pos["size"]
                    fees = (pos["entry"] + exit_px) * pos["size"] * FEE
                    net_pnl = pnl - fees
                    trades.append(net_pnl)
                    pos = None

            if pos is None and row["sig"] != 0 and idx + 1 < len(df) and pd.notna(row["atr"]):
                side = int(row["sig"])
                entry = df.iloc[idx + 1]["open"]
                atr = row["atr"]
                sl = entry - side * ATR_SL * atr
                tp = entry + side * ATR_TP * atr
                risk_dist = ATR_SL * atr
                if risk_dist > 0 and entry > 0:
                    size = min(equity * 0.02 / risk_dist, equity * LEV_CAP / entry)
                    pos = {"side": side, "entry": entry, "sl": sl, "tp": tp, "size": size}

        if len(trades) >= 10:
            pnls = np.array(trades)
            wins = pnls[pnls > 0]
            losses = pnls[pnls <= 0]
            gp = float(wins.sum()) if len(wins) > 0 else 0.0
            gl = abs(float(losses.sum())) if len(losses) > 0 else 0.0
            pf = gp / gl if gl > 0 else 999.0
            wr = len(wins) / len(pnls) * 100
            net = float(pnls.sum())
            rankings.append({
                "symbol": sym,
                "trades": len(pnls),
                "win_rate": wr,
                "profit_factor": pf,
                "net_pnl": net,
            })

    rankings.sort(key=lambda x: (x["profit_factor"], x["net_pnl"]), reverse=True)
    return rankings


def run_portfolio(universe: dict[str, pd.DataFrame], selected_symbols: set[str], initial_equity: float = 10_000.0) -> dict:
    sub_univ = {s: universe[s] for s in selected_symbols if s in universe}
    all_ts = sorted(list(set().union(*[df.index for df in sub_univ.values()])))

    equity = initial_equity
    eq_curve = []
    trades = []
    open_positions = {}

    for i, ts in enumerate(all_ts):
        eq_curve.append(equity)

        # 1. Process exits
        closed = []
        for sym, pos in open_positions.items():
            df = sub_univ[sym]
            if ts not in df.index:
                continue
            bar = df.loc[ts]
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
                    "pnl": net_pnl,
                    "r_mult": r_mult,
                    "reason": reason,
                })
                closed.append(sym)

        for sym in closed:
            del open_positions[sym]

        # 2. Check entries for free slots
        free_slots = MAX_SLOTS - len(open_positions)
        if free_slots <= 0 or equity <= 0:
            continue

        cands = []
        for sym, df in sub_univ.items():
            if sym in open_positions:
                continue
            if ts not in df.index:
                continue
            bar = df.loc[ts]
            if bar["sig"] != 0:
                cands.append({
                    "symbol": sym,
                    "side": int(bar["sig"]),
                    "vol_ratio": float(bar["vol_ratio"]) if pd.notna(bar["vol_ratio"]) else 1.0,
                    "bar": bar,
                })

        if not cands:
            continue

        # Rank candidates by volume surge ratio
        cands.sort(key=lambda x: x["vol_ratio"], reverse=True)
        selected = cands[:free_slots]

        if i + 1 < len(all_ts):
            next_ts = all_ts[i + 1]
            for c in selected:
                sym = c["symbol"]
                df = sub_univ[sym]
                if next_ts not in df.index:
                    continue
                next_bar = df.loc[next_ts]
                entry_px = float(next_bar["open"])
                prev_bar = c["bar"]
                atr = float(prev_bar["atr"])
                side = c["side"]
                risk_dist = ATR_SL * atr

                if risk_dist <= 0 or entry_px <= 0:
                    continue

                risk_usdt = equity * RISK_PCT_PER_SLOT
                raw_size = risk_usdt / risk_dist
                max_size = (equity * LEV_CAP / MAX_SLOTS) / entry_px
                size = min(raw_size, max_size)

                if size * entry_px >= 5.0:
                    open_positions[sym] = {
                        "side": side,
                        "entry": entry_px,
                        "sl": entry_px - side * ATR_SL * atr,
                        "tp": entry_px + side * ATR_TP * atr,
                        "size": size,
                        "risk_usdt": risk_usdt,
                        "dt": next_bar["dt"],
                    }

    # Summary
    df_t = pd.DataFrame(trades)
    pnls = df_t["pnl"].values if len(df_t) > 0 else np.array([])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gp = float(wins.sum()) if len(wins) > 0 else 0.0
    gl = abs(float(losses.sum())) if len(losses) > 0 else 0.0
    pf = round(gp / gl, 2) if gl > 0 else 999.0

    s_eq = pd.Series(eq_curve)
    dd = float((s_eq / s_eq.cummax() - 1).min())

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

    return {
        "n_symbols": len(selected_symbols),
        "total_trades": len(trades),
        "win_rate": round(len(wins) / len(pnls) * 100, 1) if len(pnls) > 0 else 0,
        "profit_factor": pf,
        "expectancy_r": round(float(df_t["r_mult"].mean()), 3) if len(df_t) > 0 else 0,
        "final_equity": round(equity, 2),
        "return_pct": round((equity / initial_equity - 1) * 100, 1),
        "max_drawdown_pct": round(dd * 100, 1),
        "annual": annual,
    }


def main():
    print("Loading 120 tokens with Anti-Chasing (Stretch <= 1.5) & VolMed filters...")
    universe = load_all_indicators()
    print(f"Loaded {len(universe)} tokens.\n")

    print("Screening individual token statistical edge...")
    rankings = screen_individual_edge(universe)
    print(f"Evaluated {len(rankings)} tokens with >=10 trade samples.")

    profitable = [r for r in rankings if r["profit_factor"] >= 1.0]
    print(f"Tokens with positive edge (PF >= 1.0): {len(profitable)} of {len(rankings)} ({len(profitable)/len(rankings)*100:.1f}%)")

    # Select TOP 50 Momentum Coins
    top_50 = rankings[:50]
    top_50_syms = {r["symbol"] for r in top_50}

    print("\n" + "=" * 65)
    print("     TOP 50 MOMENTUM COINS UNIVERSE SELECTION")
    print("=" * 65)
    for i, r in enumerate(top_50[:25], 1):
        print(f" {i:2d}. {r['symbol']:<10s} | {r['trades']:>3d} tr | WR: {r['win_rate']:>5.1f}% | PF: {r['profit_factor']:>5.2f} | PnL: ${r['net_pnl']:>8.2f}")
    print(" ... (showing top 25 of 50) ...")

    # Save top 50 symbols to json
    out_symbols = Path("data/top50_momentum_symbols.json")
    with open(out_symbols, "w") as fp:
        json.dump(sorted(list(top_50_syms)), fp, indent=2)
    print(f"\nSaved Top 50 Symbol Universe -> {out_symbols.resolve()}")

    print("\nRunning Multi-Coin Portfolio Simulation on TOP 50 MOMENTUM COINS...")
    print("Configuration: Max 5 Slots, 1% Equity Risk/Slot, RR 1:2, Anti-Chasing Stretch <= 1.5")
    res = run_portfolio(universe, top_50_syms)

    print("\n" + "=" * 65)
    print("     PORTFOLIO 50-COIN SIMULATION RESULTS")
    print("=" * 65)
    print(f"Universe Size     : {res['n_symbols']} Coins")
    print(f"Total Trades      : {res['total_trades']} trades")
    print(f"Average Trades/Mo : {res['total_trades'] / 65:.1f} trades/month ({(res['total_trades']/65)/4.3:.1f} trades/week)")
    print(f"Win Rate          : {res['win_rate']}%")
    print(f"Profit Factor     : {res['profit_factor']}")
    print(f"Expectancy (R)    : +{res['expectancy_r']} R")
    print(f"Net Return        : +{res['return_pct']}% (Final Equity: ${res['final_equity']:,.2f})")
    print(f"Max Drawdown      : {res['max_drawdown_pct']}%")

    print("\n--- ANNUAL BREAKDOWN ---")
    for yr, d in res["annual"].items():
        print(f"{yr}: {d['trades']:>3d} trades | Win: {d['win_rate']:>5.1f}% | Net PnL: ${d['net_pnl']:>9.2f}")


if __name__ == "__main__":
    main()
