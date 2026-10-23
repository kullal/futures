"""Compare backtest performance across 6 variants:
1. Baseline (ATR only)
2. +Onchain filter (Nagula & Alexakis 2022)
3. +GARCH filter (Ciaian et al. 2018)
4. Both (+Onchain +GARCH)
5. +RF filter (Jevtic et al. 2022)
6. +Onchain +RF
"""
import json
import pandas as pd

from backtest import backtest, metrics
from data import load_klines
from garch_vol import attach_garch_vol
from onchain import merge_onchain
from rf_filter import attach_rf_prob
from strategy import signals, to_df


def run_comparison(
    onchain_thresh: float = 0.5,
    garch_pct: float = 95.0,
    rf_thresh: float = 0.5,
    rf_n: int = 3,
) -> list[dict]:
    # Pre-load and augment base df once for speed
    df_raw = to_df(load_klines())
    df_aug = merge_onchain(df_raw)
    df_aug = attach_garch_vol(df_aug)
    df_aug = attach_rf_prob(df_aug, n_bars=rf_n)

    variants = [
        ("Baseline (ATR only)", False, False, False),
        ("+Onchain filter", True, False, False),
        ("+GARCH filter", False, True, False),
        ("+Onchain +GARCH", True, True, False),
        ("+RF filter", False, False, True),
        ("+Onchain +RF", True, False, True),
    ]

    results = []
    for name, use_oc, use_garch, use_rf in variants:
        df_sig = signals(
            df_aug,
            atr_pct_cap=6.0,
            use_onchain=use_oc,
            onchain_thresh=onchain_thresh,
            use_garch=use_garch,
            garch_pct=garch_pct,
            use_rf=use_rf,
            rf_thresh=rf_thresh,
            rf_n=rf_n,
        )
        trades, eq = backtest(df_sig)
        m = metrics(trades, eq)
        results.append({
            "name": name,
            "n_trades": int(m["n_trades"]),
            "win_rate": float(m["win_rate"]),
            "return_pct": float(m["return_pct"]),
            "max_drawdown_pct": float(m["max_drawdown_pct"]),
            "sharpe": float(m["sharpe"]),
        })

    return results


def print_table(results: list[dict]):
    header = f"{'Variant':<25} | {'Trades':>6} | {'Win Rate':>8} | {'Return':>8} | {'Max DD':>8} | {'Sharpe':>6}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(f"{r['name']:<25} | {r['n_trades']:>6} | {r['win_rate']:>7.1f}% | {r['return_pct']:>7.1f}% | {r['max_drawdown_pct']:>7.1f}% | {r['sharpe']:>6.2f}")
    print(sep)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare backtest variants.")
    parser.add_argument("--onchain-thresh", type=float, default=0.5, help="On-chain threshold")
    parser.add_argument("--garch-pct", type=float, default=95.0, help="GARCH percentile threshold")
    parser.add_argument("--rf-thresh", type=float, default=0.5, help="RF probability threshold")
    parser.add_argument("--rf-n", type=int, default=3, help="RF forward return prediction horizon (bars)")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    res = run_comparison(
        onchain_thresh=args.onchain_thresh,
        garch_pct=args.garch_pct,
        rf_thresh=args.rf_thresh,
        rf_n=args.rf_n,
    )
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print_table(res)
