"""Signal layer: VMA cross (paper 05) + ATR volatility filter (papers 01/09).

VMA(fast, slow) on 4H bars: golden cross -> LONG signal, dead cross -> SHORT.
Vol filter: skip signals when ATR% of price exceeds cap (extreme vol = liquidation risk).
On-chain filter (paper 02): suppress LONG if hash rate & active addresses momentum deteriorate,
and suppress SHORT if on-chain momentum is strongly bullish.
GARCH filter (paper 09): skip signals when conditional volatility forecast exceeds percentile threshold.
"""
import numpy as np
import pandas as pd

from garch_vol import attach_garch_vol
from onchain import merge_onchain
from rf_filter import attach_rf_prob


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def signals(
    df: pd.DataFrame,
    fast: int = 5,
    slow: int = 120,
    atr_n: int = 14,
    atr_pct_cap: float | None = 6.0,
    use_onchain: bool = False,
    onchain_thresh: float | tuple[float, float] = 0.5,
    use_garch: bool = False,
    garch_pct: float = 95.0,
    use_rf: bool = True,
    rf_thresh: float = 0.5,
    rf_n: int = 3,
) -> pd.DataFrame:
    """Return df with columns: sig (+1 long, -1 short, 0 none) on cross bars, atr_pct, filtered flag."""
    df = df.copy()
    if "dt" not in df.columns and "open_time" in df.columns:
        df["dt"] = pd.to_datetime(df["open_time"], unit="ms")

    df["sma_fast"] = sma(df["close"], fast)
    df["sma_slow"] = sma(df["close"], slow)
    df["atr"] = atr(df, atr_n)
    df["atr_pct"] = df["atr"] / df["close"] * 100

    above = df["sma_fast"] > df["sma_slow"]
    cross_up = above & ~above.shift(1, fill_value=False)
    cross_dn = ~above & above.shift(1, fill_value=True)

    df["sig"] = 0
    df.loc[cross_up, "sig"] = 1
    df.loc[cross_dn, "sig"] = -1
    df["vma_sig"] = df["sig"]

    # 1. ATR vol filter (paper 01: extreme vol -> liquidation cascade)
    if atr_pct_cap is not None and atr_pct_cap > 0:
        df.loc[df["atr_pct"] > atr_pct_cap, "sig"] = 0

    # 2. On-chain filter (paper 02: Nagula & Alexakis 2022)
    if use_onchain:
        if "hash_rate_z" not in df.columns or "n_unique_addresses_z" not in df.columns:
            df = merge_onchain(df)
        if isinstance(onchain_thresh, tuple):
            long_th, short_th = onchain_thresh
        else:
            th = abs(float(onchain_thresh))
            long_th, short_th = -th, th
        bad_long = (df["hash_rate_z"] < long_th) & (df["n_unique_addresses_z"] < long_th)
        bad_short = (df["hash_rate_z"] > short_th) & (df["n_unique_addresses_z"] > short_th)
        df.loc[bad_long & (df["sig"] == 1), "sig"] = 0
        df.loc[bad_short & (df["sig"] == -1), "sig"] = 0

    # 3. GARCH(1,1) vol filter (paper 09: Ciaian et al. 2018)
    if use_garch:
        if "garch_vol" not in df.columns:
            df = attach_garch_vol(df)
        garch_cap = float(np.nanpercentile(df["garch_vol"], garch_pct))
        df.loc[df["garch_vol"] > garch_cap, "sig"] = 0

    # 4. Random Forest confidence filter (paper 07: Jevtic et al. 2022)
    if use_rf:
        if "rf_prob_long" not in df.columns:
            df = attach_rf_prob(df, n_bars=rf_n)
        has_rf = df["rf_prob_long"].notna()
        bad_long_rf = has_rf & (df["sig"] == 1) & (df["rf_prob_long"] < rf_thresh)
        bad_short_rf = has_rf & (df["sig"] == -1) & (df["rf_prob_long"] > (1.0 - rf_thresh))
        df.loc[bad_long_rf, "sig"] = 0
        df.loc[bad_short_rf, "sig"] = 0

    return df


def to_df(klines: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(klines)
    df["dt"] = pd.to_datetime(df["open_time"], unit="ms")
    return df


if __name__ == "__main__":
    from data import load_klines

    df = to_df(load_klines())
    df = signals(df)
    s = df[df["sig"] != 0]
    print(f"{len(s)} signals from {len(df)} bars")
    print(s[["dt", "close", "sig", "atr_pct"]].tail(10).to_string(index=False))
