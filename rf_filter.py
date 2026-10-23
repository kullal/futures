"""Random Forest Confidence Filter for BTC 4H bars (Paper 07: Jevtic et al. 2022).

Predicts directional return N bars ahead (default N=3, ~12 hours on 4H timeframe)
using past technical indicators and on-chain metrics.
Walk-forward expanding window ensures zero look-ahead bias.
"""
import hashlib
import json
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from onchain import merge_onchain

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data")
CACHE_FILE = os.path.join(CACHE_DIR, "rf_cache.json")


def _compute_param_hash(params: dict) -> str:
    serialized = json.dumps(params, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Standard RSI with Wilder's exponential smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    total = avg_gain + avg_loss
    return pd.Series(
        np.where(total == 0.0, 50.0, 100.0 * avg_gain / total),
        index=close.index,
        name=f"rsi_{period}",
    )


def compute_rf_features(
    df: pd.DataFrame,
    n_bars: int = 3,
    fast: int = 5,
    slow: int = 120,
    atr_n: int = 14,
) -> tuple[pd.DataFrame, list[str]]:
    """Compute features strictly from past data (<= bar t)."""
    df_feat = df.copy()

    # Ensure on-chain features are present
    if "hash_rate_z" not in df_feat.columns or "n_unique_addresses_z" not in df_feat.columns:
        df_feat = merge_onchain(df_feat)

    close = df_feat["close"]

    # 1. Log returns over 1, 3, 6, 12 bars
    df_feat["log_ret_1"] = np.log(close / close.shift(1))
    df_feat["log_ret_3"] = np.log(close / close.shift(3))
    df_feat["log_ret_6"] = np.log(close / close.shift(6))
    df_feat["log_ret_12"] = np.log(close / close.shift(12))

    # 2. RSI(14)
    df_feat["rsi_14"] = compute_rsi(close, period=14)

    # 3. Ratio close / sma_fast & close / sma_slow
    sma_fast = close.rolling(fast).mean() if "sma_fast" not in df_feat.columns else df_feat["sma_fast"]
    sma_slow = close.rolling(slow).mean() if "sma_slow" not in df_feat.columns else df_feat["sma_slow"]
    df_feat["ratio_sma_fast"] = close / sma_fast
    df_feat["ratio_sma_slow"] = close / sma_slow

    # 4. ATR% of price
    if "atr_pct" in df_feat.columns:
        df_feat["atr_pct"] = df_feat["atr_pct"]
    else:
        hl = df_feat["high"] - df_feat["low"]
        hc = (df_feat["high"] - close.shift(1)).abs()
        lc = (df_feat["low"] - close.shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr_val = tr.rolling(atr_n).mean()
        df_feat["atr_pct"] = atr_val / close * 100.0

    # 5. Volume z-score (rolling 100)
    vol_roll = df_feat["volume"].rolling(100, min_periods=20)
    vol_std = vol_roll.std().replace(0.0, np.nan)
    df_feat["vol_z_100"] = (df_feat["volume"] - vol_roll.mean()) / vol_std

    # 6. High-low range over past N bars (normalized by close)
    df_feat["hl_range_n"] = (df_feat["high"].rolling(n_bars).max() - df_feat["low"].rolling(n_bars).min()) / close

    feature_cols = [
        "log_ret_1",
        "log_ret_3",
        "log_ret_6",
        "log_ret_12",
        "rsi_14",
        "ratio_sma_fast",
        "ratio_sma_slow",
        "atr_pct",
        "vol_z_100",
        "hl_range_n",
        "hash_rate_z",
        "n_unique_addresses_z",
    ]
    return df_feat, feature_cols


def compute_rf_prob(
    df: pd.DataFrame,
    n_bars: int = 3,
    min_train: int = 2000,
    retrain_every: int = 500,
    n_estimators: int = 100,
    max_depth: int = 5,
    min_samples_leaf: int = 20,
    random_state: int = 42,
    use_cache: bool = True,
    force: bool = False,
) -> pd.Series:
    """Walk-forward expanding window estimation of P(close[t+N] > close[t]).

    Strictly lookahead-free:
    When predicting bar t in chunk [chunk_start, chunk_end):
    The training set only contains samples i where i + n_bars < chunk_start.
    Thus, every label in training was determined strictly before chunk_start.
    Bars before min_train return NaN (unfiltered).
    Caches calculated probabilities to data/rf_cache.json keyed by open_time.
    """
    df_feat, feature_cols = compute_rf_features(df, n_bars=n_bars)

    # Target label: 1 if close N bars ahead is higher, else 0
    y_series = (df_feat["close"].shift(-n_bars) > df_feat["close"]).astype(int)

    n_total = len(df_feat)
    rf_probs = pd.Series(np.nan, index=df_feat.index, dtype=float, name="rf_prob_long")

    params = {
        "n_bars": n_bars,
        "min_train": min_train,
        "retrain_every": retrain_every,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "random_state": random_state,
    }
    param_hash = _compute_param_hash(params)

    cached_probs: dict[str, float] = {}
    if use_cache and not force and os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                cdata = json.load(f)
            if cdata.get("param_hash") == param_hash:
                if "probs" in cdata and isinstance(cdata["probs"], dict):
                    cached_probs = {str(k): float(v) for k, v in cdata["probs"].items()}
                else:
                    cached_probs = {
                        str(k): float(v)
                        for k, v in cdata.items()
                        if k not in ("param_hash", "params", "probs")
                    }
        except Exception:
            cached_probs = {}

    if "open_time" in df_feat.columns:
        open_time_keys = [str(x) for x in df_feat["open_time"]]
    elif "dt" in df_feat.columns:
        open_time_keys = [str(x) for x in df_feat["dt"]]
    else:
        open_time_keys = [str(x) for x in df_feat.index]

    # Burn-in for rolling indicators (sma_slow=120, vol_z_100=100)
    burn_in = 120
    cache_updated = False

    for chunk_start in range(min_train, n_total, retrain_every):
        chunk_end = min(chunk_start + retrain_every, n_total)
        chunk_keys = open_time_keys[chunk_start:chunk_end]

        if all(k in cached_probs for k in chunk_keys):
            for idx, k in enumerate(chunk_keys):
                rf_probs.iloc[chunk_start + idx] = cached_probs[k]
            continue

        # Label of bar i uses close[i + n_bars].
        # For zero lookahead, we require i + n_bars < chunk_start, so i <= chunk_start - n_bars - 1.
        train_end = chunk_start - n_bars - 1
        if train_end <= burn_in:
            continue

        X_train = df_feat.loc[burn_in:train_end, feature_cols]
        y_train = y_series.loc[burn_in:train_end]

        # Valid rows only
        valid = ~X_train.isna().any(axis=1) & ~y_train.isna()
        X_tr = X_train.loc[valid]
        y_tr = y_train.loc[valid]

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
        clf.fit(X_tr, y_tr)

        idx_1 = list(clf.classes_).index(1) if 1 in clf.classes_ else 0

        X_test = df_feat.loc[chunk_start : chunk_end - 1, feature_cols].fillna(0.0)
        pred_probs = clf.predict_proba(X_test)[:, idx_1]
        rf_probs.iloc[chunk_start:chunk_end] = pred_probs

        for idx, k in enumerate(chunk_keys):
            cached_probs[k] = float(pred_probs[idx])
        cache_updated = True

    if use_cache and (cache_updated or force):
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_payload = {
            "param_hash": param_hash,
            "params": params,
            "probs": cached_probs,
            **cached_probs,
        }
        with open(CACHE_FILE, "w") as f:
            json.dump(cache_payload, f)

    return rf_probs


def attach_rf_prob(
    df: pd.DataFrame,
    n_bars: int = 3,
    min_train: int = 2000,
    retrain_every: int = 500,
    n_estimators: int = 100,
    max_depth: int = 5,
    min_samples_leaf: int = 20,
    random_state: int = 42,
    force: bool = False,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Attach 'rf_prob_long' column to df if not already present."""
    df_out = df.copy()
    col_name = "rf_prob_long"
    if force or col_name not in df_out.columns:
        df_out[col_name] = compute_rf_prob(
            df_out,
            n_bars=n_bars,
            min_train=min_train,
            retrain_every=retrain_every,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            force=force,
            use_cache=use_cache,
        )
    return df_out


if __name__ == "__main__":
    from data import load_klines
    from strategy import to_df

    print("Testing rf_filter...")
    df_k = to_df(load_klines())
    df_res = attach_rf_prob(df_k, n_bars=3)
    valid_rf = df_res["rf_prob_long"].dropna()
    print(f"Generated {len(valid_rf)} RF probabilities out of {len(df_res)} bars.")
    print(valid_rf.describe())
