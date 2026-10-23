"""BTC on-chain data fetcher & feature generator (Nagula & Alexakis 2022).

Fetches hash rate and unique active addresses from blockchain.info charts API (free, no API key).
Caches to data/onchain_*.json. Computes rolling z-scores and resamples to 4H bars with forward-fill.
"""
import json
import os
import time

import pandas as pd
import requests

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data")
BASE_URL = "https://api.blockchain.info/charts"
METRICS = {
    "hash_rate": "hash-rate",
    "n_unique_addresses": "n-unique-addresses",
}


def fetch_chart(metric_slug: str, force: bool = False, ttl_sec: int = 86400) -> list[dict]:
    """Fetch chart data from blockchain.info with local file caching."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"onchain_{metric_slug.replace('-', '_')}.json")

    if not force and os.path.exists(cache_file):
        age = time.time() - os.path.getmtime(cache_file)
        if age < ttl_sec:
            with open(cache_file, "r") as f:
                data = json.load(f)
            return data.get("values", [])

    url = f"{BASE_URL}/{metric_slug}?timespan=all&sampled=false&format=json&cors=true"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        payload = r.json()
        with open(cache_file, "w") as f:
            json.dump(payload, f)
        return payload.get("values", [])
    except Exception as e:
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                data = json.load(f)
            return data.get("values", [])
        raise RuntimeError(f"Failed to fetch {metric_slug} and no cache available: {e}") from e


def load_onchain(window: int = 30, force: bool = False) -> pd.DataFrame:
    """Load daily on-chain metrics, compute rolling z-scores, and shift 1 day forward to prevent lookahead."""
    dfs = []
    for col_name, slug in METRICS.items():
        vals = fetch_chart(slug, force=force)
        df_m = pd.DataFrame(vals).rename(columns={"y": col_name})
        df_m["dt"] = pd.to_datetime(df_m["x"], unit="s")
        df_m = df_m[["dt", col_name]].drop_duplicates(subset=["dt"]).sort_values("dt")
        dfs.append(df_m)

    merged = dfs[0]
    for df_m in dfs[1:]:
        merged = pd.merge(merged, df_m, on="dt", how="inner")

    merged = merged.sort_values("dt").reset_index(drop=True)
    for col_name in METRICS:
        roll = merged[col_name].rolling(window, min_periods=max(5, window // 2))
        merged[f"{col_name}_z"] = (merged[col_name] - roll.mean()) / roll.std()

    # Shift timestamps forward by 1 day: daily metric of day T is only observable at T+1 00:00:00
    merged["dt"] = merged["dt"] + pd.Timedelta(days=1)
    return merged[["dt", "hash_rate", "n_unique_addresses", "hash_rate_z", "n_unique_addresses_z"]]


def merge_onchain(df: pd.DataFrame, df_onchain: pd.DataFrame | None = None, window: int = 30) -> pd.DataFrame:
    """Merge daily on-chain z-scores into 4H klines dataframe using merge_asof backward."""
    if df_onchain is None:
        df_onchain = load_onchain(window=window)

    df_out = df.copy()
    was_indexed = "dt" not in df_out.columns and isinstance(df_out.index, pd.DatetimeIndex)
    if was_indexed:
        df_out = df_out.reset_index().rename(columns={"index": "dt"})

    cols_to_merge = ["dt", "hash_rate_z", "n_unique_addresses_z"]
    df_merged = pd.merge_asof(
        df_out.sort_values("dt"),
        df_onchain[cols_to_merge].sort_values("dt"),
        on="dt",
        direction="backward",
    )
    # forward-fill any remaining NaNs in initial bars
    df_merged["hash_rate_z"] = df_merged["hash_rate_z"].bfill().ffill()
    df_merged["n_unique_addresses_z"] = df_merged["n_unique_addresses_z"].bfill().ffill()

    if was_indexed:
        df_merged = df_merged.set_index("dt")
    return df_merged


if __name__ == "__main__":
    df_oc = load_onchain()
    print(f"Loaded {len(df_oc)} daily on-chain records from {df_oc['dt'].iloc[0].date()} to {df_oc['dt'].iloc[-1].date()}")
    print(df_oc.tail(5).to_string(index=False))
