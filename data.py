"""Fetch & cache BTC klines. fapi.binance.com is ISP-blocked; use data-api.binance.vision (spot) as perp proxy."""
import json
import os
import time

import requests

BASE = "https://data-api.binance.vision/api/v3/klines"
CACHE = os.path.join(os.path.dirname(__file__), "data")
COLS = ["open_time", "open", "high", "low", "close", "volume",
        "close_time", "qvol", "trades", "tbb", "tbq", "ignore"]


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int | None = None) -> list[list]:
    """Paginated kline fetch (1000/request)."""
    out = []
    cur = start_ms
    end = end_ms or int(time.time() * 1000)
    while cur < end:
        r = requests.get(BASE, params={
            "symbol": symbol, "interval": interval,
            "startTime": cur, "endTime": end, "limit": 1000,
        }, timeout=15)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
        cur = batch[-1][6] + 1  # close_time + 1ms
        if len(batch) < 1000:
            break
        time.sleep(0.25)  # politeness
    # dedupe by open_time
    seen = {}
    for row in out:
        seen[row[0]] = row
    return [seen[k] for k in sorted(seen)]


def load_klines(symbol: str = "BTCUSDT", interval: str = "4h",
                since: str = "2019-01-01", force: bool = False):
    """Load cached klines as list of dicts; incremental refetch if cache older than 15 min."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{symbol}_{interval}.json")
    cached = None
    if os.path.exists(path):
        with open(path) as f:
            cached = json.load(f)
    if cached and not force and time.time() - os.path.getmtime(path) < 900:
        return cached
    if cached and not force:
        start_ms = int(cached[-1]["open_time"])  # refetch last bar too (may have been open)
    else:
        start_ms = int(time.mktime(time.strptime(since, "%Y-%m-%d")) * 1000)
    rows = fetch_klines(symbol, interval, start_ms)
    data = [{"open_time": row[0], "open": float(row[1]), "high": float(row[2]),
             "low": float(row[3]), "close": float(row[4]), "volume": float(row[5])}
            for row in rows]
    if cached and not force:
        merged = {d["open_time"]: d for d in cached}
        merged.update({d["open_time"]: d for d in data})
        data = [merged[k] for k in sorted(merged)]
    with open(path, "w") as f:
        json.dump(data, f)
    return data


if __name__ == "__main__":
    kl = load_klines(force=True)
    import datetime
    t0 = datetime.datetime.utcfromtimestamp(kl[0]["open_time"] / 1000)
    t1 = datetime.datetime.utcfromtimestamp(kl[-1]["open_time"] / 1000)
    print(f"{len(kl)} klines | {t0.date()} -> {t1.date()} | last close: {kl[-1]['close']}")
