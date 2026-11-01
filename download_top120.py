"""Download historical 4H klines for Top 120 liquid USDT-M Futures pairs.

Skips symbols already cached with sufficient bars.
Saves to data/universe_4h/{SYMBOL}.json.
"""
import datetime
import json
from pathlib import Path
import time
import requests

SAVE_DIR = Path("data/universe_4h")
BASE_URL = "https://demo-fapi.binance.com"


def get_top_120_symbols() -> list[str]:
    fapi_info = requests.get(f"{BASE_URL}/fapi/v1/exchangeInfo", timeout=10).json()
    fapi_symbols = {
        s["symbol"]
        for s in fapi_info["symbols"]
        if s["contractType"] == "PERPETUAL"
        and s["quoteAsset"] == "USDT"
        and s["status"] == "TRADING"
        and s["symbol"].replace("USDT", "").isalnum()
        and s["symbol"].isascii()
    }

    spot_tickers = requests.get("https://data-api.binance.vision/api/v3/ticker/24hr", timeout=10).json()
    exclude = {
        "USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "BUSDUSDT", "EURUSDT",
        "AEURUSDT", "USDPUSDT", "RLUSDUSDT", "USD1USDT", "RWUSDUSDT",
    }

    candidates = []
    for t in spot_tickers:
        sym = t["symbol"]
        if sym in fapi_symbols and sym not in exclude:
            candidates.append((sym, float(t.get("quoteVolume", 0))))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return [c[0] for c in candidates[:120]]


def fetch_klines(symbol: str, max_pages: int = 5, limit: int = 1500) -> list[dict]:
    all_raw = []
    end_time = None

    for _ in range(max_pages):
        params = {"symbol": symbol, "interval": "4h", "limit": limit}
        if end_time:
            params["endTime"] = int(end_time)
        try:
            r = requests.get(f"{BASE_URL}/fapi/v1/klines", params=params, timeout=10)
            if r.status_code != 200:
                break
            data = r.json()
            if not data or not isinstance(data, list):
                break
            all_raw = data + all_raw
            end_time = data[0][0] - 1
            if len(data) < limit:
                break
            time.sleep(0.04)
        except Exception:
            break

    unique = {b[0]: b for b in all_raw}
    sorted_raw = [unique[k] for k in sorted(unique.keys())]

    klines = []
    for b in sorted_raw:
        klines.append({
            "open_time": int(b[0]),
            "open": float(b[1]),
            "high": float(b[2]),
            "low": float(b[3]),
            "close": float(b[4]),
            "volume": float(b[5]),
            "close_time": int(b[6]),
            "quote_volume": float(b[7]),
            "count": int(b[8]),
        })
    return klines


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    symbols = get_top_120_symbols()
    print(f"Top 120 symbols targeted. Checking existing cache...\n")

    to_download = []
    for s in symbols:
        p = SAVE_DIR / f"{s}.json"
        if not p.exists():
            to_download.append(s)
        else:
            try:
                with open(p) as fp:
                    d = json.load(fp)
                if len(d) < 1000:
                    to_download.append(s)
            except Exception:
                to_download.append(s)

    print(f"Already cached: {len(symbols) - len(to_download)} symbols. Downloading remaining {len(to_download)} symbols...\n")

    t0 = time.time()
    for idx, sym in enumerate(to_download, 1):
        target_path = SAVE_DIR / f"{sym}.json"
        klines = fetch_klines(sym, max_pages=5)
        if not klines or len(klines) < 150:
            print(f"[{idx:2d}/{len(to_download)}] {sym:10s} -> SKIPPED (insufficient data)")
            continue

        with open(target_path, "w") as fp:
            json.dump(klines, fp)

        d_start = datetime.datetime.fromtimestamp(klines[0]["open_time"] / 1000, datetime.UTC).date()
        d_end = datetime.datetime.fromtimestamp(klines[-1]["open_time"] / 1000, datetime.UTC).date()
        print(f"[{idx:2d}/{len(to_download)}] {sym:10s} -> {len(klines):>5d} bars ({d_start} to {d_end})")

    print(f"\n[OK] Download process completed in {round(time.time() - t0, 1)}s.")


if __name__ == "__main__":
    main()
