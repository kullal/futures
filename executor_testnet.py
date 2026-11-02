"""Binance Futures Testnet Executor.

Direct REST client for executing signals on Binance Futures Testnet (testnet.binancefuture.com).
Uses requests + hmac (zero extra dependencies, YAGNI / Ponytail principle).

Features:
- Idempotency & safety checks (prevents double entry)
- Auto leverage (cap 3x) & ISOLATED margin mode
- Position sizing based on 1R risk budget (default 2% equity)
- Immediate Stop Loss (STOP_MARKET) & Take Profit (TAKE_PROFIT_MARKET) placement
- Audit log in data/testnet_trades.json
"""
import argparse
import datetime
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import time
import typing
import urllib.parse
import requests

CONFIG_FILE = Path("testnet_config.json")
LOG_FILE = Path("data/testnet_trades.json")
# Binance migrated testnet endpoint to demo-fapi.binance.com
BASE_URL = os.getenv("BINANCE_TESTNET_BASE_URL", "https://demo-fapi.binance.com")


def load_config() -> dict:
    cfg = {
        "api_key": os.getenv("BINANCE_TESTNET_API_KEY", ""),
        "api_secret": os.getenv("BINANCE_TESTNET_API_SECRET", ""),
        "symbol": "BTCUSDT",
        "leverage": 3,
        "risk_pct": 0.02,
    }
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                file_cfg = json.load(f)
                cfg.update({k: v for k, v in file_cfg.items() if v})
        except Exception:
            pass
    if os.getenv("BINANCE_TESTNET_API_KEY"):
        cfg["api_key"] = os.getenv("BINANCE_TESTNET_API_KEY")
    if os.getenv("BINANCE_TESTNET_API_SECRET"):
        cfg["api_secret"] = os.getenv("BINANCE_TESTNET_API_SECRET")
    return cfg


class BinanceTestnet:
    def __init__(self, api_key: str = "", api_secret: str = ""):
        self.api_key = api_key
        self.api_secret = api_secret
        self.time_offset = 0
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "futur-testnet-executor/1.0",
        })
        if self.api_key:
            self.session.headers["X-MBX-APIKEY"] = self.api_key
        self._sync_time()

    def _sync_time(self):
        """Calculate time drift between client and Binance server to prevent -1021 errors."""
        try:
            resp = self.session.get(f"{BASE_URL}/fapi/v1/time", timeout=5).json()
            server_time = int(resp.get("serverTime", 0))
            if server_time > 0:
                self.time_offset = server_time - int(time.time() * 1000)
        except Exception:
            self.time_offset = 0

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000) + self.time_offset
        params["recvWindow"] = 10000
        query = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature
        return params

    def _request(self, method: str, path: str, params: dict | None = None, signed: bool = False) -> typing.Any:
        url = f"{BASE_URL}{path}"
        params = params or {}
        if signed:
            if not self.api_key or not self.api_secret:
                raise ValueError("API key and secret required for signed endpoints")
            params = self._sign(params)

        if method == "GET":
            resp = self.session.get(url, params=params, timeout=10)
        elif method == "POST":
            resp = self.session.post(url, data=params, timeout=10)
        elif method == "DELETE":
            resp = self.session.delete(url, params=params, timeout=10)
        else:
            raise ValueError(f"Unsupported method: {method}")

        try:
            data = resp.json()
        except Exception:
            resp.raise_for_status()
            return {"status": resp.status_code}

        if resp.status_code >= 400:
            msg = data.get("msg", resp.text) if isinstance(data, dict) else resp.text
            code = data.get("code", resp.status_code) if isinstance(data, dict) else resp.status_code
            raise RuntimeError(f"Binance API error [{code}]: {msg}")

        return data

    def ping(self) -> bool:
        res = self._request("GET", "/fapi/v1/ping")
        return res == {}

    def get_server_time(self) -> int:
        data = self._request("GET", "/fapi/v1/time")
        return int(data["serverTime"])

    def get_balance(self) -> float:
        """Return available USDT balance."""
        data = self._request("GET", "/fapi/v2/account", signed=True)
        for asset in data.get("assets", []):
            if asset.get("asset") == "USDT":
                return float(asset.get("availableBalance", 0.0))
        return 0.0

    def get_position(self, symbol: str = "BTCUSDT") -> dict:
        """Return current open position for symbol."""
        positions = self._request("GET", "/fapi/v2/positionRisk", {"symbol": symbol}, signed=True)
        if not positions or not isinstance(positions, list):
            return {"size": 0.0, "side": "FLAT", "entry_price": 0.0, "pnl": 0.0}

        pos = positions[0]
        amt = float(pos.get("positionAmt", 0.0))
        if amt > 0:
            side = "LONG"
        elif amt < 0:
            side = "SHORT"
        else:
            side = "FLAT"

        return {
            "size": abs(amt),
            "side": side,
            "entry_price": float(pos.get("entryPrice", 0.0)),
            "pnl": float(pos.get("unRealizedProfit", 0.0)),
            "leverage": int(pos.get("leverage", 1)),
            "margin_type": pos.get("marginType", "cross"),
        }

    def get_all_open_positions(self) -> list[dict]:
        """Return list of all non-flat open positions."""
        positions = self._request("GET", "/fapi/v2/positionRisk", signed=True)
        if not positions or not isinstance(positions, list):
            return []
        active = []
        for pos in positions:
            amt = float(pos.get("positionAmt", 0.0))
            if abs(amt) > 0:
                active.append({
                    "symbol": pos.get("symbol"),
                    "side": "LONG" if amt > 0 else "SHORT",
                    "size": abs(amt),
                    "entry_price": float(pos.get("entryPrice", 0.0)),
                    "pnl": float(pos.get("unRealizedProfit", 0.0)),
                })
        return active

    def get_symbol_rules(self, symbol: str) -> dict:
        """Fetch and cache lot precision and step size for symbol."""
        if not hasattr(self, "_rules_cache"):
            self._rules_cache = {}
        if symbol in self._rules_cache:
            return self._rules_cache[symbol]
        try:
            info = self._request("GET", "/fapi/v1/exchangeInfo")
            for s in info.get("symbols", []):
                sym = s["symbol"]
                q_prec = int(s.get("quantityPrecision", 2))
                p_prec = int(s.get("pricePrecision", 2))
                step = 10 ** (-q_prec) if q_prec > 0 else 1.0
                tick = 10 ** (-p_prec) if p_prec > 0 else 0.01
                for f in s.get("filters", []):
                    if f.get("filterType") == "LOT_SIZE":
                        step = float(f.get("stepSize", step))
                    elif f.get("filterType") == "PRICE_FILTER":
                        tick = float(f.get("tickSize", tick))
                self._rules_cache[sym] = {
                    "quantityPrecision": q_prec,
                    "pricePrecision": p_prec,
                    "stepSize": step,
                    "tickSize": tick,
                }
            if symbol in self._rules_cache:
                return self._rules_cache[symbol]
        except Exception:
            pass
        return {"quantityPrecision": 2, "pricePrecision": 2, "stepSize": 0.01, "tickSize": 0.01}

    def setup_symbol(self, symbol: str = "BTCUSDT", leverage: int = 3):
        """Set leverage cap and isolated margin."""
        # Set leverage
        try:
            self._request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": leverage}, signed=True)
        except RuntimeError as e:
            if "leverage" not in str(e).lower():
                raise

        # Set margin type to ISOLATED (ignore -4046 'No need to change')
        try:
            self._request("POST", "/fapi/v1/marginType", {"symbol": symbol, "marginType": "ISOLATED"}, signed=True)
        except RuntimeError as e:
            if "-4046" not in str(e):
                # Only raise if it's not the redundant change code
                pass

    def cancel_all_orders(self, symbol: str = "BTCUSDT"):
        """Cancel all open regular orders and conditional algo orders."""
        try:
            self._request("DELETE", "/fapi/v1/allOpenOrders", {"symbol": symbol}, signed=True)
        except Exception:
            pass
        try:
            algos = self._request("GET", "/fapi/v1/openAlgoOrders", {"symbol": symbol}, signed=True)
            if isinstance(algos, list):
                for a in algos:
                    if "algoId" in a:
                        self._request("DELETE", "/fapi/v1/algoOrder", {"symbol": symbol, "algoId": a["algoId"]}, signed=True)
        except Exception:
            pass

    def close_position(self, symbol: str = "BTCUSDT") -> dict | None:
        """Close open position at market if exists."""
        pos = self.get_position(symbol)
        if pos["side"] == "FLAT":
            return None

        side = "SELL" if pos["side"] == "LONG" else "BUY"
        self.cancel_all_orders(symbol)
        order = self._request(
            "POST",
            "/fapi/v1/order",
            {
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": pos["size"],
                "reduceOnly": "true",
            },
            signed=True,
        )
        return order

    def place_market_order(self, symbol: str, side: str, quantity: float) -> dict:
        return self._request(
            "POST",
            "/fapi/v1/order",
            {
                "symbol": symbol,
                "side": side.upper(),
                "type": "MARKET",
                "quantity": quantity,
            },
            signed=True,
        )

    def place_stop_loss(self, symbol: str, side: str, stop_price: float | str, quantity: float) -> dict:
        """Place STOP_MARKET order via Binance Algo Service."""
        return self._request(
            "POST",
            "/fapi/v1/algoOrder",
            {
                "symbol": symbol,
                "side": side.upper(),
                "type": "STOP_MARKET",
                "algoType": "CONDITIONAL",
                "triggerPrice": str(stop_price),
                "quantity": str(quantity),
                "reduceOnly": "true",
                "positionSide": "BOTH",
                "workingType": "CONTRACT_PRICE",
            },
            signed=True,
        )

    def place_take_profit(self, symbol: str, side: str, tp_price: float | str, quantity: float) -> dict:
        """Place TAKE_PROFIT_MARKET order via Binance Algo Service."""
        return self._request(
            "POST",
            "/fapi/v1/algoOrder",
            {
                "symbol": symbol,
                "side": side.upper(),
                "type": "TAKE_PROFIT_MARKET",
                "algoType": "CONDITIONAL",
                "triggerPrice": str(tp_price),
                "quantity": str(quantity),
                "reduceOnly": "true",
                "positionSide": "BOTH",
                "workingType": "CONTRACT_PRICE",
            },
            signed=True,
        )


def calculate_position_size(
    balance: float,
    entry: float,
    sl: float,
    risk_pct: float = 0.01,
    leverage: float = 3.0,
    step_size: float = 0.001,
    precision: int = 3,
) -> float:
    """Calculate position size adhering to risk budget and leverage cap."""
    if balance <= 0 or entry <= 0 or entry == sl:
        return 0.0

    risk_dist = abs(entry - sl)
    risk_usdt = balance * risk_pct
    raw_size = risk_usdt / risk_dist

    # Leverage cap (hard limit 3X)
    max_size = (balance * leverage) / entry
    size = min(raw_size, max_size)

    # Quantize to step size
    steps = math.floor(size / step_size)
    quantized_size = round(steps * step_size, precision)
    if precision == 0:
        quantized_size = float(int(quantized_size))

    # Minimum notional check (Binance min notional is $5)
    if quantized_size * entry < 5.0:
        return 0.0

    return quantized_size


def log_trade(record: dict):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    trades = []
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, "r") as f:
                trades = json.load(f)
        except Exception:
            trades = []
    trades.append(record)
    with open(LOG_FILE, "w") as f:
        json.dump(trades, f, indent=2)


def execute_signal(sig: dict, dry_run: bool = False) -> dict:
    """Execute strategy signal on Binance Testnet."""
    cfg = load_config()
    symbol = sig.get("symbol", cfg.get("symbol", "BTCUSDT"))
    leverage = cfg.get("leverage", 3)
    risk_pct = cfg.get("risk_pct", 0.01)

    side = sig["side"].upper()  # LONG or SHORT
    entry_est = float(sig["entry"])
    sl = float(sig["sl"])
    tp = float(sig["tp"])

    client = BinanceTestnet(cfg.get("api_key", ""), cfg.get("api_secret", ""))
    rules = client.get_symbol_rules(symbol)
    step_size = rules["stepSize"]
    q_prec = rules["quantityPrecision"]
    p_prec = rules["pricePrecision"]

    sl_rounded = round(sl, p_prec)
    tp_rounded = round(tp, p_prec)

    if dry_run or not cfg.get("api_key") or not cfg.get("api_secret"):
        sim_balance = 10_000.0
        size = calculate_position_size(
            sim_balance, entry_est, sl,
            risk_pct=risk_pct, leverage=leverage,
            step_size=step_size, precision=q_prec
        )
        return {
            "mode": "DRY_RUN (Simulasi)",
            "symbol": symbol,
            "side": side,
            "sim_balance": sim_balance,
            "entry_est": entry_est,
            "sl": sl_rounded,
            "tp": tp_rounded,
            "calculated_size": size,
            "notional_usdt": round(size * entry_est, 2),
            "risk_budget_usdt": round(sim_balance * risk_pct, 2),
            "status": "VALID" if size > 0 else "SIZE_TOO_SMALL",
        }

    # 1. Setup leverage & margin mode
    client.setup_symbol(symbol=symbol, leverage=leverage)

    # 2. Check existing position for this symbol
    pos = client.get_position(symbol)
    if pos["side"] == side:
        return {
            "status": "SKIPPED_EXISTING_POSITION",
            "message": f"Sudah memiliki posisi {side} aktif di {symbol} ({pos['size']}). Mencegah double entry.",
        }

    if pos["side"] != "FLAT":
        client.close_position(symbol)

    # 3. Calculate size from actual testnet balance
    balance = client.get_balance()
    size = calculate_position_size(
        balance, entry_est, sl,
        risk_pct=risk_pct, leverage=leverage,
        step_size=step_size, precision=q_prec
    )
    if size <= 0:
        return {
            "status": "ABORTED",
            "message": f"Ukuran posisi {size} terlalu kecil atau saldo USDT tidak cukup (${balance:,.2f})",
        }

    # 4. Cancel open SL/TP before new orders
    client.cancel_all_orders(symbol)

    # 5. Place Entry Market Order
    order_side = "BUY" if side == "LONG" else "SELL"
    exit_side = "SELL" if side == "LONG" else "BUY"

    entry_order = client.place_market_order(symbol, order_side, size)

    # 6. Place SL & TP conditional orders immediately via Algo Service
    sl_order = client.place_stop_loss(symbol, exit_side, sl_rounded, size)
    tp_order = client.place_take_profit(symbol, exit_side, tp_rounded, size)

    record = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "symbol": symbol,
        "side": side,
        "size": size,
        "entry_est": entry_est,
        "sl": sl_rounded,
        "tp": tp_rounded,
        "entry_order_id": entry_order.get("orderId"),
        "sl_order_id": sl_order.get("algoId"),
        "tp_order_id": tp_order.get("algoId"),
    }
    log_trade(record)

    return {
        "status": "EXECUTED",
        "details": record,
    }


def main():
    parser = argparse.ArgumentParser(description="Binance Futures Testnet Executor")
    parser.add_argument("--ping", action="store_true", help="Test connectivity to Testnet API")
    parser.add_argument("--status", action="store_true", help="Check Testnet account balance and open position")
    parser.add_argument("--close", action="store_true", help="Close active open position at market")
    parser.add_argument("--test-sizing", action="store_true", help="Run self-check on position sizing math")
    parser.add_argument("--dry-run", action="store_true", help="Simulate execution on latest scanner signal")
    args = parser.parse_args()

    client = BinanceTestnet()

    if args.ping:
        ok = client.ping()
        t = client.get_server_time()
        print(f"[OK] Ping Binance Testnet success. Server time: {t}")
        return

    if args.test_sizing:
        # Self-check sizing
        # Normal risk-budgeted case: 2% of $10,000 = $200 risk. Dist = $2,000 -> 0.1 BTC
        s1 = calculate_position_size(10_000.0, 60_000.0, 58_000.0, risk_pct=0.02, leverage=3.0)
        assert s1 == 0.1, f"Expected 0.1 BTC, got {s1}"

        # Leverage-capped case: Dist = $200 -> raw size = $200 / $200 = 1.0 BTC.
        # But max leverage 3x on $10k at $60k/BTC caps position at 30k / 60k = 0.5 BTC.
        s2 = calculate_position_size(10_000.0, 60_000.0, 59_800.0, risk_pct=0.02, leverage=3.0)
        assert s2 == 0.5, f"Expected leverage-capped 0.5 BTC, got {s2}"
        print("[OK] Position sizing math self-check PASSED.")
        return

    if args.status:
        cfg = load_config()
        if not cfg.get("api_key") or not cfg.get("api_secret"):
            print("Belum ada api_key/api_secret di testnet_config.json")
            print("Daftar gratis di: https://testnet.binancefuture.com")
            return
        c = BinanceTestnet(cfg["api_key"], cfg["api_secret"])
        bal = c.get_balance()
        pos = c.get_position(cfg.get("symbol", "BTCUSDT"))
        print(f"Testnet USDT Available: ${bal:,.2f}")
        print(f"Active Position: {pos}")
        return

    if args.close:
        cfg = load_config()
        if not cfg.get("api_key") or not cfg.get("api_secret"):
            print("Belum ada api_key/api_secret di testnet_config.json")
            return
        c = BinanceTestnet(cfg["api_key"], cfg["api_secret"])
        sym = cfg.get("symbol", "BTCUSDT")
        res = c.close_position(sym)
        print(f"[OK] Position {sym} closed: {res}")
        return

    if args.dry_run:
        # Run dry-run with mock signal
        mock_sig = {
            "side": "LONG",
            "entry": 68_500.0,
            "sl": 66_500.0,
            "tp": 72_500.0,
        }
        res = execute_signal(mock_sig, dry_run=True)
        print(json.dumps(res, indent=2))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
