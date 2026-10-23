"""Telegram notifier for BTC futures signal system."""
import json
import os
import sys
from pathlib import Path
import requests

CONFIG_PATH = Path(__file__).resolve().parent / "notifier_config.json"


def get_config() -> dict | None:
    """Load Telegram configuration from env vars or notifier_config.json."""
    env_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    env_chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if env_token and env_chat_id:
        return {"bot_token": env_token, "chat_id": env_chat_id}

    if not CONFIG_PATH.exists():
        sys.stderr.write(f"Warning: {CONFIG_PATH.name} tidak ditemukan dan env vars kosong. Notifikasi dilewati.\n")
        return None
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        tg = cfg.get("telegram", {})
        bot_token = tg.get("bot_token")
        chat_id = tg.get("chat_id")
        if not bot_token or not chat_id:
            sys.stderr.write("Warning: bot_token atau chat_id tidak lengkap di notifier_config.json\n")
            return None
        return {"bot_token": bot_token, "chat_id": chat_id}
    except Exception as e:
        sys.stderr.write(f"Warning: Gagal membaca {CONFIG_PATH.name}: {e}\n")
        return None


def send_message(text: str, parse_mode: str = "HTML") -> dict | None:
    """Kirim pesan Telegram dan kembalikan respon API (dict) atau None jika gagal."""
    cfg = get_config()
    if not cfg:
        return None

    url = f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage"
    payload = {
        "chat_id": cfg["chat_id"],
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True,
    }

    try:
        resp = requests.post(url, json=payload, timeout=15)
        data = resp.json()
        if resp.status_code == 200 and data.get("ok"):
            return data
        sys.stderr.write(f"Error Telegram API [{resp.status_code}]: {resp.text}\n")
        return None
    except Exception as e:
        sys.stderr.write(f"Error send_telegram: {e}\n")
        return None


def send_telegram(text: str, parse_mode: str = "HTML") -> bool:
    """Kirim pesan Telegram via Bot API. Return True jika berhasil, False jika gagal."""
    res = send_message(text, parse_mode=parse_mode)
    return bool(res and res.get("ok"))


if __name__ == "__main__":
    test_msg = "✅ futur notifier aktif"
    print(f"Mengirim pesan uji: '{test_msg}'...")
    res = send_message(test_msg)
    if res and res.get("ok"):
        result = res.get("result", {})
        msg_id = result.get("message_id")
        print(f"Status: HTTP 200, ok=True, message_id={msg_id}")
    else:
        print("Pengiriman pesan gagal.")
        sys.exit(1)
