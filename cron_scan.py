"""Windows Task Scheduler runner for futur 4H scanner.
Executes scanner.py --notify and logs result to data/cron.log.
"""
import datetime
import os
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LOG_FILE = os.path.join(DATA_DIR, "cron.log")
PYTHON_EXE = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")

os.makedirs(DATA_DIR, exist_ok=True)

now_utc = datetime.datetime.now(datetime.UTC)
now_wib = now_utc + datetime.timedelta(hours=7)
timestamp = now_wib.strftime("%Y-%m-%d %H:%M:%S WIB")

try:
    # 0x08000000 = CREATE_NO_WINDOW
    r = subprocess.run(
        [PYTHON_EXE, "scanner.py", "--notify"],
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        timeout=300,
        creationflags=0x08000000,
    )
    stdout_msg = r.stdout.strip() if r.stdout else ""
    stderr_msg = r.stderr.strip() if r.stderr else ""
    msg = stdout_msg if stdout_msg else (stderr_msg if stderr_msg else "OK (silent)")

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] Exit {r.returncode}: {msg}\n")
except Exception as e:
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] EXCEPTION: {e}\n")

