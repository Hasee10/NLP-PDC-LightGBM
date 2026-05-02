"""
RetailEL — Full Pipeline Runner
Runs all steps in sequence and saves output to log.txt

Run order:
    A-generate_synthetic_data.py   — build data/ from instacart_data/ CSVs
    B-real_data_loader.py          — build data_real/ from instacart_data/ CSVs
    C-evaluate_full.py             — evaluate pipeline, save results/

To start the API after evaluation:
    python D-api_server.py
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

COMMANDS = [
    [sys.executable, "A-generate_synthetic_data.py"],
    [sys.executable, "B-real_data_loader.py"],
    [sys.executable, "C-evaluate_full.py", "--mode", "real"],
]

LOG_FILE = Path("log.txt")

def run():
    log_lines = []

    def log(text):
        print(text)
        log_lines.append(text)

    log("=" * 60)
    log("RetailEL Pipeline Run")
    log(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    for cmd in COMMANDS:
        log(f"\nRunning: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        if result.stdout:
            log(result.stdout)
        if result.stderr:
            log("\n[STDERR]\n" + result.stderr)

        if result.returncode != 0:
            log(f"\nFAILED with exit code {result.returncode}")
            break

        log("\nOK")

    LOG_FILE.write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\nLog saved -> {LOG_FILE.resolve()}")

if __name__ == "__main__":
    run()
