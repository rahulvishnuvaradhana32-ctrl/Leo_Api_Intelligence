#!/usr/bin/env python3
"""
kaggle_sync.py — Auto-sync Kaggle outputs to local models/

Watches your Downloads folder. The moment any Kaggle result file
appears (even with (1), (2) suffix), it copies it to models/ instantly.

Usage:
    python scripts/kaggle_sync.py

Keep this running in a background terminal while you work on Kaggle.
Press Ctrl+C to stop.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import shutil
import time
from pathlib import Path
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────────────
DOWNLOADS = Path.home() / "Downloads"
MODELS    = Path(__file__).parent.parent / "models"
MODELS.mkdir(exist_ok=True)

# Files to watch for — maps base name → destination name in models/
WATCH_FILES = {
    "stress_test_best_model": "stress_test_best_model.pth",
    "scaler":                 "scaler.pkl",
    "lstm_results":           "lstm_results.json",
    "lstm_roc_curves":        "lstm_roc_curves.png",
    "evaluation_results":     "evaluation_results.json",
    "ablation_results":       "ablation_results.json",
    "ablation_results":       "ablation_results.png",
    "conformal_results":      "conformal_results.json",
    "conformal_calibration":  "conformal_calibration.png",
    "agent_simulation_results": "agent_simulation_results.json",
    "agent_simulation_chart":   "agent_simulation_chart.png",
}

# Track already-processed files so we don't copy the same file twice
_already_synced: set = set()

def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

def _stem(path: Path) -> str:
    """Strip (1), (2) suffixes from filenames like 'scaler (2).pkl'"""
    name = path.stem  # e.g. "scaler (2)"
    # Remove trailing " (N)" pattern
    import re
    return re.sub(r"\s*\(\d+\)$", "", name).strip()

def _dest_name(path: Path) -> str | None:
    """Return destination filename in models/ or None if not a watched file."""
    stem = _stem(path)
    ext  = path.suffix.lstrip(".")
    # Try exact stem match first
    if stem in WATCH_FILES:
        return WATCH_FILES[stem]
    # Try stem + extension match
    candidate = f"{stem}.{ext}"
    for _, dest in WATCH_FILES.items():
        if dest == candidate:
            return dest
    return None

def scan_and_sync():
    """Scan Downloads and copy any new matching files to models/."""
    synced = 0
    for f in DOWNLOADS.iterdir():
        if not f.is_file():
            continue
        # Skip temp/partial downloads
        if f.suffix in (".crdownload", ".part", ".tmp"):
            continue
        if str(f) in _already_synced:
            continue

        dest_name = _dest_name(f)
        if dest_name is None:
            continue

        dest = MODELS / dest_name

        # Only copy if the Downloads file is newer than what's in models/
        if dest.exists():
            if f.stat().st_mtime <= dest.stat().st_mtime:
                _already_synced.add(str(f))
                continue

        # Copy
        shutil.copy2(f, dest)
        _already_synced.add(str(f))
        size_kb = f.stat().st_size // 1024
        _log(f"SYNCED  {f.name}  →  models/{dest_name}  ({size_kb} KB)")
        synced += 1

    return synced

def main():
    _log(f"Watching: {DOWNLOADS}")
    _log(f"Syncing to: {MODELS}")
    _log("Waiting for Kaggle output files... (Ctrl+C to stop)\n")

    # Print what we're watching for
    seen = set()
    for dest in WATCH_FILES.values():
        if dest not in seen:
            print(f"  Watching for: {dest}")
            seen.add(dest)
    print()

    try:
        while True:
            n = scan_and_sync()
            if n > 0:
                _log(f"Sync complete — {n} file(s) copied to models/\n")
            time.sleep(3)   # check every 3 seconds
    except KeyboardInterrupt:
        _log("Stopped.")

if __name__ == "__main__":
    main()
