#!/usr/bin/env python3
"""
run_kaggle_training.py — Push LSTM training to Kaggle GPU and pull results back

What this script does:
  1. Reads your Kaggle credentials from ~/.kaggle/kaggle.json
  2. (One-time) Uploads data/banking_api_features_v7.csv as a private Kaggle dataset
  3. Patches kaggle_kernel/kernel-metadata.json with your username
  4. Pushes the training kernel to Kaggle (runs on 2×T4 GPU)
  5. Polls every 60s until the kernel finishes
  6. Downloads outputs and copies them into local models/

Usage:
    # First time — upload dataset AND run training
    python scripts/run_kaggle_training.py

    # After dataset is already on Kaggle — skip upload
    python scripts/run_kaggle_training.py --skip-upload

    # Check status of a running kernel without pushing a new one
    python scripts/run_kaggle_training.py --status-only

Prerequisites:
    pip install kaggle
    Place ~/.kaggle/kaggle.json  (from kaggle.com → Account → API → Create New Token)
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT         = Path(__file__).parent.parent
KAGGLE_DIR   = ROOT / "kaggle_kernel"
DATA_DIR     = ROOT / "data"
MODELS_DIR   = ROOT / "models"
CREDENTIALS  = Path.home() / ".kaggle" / "kaggle.json"

DATASET_NAME = "leo-api-v7-dataset"
KERNEL_NAME  = "leo-api-lstm-training"
DATA_FILE    = DATA_DIR / "banking_api_features_v7.csv"

# Files to pull back from Kaggle output into local models/
TARGET_FILES = [
    "stress_test_best_model.pth",
    "scaler.pkl",
    "lstm_results.json",
    "lstm_roc_curves.png",
]


# ── Helpers ────────────────────────────────────────────────────────────────────
def _run(cmd: list, check=True, capture=False) -> subprocess.CompletedProcess:
    """Run a shell command, print it, return the result."""
    print(f"  > {' '.join(str(c) for c in cmd)}")
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
    )


def _kaggle(*args, capture=False) -> subprocess.CompletedProcess:
    return _run(["kaggle"] + list(args), check=False, capture=capture)


def _read_credentials() -> dict:
    if not CREDENTIALS.exists():
        print("\nERROR: ~/.kaggle/kaggle.json not found.")
        print("  Go to https://www.kaggle.com → Account → API → Create New Token")
        print("  Place the downloaded kaggle.json at:", CREDENTIALS)
        sys.exit(1)
    return json.loads(CREDENTIALS.read_text())


def _patch_metadata(username: str) -> None:
    """Replace KAGGLE_USERNAME placeholder in kernel-metadata.json with real username."""
    meta_path = KAGGLE_DIR / "kernel-metadata.json"
    meta      = json.loads(meta_path.read_text())
    meta["id"]                = f"{username}/{KERNEL_NAME}"
    meta["dataset_sources"][0] = f"{username}/{DATASET_NAME}"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  Patched kernel-metadata.json → id: {meta['id']}")


def _upload_dataset(username: str) -> None:
    """Create or update the v7 dataset on Kaggle."""
    if not DATA_FILE.exists():
        print(f"\nERROR: {DATA_FILE} not found.")
        print("  Run scripts/integrate_new_kaggle_data.py first to generate v7.csv")
        sys.exit(1)

    # Build a temp staging directory with dataset-metadata.json + the CSV
    staging = ROOT / ".kaggle_dataset_staging"
    staging.mkdir(exist_ok=True)

    dataset_meta = {
        "title":    "LEO API Features v7",
        "id":       f"{username}/{DATASET_NAME}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (staging / "dataset-metadata.json").write_text(json.dumps(dataset_meta, indent=2))

    # Symlink or copy the CSV into staging
    dest = staging / DATA_FILE.name
    if dest.exists():
        dest.unlink()
    # Copy (not symlink) for cross-platform reliability on Windows
    print(f"\nCopying {DATA_FILE.name} to staging dir (this may take a minute for 856 MB)...")
    shutil.copy2(DATA_FILE, dest)

    # Check if dataset exists already
    check = _kaggle("datasets", "status", f"{username}/{DATASET_NAME}", capture=True)
    if check.returncode == 0 and "error" not in check.stdout.lower():
        print("\nDataset already exists on Kaggle — creating new version ...")
        _kaggle("datasets", "version", "-p", str(staging), "-m", "v7 update")
    else:
        print("\nCreating new dataset on Kaggle (first time) ...")
        _kaggle("datasets", "create", "-p", str(staging))

    # Clean up staging
    shutil.rmtree(staging, ignore_errors=True)
    print("  Dataset upload submitted.\n")


def _push_kernel() -> None:
    print("\nPushing kernel to Kaggle ...")
    result = _kaggle("kernels", "push", "-p", str(KAGGLE_DIR))
    if result.returncode != 0:
        print("ERROR: kernel push failed. Check output above.")
        sys.exit(1)
    print("  Kernel submitted — now running on Kaggle GPU.")


def _poll_status(kernel_slug: str) -> str:
    """Poll until kernel is complete/error/cancelled. Returns final status."""
    print(f"\nPolling kernel status every 60s (kernel: {kernel_slug}) ...")
    print("  You can also monitor at: https://www.kaggle.com/code\n")

    while True:
        result = _kaggle("kernels", "status", kernel_slug, capture=True)
        output = (result.stdout + result.stderr).strip().lower()

        if "complete" in output:
            print(f"\n  Status: COMPLETE")
            return "complete"
        elif "error" in output:
            print(f"\n  Status: ERROR\n  {result.stdout.strip()}")
            return "error"
        elif "cancel" in output:
            print(f"\n  Status: CANCELLED")
            return "cancelled"
        else:
            ts = time.strftime("%H:%M:%S")
            # Extract the raw status line for display
            lines = [l for l in result.stdout.splitlines() if kernel_slug.split("/")[1] in l.lower() or "running" in l.lower() or "queued" in l.lower()]
            status_line = lines[0].strip() if lines else result.stdout.strip()[:80]
            print(f"  [{ts}] {status_line}")
            time.sleep(60)


def _download_outputs(kernel_slug: str) -> None:
    """Download kernel outputs and copy target files into local models/."""
    download_dir = ROOT / ".kaggle_output_tmp"
    download_dir.mkdir(exist_ok=True)

    print(f"\nDownloading outputs from Kaggle ...")
    result = _kaggle("kernels", "output", kernel_slug, "-p", str(download_dir))
    if result.returncode != 0:
        print("WARNING: output download may have issues — check .kaggle_output_tmp/")

    # Walk download dir and copy target files to local models/
    MODELS_DIR.mkdir(exist_ok=True)
    found = []
    for target in TARGET_FILES:
        for found_path in download_dir.rglob(target):
            dest = MODELS_DIR / target
            shutil.copy2(found_path, dest)
            found.append(target)
            print(f"  Copied → models/{target}  ({found_path.stat().st_size // 1024:,} KB)")
            break

    if not found:
        print("\nWARNING: No target files found in download. Files in output dir:")
        for f in download_dir.rglob("*"):
            if f.is_file():
                print(f"  {f.relative_to(download_dir)}")
    else:
        print(f"\n  {len(found)}/{len(TARGET_FILES)} files copied to models/")

    shutil.rmtree(download_dir, ignore_errors=True)


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run LEO API LSTM training on Kaggle GPU and pull results locally"
    )
    parser.add_argument("--skip-upload", action="store_true",
                        help="Skip dataset upload (use if v7 dataset already on Kaggle)")
    parser.add_argument("--status-only", action="store_true",
                        help="Only poll status of last kernel run, no new push")
    parser.add_argument("--download-only", action="store_true",
                        help="Skip upload+push — just download outputs from a completed kernel")
    parser.add_argument("--skip-poll", action="store_true",
                        help="With --download-only: skip status polling, attempt download immediately")
    parser.add_argument("--kernel", type=str, default=None,
                        help="Override kernel slug for --download-only (e.g. username/kernel-name)")
    args = parser.parse_args()

    # ── 1. Check kaggle CLI ────────────────────────────────────────────────
    if shutil.which("kaggle") is None:
        print("ERROR: kaggle CLI not found. Install it:")
        print("  pip install kaggle")
        sys.exit(1)

    # ── 2. Read credentials ────────────────────────────────────────────────
    creds    = _read_credentials()
    username = creds["username"]
    print(f"\nKaggle account: {username}")

    kernel_slug = args.kernel if args.kernel else f"{username}/{KERNEL_NAME}"

    # ── Download-only mode (manual upload + run on Kaggle web UI) ─────────
    if args.download_only:
        print(f"\nDownload-only mode — fetching outputs from: {kernel_slug}")
        if args.skip_poll:
            # Skip status check — attempt download directly (use when API auth
            # blocks kernels status but the notebook is already complete)
            print("Skipping status poll (--skip-poll) — downloading directly ...")
            _download_outputs(kernel_slug)
            print("\nDone. Local models/ updated.")
            print("  python scripts/dashboard_server.py")
        else:
            print("Polling until kernel is complete ...")
            status = _poll_status(kernel_slug)
            if status == "complete":
                _download_outputs(kernel_slug)
                print("\nDone. Local models/ updated.")
                print("  python scripts/dashboard_server.py")
            else:
                print(f"Kernel status: {status} — nothing downloaded.")
                print("If the notebook is already complete on Kaggle, retry with --skip-poll:")
                print(f"  python scripts/run_kaggle_training.py --download-only --skip-poll --kernel {kernel_slug}")
        return

    if args.status_only:
        status = _poll_status(kernel_slug)
        if status == "complete":
            _download_outputs(kernel_slug)
        return

    # ── 3. Upload dataset (one-time) ───────────────────────────────────────
    if not args.skip_upload:
        _upload_dataset(username)
    else:
        print("\nSkipping dataset upload (--skip-upload)")

    # ── 4. Patch kernel-metadata.json ─────────────────────────────────────
    _patch_metadata(username)

    # ── 5. Push kernel ────────────────────────────────────────────────────
    _push_kernel()

    # ── 6. Poll until done ────────────────────────────────────────────────
    status = _poll_status(kernel_slug)

    # ── 7. Download outputs ───────────────────────────────────────────────
    if status == "complete":
        _download_outputs(kernel_slug)
        print("\nDone. Local models/ is updated with fresh Kaggle results.")
        print("Restart the dashboard to see updated metrics:")
        print("  python scripts/dashboard_server.py")
    else:
        print(f"\nKernel ended with status: {status}")
        print("Check logs at: https://www.kaggle.com/code")


if __name__ == "__main__":
    main()
