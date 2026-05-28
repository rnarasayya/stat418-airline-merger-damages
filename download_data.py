"""
download_data.py
================
STAT 418 Final Project — Airline Merger Damages Estimator
Data collection script for:
  1. DOT DB1B Market data (BTS bulk download) — 2010–2017, 32 quarters
  2. Jet fuel prices (FRED API)

Usage:
    python download_data.py --fred-key YOUR_FRED_API_KEY

Get a free FRED API key at: https://fred.stlouisfed.org/docs/api/api_key.html

Output directory structure:
    data/
    ├── raw/
    │   ├── db1b/          ← one .parquet file per quarter
    │   └── fred/          ← fuel prices CSV
    └── log/
        └── download.log   ← full download log
"""

import argparse
import io
import logging
import os
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# Fix Windows console encoding so plain ASCII logging works everywhere
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Configuration ─────────────────────────────────────────────────────────────

# Study period — AA/US Airways merger closed December 2013
# Pre-merger: 2010–2013  |  Post-merger: 2014–2017
START_YEAR = 2010
END_YEAR   = 2017

# FRED series to download
FRED_SERIES = {
    "jet_fuel_weekly": "WJFUELUSGULF",        # Weekly U.S. kerosene-type jet fuel price
    "airline_cpi":     "CUSR0000SETG01",  # Seasonally adjusted airline CPI (bonus control)
}

# Output directories
DATA_DIR  = Path("data")
DB1B_DIR  = DATA_DIR / "raw" / "db1b"
FRED_DIR  = DATA_DIR / "raw" / "fred"
LOG_DIR   = DATA_DIR / "log"

# BTS bulk download URL pattern
DB1B_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "Origin_and_Destination_Survey_DB1BMarket_{year}_{quarter}.zip"
)

# FRED API URL pattern
FRED_URL = (
    "https://api.stlouisfed.org/fred/series/observations"
    "?series_id={series_id}&api_key={api_key}&file_type=json"
    "&observation_start=2009-01-01&observation_end=2018-12-31"
)

# ── Logging setup ──────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("download")
    logger.setLevel(logging.DEBUG)

    # File handler — full detail
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))

    # Console handler — concise
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Log file: {log_path}")
    return logger


# ── DB1B Download ──────────────────────────────────────────────────────────────

def download_db1b_quarter(
    year: int,
    quarter: int,
    out_dir: Path,
    logger: logging.Logger,
    retries: int = 3,
) -> bool:
    """
    Download one quarter of DB1B Market data, save as parquet.
    Returns True on success, False on failure.
    """
    out_path = out_dir / f"db1b_market_{year}_Q{quarter}.parquet"

    # Skip if already downloaded
    if out_path.exists():
        logger.info(f"  [SKIP] {year} Q{quarter} -- already exists, skipping")
        return True

    url = DB1B_URL.format(year=year, quarter=quarter)
    logger.debug(f"URL: {url}")

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"  [DOWN] {year} Q{quarter} -- downloading (attempt {attempt}/{retries})...")
            response = requests.get(url, timeout=180)
            response.raise_for_status()

            size_mb = len(response.content) / 1e6
            logger.debug(f"  Downloaded {size_mb:.1f} MB compressed")

            # Unzip and load
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_files = [f for f in z.namelist() if f.endswith(".csv")]
                if not csv_files:
                    logger.error(f"  [FAIL] No CSV found in zip for {year} Q{quarter}")
                    return False
                with z.open(csv_files[0]) as f:
                    df = pd.read_csv(f, low_memory=False)

            # Drop the trailing empty column BTS includes
            df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

            # Basic validation
            required_cols = ["Origin", "Dest", "MktFare", "Passengers", "OpCarrier"]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                logger.error(f"  [FAIL] Missing columns: {missing}")
                return False

            if len(df) < 100_000:
                logger.warning(f"  [WARN] Suspiciously low row count: {len(df):,}")

            # Save
            df.to_parquet(out_path, index=False)
            logger.info(f"  [OK]   {year} Q{quarter} -- {len(df):,} rows saved -> {out_path.name}")
            return True

        except requests.HTTPError as e:
            logger.warning(f"  HTTP error: {e}")
        except requests.Timeout:
            logger.warning(f"  Timeout on attempt {attempt}")
        except Exception as e:
            logger.error(f"  Unexpected error: {e}")

        if attempt < retries:
            wait = 5 * attempt
            logger.info(f"  Retrying in {wait}s...")
            time.sleep(wait)

    logger.error(f"  [FAIL] FAILED after {retries} attempts: {year} Q{quarter}")
    return False


def download_all_db1b(logger: logging.Logger) -> dict:
    """Download all 32 quarters of DB1B Market data."""
    DB1B_DIR.mkdir(parents=True, exist_ok=True)

    quarters = [(y, q) for y in range(START_YEAR, END_YEAR + 1) for q in range(1, 5)]
    total = len(quarters)

    logger.info("\n" + "=" * 55)
    logger.info(f"  DOT DB1B MARKET DATA DOWNLOAD")
    logger.info(f"  {total} quarters: {START_YEAR} Q1 to {END_YEAR} Q4")
    logger.info("=" * 55)

    results = {"success": [], "failed": [], "skipped": []}

    for i, (year, quarter) in enumerate(quarters, 1):
        logger.info(f"\n[{i:02d}/{total}] {year} Q{quarter}")

        out_path = DB1B_DIR / f"db1b_market_{year}_Q{quarter}.parquet"
        if out_path.exists():
            results["skipped"].append((year, quarter))
            logger.info(f"  [SKIP] Already exists, skipping")
            continue

        success = download_db1b_quarter(year, quarter, DB1B_DIR, logger)
        if success:
            results["success"].append((year, quarter))
        else:
            results["failed"].append((year, quarter))

        # Polite delay between requests — don't hammer BTS servers
        if i < total:
            time.sleep(2)

    return results


# ── FRED Download ──────────────────────────────────────────────────────────────

def download_fred_series(
    series_id: str,
    name: str,
    api_key: str,
    out_dir: Path,
    logger: logging.Logger,
) -> bool:
    """Download one FRED series and save as CSV."""
    out_path = out_dir / f"{name}.csv"

    if out_path.exists():
        logger.info(f"  [SKIP] {name} ({series_id}) -- already exists, skipping")
        return True

    url = FRED_URL.format(series_id=series_id, api_key=api_key)
    logger.info(f"  [DOWN] {name} ({series_id}) -- fetching from FRED...")
    logger.debug(f"  URL: {url.replace(api_key, '***')}")

    try:
        response = requests.get(url, timeout=30)

        # Handle bad API key gracefully
        if response.status_code == 400:
            data = response.json()
            logger.error(f"  [FAIL] FRED API error: {data.get('error_message', 'Bad request')}")
            logger.error("     Check your API key at https://fred.stlouisfed.org/docs/api/api_key.html")
            return False

        response.raise_for_status()
        data = response.json()

        observations = data.get("observations", [])
        if not observations:
            logger.error(f"  [FAIL] No observations returned for {series_id}")
            return False

        df = pd.DataFrame(observations)[["date", "value"]]
        df.columns = ["date", series_id]
        df["date"] = pd.to_datetime(df["date"])

        # FRED uses "." for missing values
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
        df = df.dropna(subset=[series_id])

        df.to_csv(out_path, index=False)
        logger.info(f"  [OK]   {name} -- {len(df):,} observations saved -> {out_path.name}")
        logger.info(f"         Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        return True

    except requests.Timeout:
        logger.error(f"  [FAIL] Timeout fetching {series_id}")
        return False
    except Exception as e:
        logger.error(f"  [FAIL] Error fetching {series_id}: {e}")
        return False


def download_all_fred(api_key: str, logger: logging.Logger) -> dict:
    """Download all FRED series."""
    FRED_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("\n" + "=" * 55)
    logger.info("  FRED API DATA DOWNLOAD")
    logger.info(f"  {len(FRED_SERIES)} series: jet fuel prices + airline CPI")
    logger.info("=" * 55 + "\n")

    results = {"success": [], "failed": []}

    for name, series_id in FRED_SERIES.items():
        success = download_fred_series(series_id, name, api_key, FRED_DIR, logger)
        if success:
            results["success"].append(series_id)
        else:
            results["failed"].append(series_id)
        time.sleep(0.5)  # Brief pause between FRED calls

    return results


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(
    db1b_results: dict,
    fred_results: dict,
    logger: logging.Logger,
    elapsed: float,
):
    logger.info("\n" + "=" * 55)
    logger.info("  DOWNLOAD SUMMARY")
    logger.info("=" * 55)

    # DB1B summary
    n_success = len(db1b_results["success"])
    n_skipped = len(db1b_results["skipped"])
    n_failed  = len(db1b_results["failed"])
    logger.info(f"\n  DB1B Market Data:")
    logger.info(f"    [OK]   Downloaded: {n_success} quarters")
    logger.info(f"    [SKIP] Skipped:    {n_skipped} quarters (already existed)")
    logger.info(f"    [FAIL] Failed:     {n_failed} quarters")

    if db1b_results["failed"]:
        logger.info("    Failed quarters:")
        for y, q in db1b_results["failed"]:
            logger.info(f"      {y} Q{q}")

    # FRED summary
    logger.info(f"\n  FRED Series:")
    logger.info(f"    [OK]   Downloaded: {len(fred_results['success'])} series")
    logger.info(f"    [FAIL] Failed:     {len(fred_results['failed'])} series")

    # File sizes
    logger.info(f"\n  Output files:")
    total_size = 0
    for f in sorted(DB1B_DIR.glob("*.parquet")):
        size_mb = f.stat().st_size / 1e6
        total_size += size_mb
    logger.info(f"    DB1B parquet files: {total_size:.0f} MB total")

    logger.info(f"\n  Total time: {elapsed / 60:.1f} minutes")

    all_ok = n_failed == 0 and len(fred_results["failed"]) == 0
    if all_ok:
        logger.info("\n  *** All downloads complete -- ready for EDA notebook! ***")
    else:
        logger.info("\n  [WARN] Some downloads failed. Re-run the script to retry.")
        logger.info("         (Script skips already-downloaded files automatically)")

    logger.info("=" * 55)


# ── Entrypoint ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download DB1B + FRED data for airline merger damages project"
    )
    parser.add_argument(
        "--fred-key",
        type=str,
        required=True,
        help="FRED API key (get one free at https://fred.stlouisfed.org/docs/api/api_key.html)",
    )
    parser.add_argument(
        "--db1b-only",
        action="store_true",
        help="Only download DB1B data (skip FRED)",
    )
    parser.add_argument(
        "--fred-only",
        action="store_true",
        help="Only download FRED data (skip DB1B)",
    )
    parser.add_argument(
        "--years",
        type=str,
        default=None,
        help="Comma-separated years to download, e.g. --years 2012,2013,2014 (default: all 2010-2017)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging()

    logger.info("=" * 55)
    logger.info("  AIRLINE MERGER DAMAGES — DATA DOWNLOAD")
    logger.info("  STAT 418 Final Project")
    logger.info(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 55)

    # Handle --years override
    global START_YEAR, END_YEAR
    if args.years:
        selected_years = sorted([int(y.strip()) for y in args.years.split(",")])
        logger.info(f"\n  Custom year selection: {selected_years}")

    start_time = time.time()
    db1b_results = {"success": [], "failed": [], "skipped": []}
    fred_results = {"success": [], "failed": []}

    # DB1B download
    if not args.fred_only:
        db1b_results = download_all_db1b(logger)

    # FRED download
    if not args.db1b_only:
        fred_results = download_all_fred(args.fred_key, logger)

    elapsed = time.time() - start_time
    print_summary(db1b_results, fred_results, logger, elapsed)


if __name__ == "__main__":
    main()
