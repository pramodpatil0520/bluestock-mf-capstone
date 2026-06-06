import argparse
import json
import logging
import time
from pathlib import Path
 
import pandas as pd
import requests
 
# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
 
# ── config ─────────────────────────────────────────────────────────────────────
BASE_URL  = "https://api.mfapi.in/mf"
RAW_DIR   = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
 
# The 6 schemes we track.  Key = AMFI code, value = short label for filenames.
SCHEMES = {
    "125497": "hdfc_top100",
    "119551": "sbi_bluechip",
    "120503": "icici_bluechip",
    "118632": "nippon_largecap",
    "119092": "axis_bluechip",
    "120841": "kotak_bluechip",
}
 
# be a polite API consumer — don't hammer with rapid-fire requests
REQUEST_DELAY_SEC = 0.8
 
 
# ==============================================================================
# Core fetch helpers
# ==============================================================================
 
def fetch_scheme_data(amfi_code: str, retries: int = 3) -> dict | None:
    """
    GET https://api.mfapi.in/mf/{amfi_code}
 
    Returns the full JSON response (scheme meta + all NAV history) or None
    if every retry fails.  mfapi.in is usually stable but occasionally
    returns 5xx when their upstream AMFI pull is running.
    """
    url = f"{BASE_URL}/{amfi_code}"
 
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            log.info(f"  [{amfi_code}] fetched {len(data.get('data', []))} NAV records")
            return data
        except requests.exceptions.HTTPError as e:
            log.warning(f"  [{amfi_code}] HTTP {e.response.status_code} on attempt {attempt}")
        except requests.exceptions.ConnectionError:
            log.warning(f"  [{amfi_code}] Connection error on attempt {attempt}")
        except requests.exceptions.Timeout:
            log.warning(f"  [{amfi_code}] Timeout on attempt {attempt}")
        except json.JSONDecodeError:
            log.error(f"  [{amfi_code}] Response wasn't valid JSON — skipping")
            return None
 
        if attempt < retries:
            time.sleep(2 ** attempt)   # exponential back-off: 2s, 4s
 
    log.error(f"  [{amfi_code}] All {retries} attempts failed.")
    return None
 
 
def parse_nav_records(amfi_code: str, raw_data: dict) -> pd.DataFrame:
    """
    mfapi.in response looks like:
      {
        "meta": { "scheme_name": "...", "fund_house": "...", ... },
        "data": [ { "date": "31-05-2026", "nav": "892.4560" }, ... ]
      }
 
    This turns that into a tidy DataFrame with proper types.
    """
    nav_records = raw_data.get("data", [])
    if not nav_records:
        log.warning(f"  [{amfi_code}] No NAV records in response.")
        return pd.DataFrame()
 
    df = pd.DataFrame(nav_records)
    df.rename(columns={"date": "nav_date"}, inplace=True)
 
    # mfapi returns dates as DD-MM-YYYY strings
    df["nav_date"] = pd.to_datetime(df["nav_date"], format="%d-%m-%Y", errors="coerce")
    df["nav"]      = pd.to_numeric(df["nav"], errors="coerce")
    df["amfi_code"] = amfi_code
 
    # drop NaT dates or zero/null NAVs
    df.dropna(subset=["nav_date", "nav"], inplace=True)
    df = df[df["nav"] > 0].copy()
 
    df.sort_values("nav_date", inplace=True)
    df.reset_index(drop=True, inplace=True)
 
    return df[["amfi_code", "nav_date", "nav"]]
 
 
def parse_scheme_meta(amfi_code: str, raw_data: dict) -> dict:
    """Pull the metadata block out of the mfapi response."""
    meta = raw_data.get("meta", {})
    return {
        "amfi_code":   amfi_code,
        "scheme_name": meta.get("scheme_name", ""),
        "fund_house":  meta.get("fund_house", ""),
        "scheme_type": meta.get("scheme_type", ""),
        "scheme_category": meta.get("scheme_category", ""),
    }
 
 
# ==============================================================================
# Latest NAV only (lightweight)
# ==============================================================================
 
def fetch_latest_nav(amfi_code: str) -> dict | None:
    """
    mfapi has a /latest endpoint but it isn't always reliable,
    so we just pull the full history and grab the last row.
    Slightly wasteful on bandwidth but more dependable.
    """
    raw = fetch_scheme_data(amfi_code)
    if raw is None:
        return None
    df = parse_nav_records(amfi_code, raw)
    if df.empty:
        return None
    last = df.iloc[-1]
    return {
        "amfi_code": amfi_code,
        "nav_date":  last["nav_date"].strftime("%Y-%m-%d"),
        "nav":       last["nav"],
    }
 
 
# ==============================================================================
# Full history fetch + save
# ==============================================================================
 
def fetch_and_save_history(amfi_code: str, label: str) -> pd.DataFrame | None:
    """
    Fetch full NAV history for one scheme and save it as a CSV.
    Also appends/updates the master nav_history file.
    """
    raw = fetch_scheme_data(amfi_code)
    if raw is None:
        return None
 
    nav_df  = parse_nav_records(amfi_code, raw)
    meta    = parse_scheme_meta(amfi_code, raw)
 
    if nav_df.empty:
        log.warning(f"  [{amfi_code}] Empty NAV DataFrame — nothing saved.")
        return None
 
    # individual scheme file
    out_path = RAW_DIR / f"nav_{label}_{amfi_code}.csv"
    nav_df.to_csv(out_path, index=False)
    log.info(f"  [{amfi_code}] Saved {len(nav_df):,} rows → {out_path.name}")
 
    # log metadata to console for quick reference
    log.info(
        f"  [{amfi_code}] {meta['scheme_name']} | "
        f"Latest NAV: {nav_df.iloc[-1]['nav']:.4f} on {nav_df.iloc[-1]['nav_date'].date()}"
    )
 
    return nav_df
 
 
def fetch_all_and_merge():
    """
    Fetch all 6 schemes and merge into one combined CSV.
    This combined file can directly feed into the ETL pipeline's
    02_nav_history.csv slot if you want live data instead of the packaged one.
    """
    all_frames = []
 
    for code, label in SCHEMES.items():
        log.info(f"Fetching {label} ({code}) ...")
        df = fetch_and_save_history(code, label)
        if df is not None:
            all_frames.append(df)
        time.sleep(REQUEST_DELAY_SEC)
 
    if not all_frames:
        log.error("No data fetched at all. Check your internet connection.")
        return
 
    combined = pd.concat(all_frames, ignore_index=True)
    combined.sort_values(["amfi_code", "nav_date"], inplace=True)
 
    out_path = RAW_DIR / "live_nav_combined.csv"
    combined.to_csv(out_path, index=False)
    log.info(f"Combined NAV file saved → {out_path}  ({len(combined):,} rows total)")
 
    # quick summary table
    summary = (
        combined.groupby("amfi_code")
                .agg(records=("nav", "count"),
                     from_date=("nav_date", "min"),
                     to_date=("nav_date", "max"),
                     latest_nav=("nav", "last"))
                .reset_index()
    )
    log.info("\nSummary:\n" + summary.to_string(index=False))
 
 
# ==============================================================================
# Latest-only mode (quick morning check)
# ==============================================================================
 
def print_latest_navs():
    """Print today's NAV for all 6 schemes — handy for a quick sanity check."""
    rows = []
    for code, label in SCHEMES.items():
        rec = fetch_latest_nav(code)
        if rec:
            rows.append({
                "Fund":      label,
                "AMFI Code": code,
                "NAV Date":  rec["nav_date"],
                "NAV (Rs.)": f"{rec['nav']:.4f}",
            })
        time.sleep(REQUEST_DELAY_SEC)
 
    if rows:
        print("\n" + pd.DataFrame(rows).to_string(index=False))
    else:
        print("Could not fetch any NAV data.")
 
 
# ==============================================================================
# CLI
# ==============================================================================
 
def main():
    parser = argparse.ArgumentParser(
        description="Fetch live NAV data from mfapi.in for Bluestock MF project"
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Only print the latest NAV for each scheme (no CSV saved)"
    )
    parser.add_argument(
        "--code",
        type=str,
        default=None,
        help="Fetch a single scheme by AMFI code (e.g. 125497)"
    )
    args = parser.parse_args()
 
    if args.latest and args.code:
        # single scheme latest
        rec = fetch_latest_nav(args.code)
        if rec:
            print(f"\nAMFI {rec['amfi_code']}  |  Date: {rec['nav_date']}  |  NAV: Rs.{rec['nav']:.4f}")
        else:
            print(f"Could not fetch NAV for code {args.code}")
 
    elif args.latest:
        print_latest_navs()
 
    elif args.code:
        label = SCHEMES.get(args.code, f"scheme_{args.code}")
        fetch_and_save_history(args.code, label)
 
    else:
        # default: fetch everything and save
        log.info("Fetching full NAV history for all 6 schemes ...")
        fetch_all_and_merge()
 
 
if __name__ == "__main__":
    main()
 