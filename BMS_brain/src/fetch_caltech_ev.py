import re
import json
import time
import requests
import pandas as pd
from pathlib import Path

BASE_URL = "https://ev.caltech.edu"
DATASET_URL = f"{BASE_URL}/dataset"
SITE = "Caltech"
YEAR = 2019

def fetch_caltech_sessions(year: int = 2019, site: str = "Caltech") -> pd.DataFrame:
    """
    Downloads EV charging sessions from Caltech ACN-Data in monthly batches.
    Avoids single-request buffer truncation issues.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    
    print(f"Connecting to {DATASET_URL}...")
    r = session.get(DATASET_URL, timeout=20)
    r.raise_for_status()
    
    # Extract CSRF token
    csrf_match = re.search(r'name=["\']csrf_token["\'][^>]*value=["\']([^"\']+)["\']', r.text)
    if not csrf_match:
        csrf_match = re.search(r'value=["\']([^"\']+)["\'][^>]*name=["\']csrf_token["\']', r.text)
    if not csrf_match:
        raise RuntimeError("Failed to extract CSRF token from Caltech dataset page.")
    
    token = csrf_match.group(1)
    print(f"Acquired CSRF token: {token[:12]}...")
    
    # Monthly ranges
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    all_sessions = []
    
    print(f"\nDownloading sessions for {site} - Year {year} in 12 monthly batches...")
    for month in range(1, 13):
        start_str = f"{month:02d}/01/{year} 00:00:00"
        end_str = f"{month:02d}/{days_in_month[month-1]:02d}/{year} 23:59:59"
        
        payload = {
            "csrf_token": token,
            "site": site,
            "start": start_str,
            "end": end_str,
            "min_kWh": "",
            "submit": "Download"
        }
        
        t0 = time.time()
        resp = session.post(DATASET_URL, data=payload, timeout=60)
        resp.raise_for_status()
        
        data = resp.json()
        items = data.get("_items", [])
        all_sessions.extend(items)
        elapsed = time.time() - t0
        kwh_sum = sum(item.get("kWhDelivered", 0) for item in items)
        print(f"  Month {month:02d}/{year}: {len(items):>4} sessions | {kwh_sum:>8.2f} kWh | {elapsed:4.2f}s")
        time.sleep(0.5) # respectful interval
        
    print(f"\nTotal sessions downloaded: {len(all_sessions)}")
    df = pd.DataFrame(all_sessions)
    return df

if __name__ == "__main__":
    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "caltech_ev_sessions_2019.parquet"
    
    df_sessions = fetch_caltech_sessions(YEAR, SITE)
    print(f"Dataset shape: {df_sessions.shape}")
    print(f"Columns: {df_sessions.columns.tolist()}")
    print(f"Total energy delivered: {df_sessions['kWhDelivered'].sum():,.2f} kWh")
    
    # Convert userInputs column to JSON string if needed for parquet compatibility
    if "userInputs" in df_sessions.columns:
        df_sessions["userInputs"] = df_sessions["userInputs"].apply(lambda x: json.dumps(x) if x is not None else None)
        
    df_sessions.to_parquet(out_file, index=False)
    print(f"Saved raw sessions to {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")
