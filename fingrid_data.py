"""Fetch real open data from Fingrid's open data portal (optional).

Fingrid open data is free under a Creative Commons Attribution licence, but
the API requires a personal API key. Getting one takes a few minutes:

    1. Register at https://data.fingrid.fi/en/instructions
    2. In the developer portal, subscribe to "Open Data starter"
    3. Copy your API key

Then:
    export FINGRID_API_KEY="your-key"
    python fingrid_data.py --dataset 319 --start 2026-06-01 --end 2026-11-02

Verified dataset IDs (from data.fingrid.fi):
    319  Imbalance price (EUR/MWh, 15 min)

Other series used by the challenge (mFRR/aFRR activations, consumption and
production, cross-border flows, day-ahead forecasts 165/246/247) are on the
same portal; look up their dataset IDs at https://data.fingrid.fi/en/data
and pass them with --dataset.

NOTE: the Nord Pool day-ahead spot price is NOT open data. Turku Energia
licensed it for the hackathon only. Do not download it from unofficial
sources and do not commit any copy of it here. Ask Turku Energia how they
want to supply it (the team is asking Anne-Mari Angervo / Vesa Lukka).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://data.fingrid.fi/api"
DATA_DIR = Path(__file__).resolve().parent / "data"


def fetch_dataset(dataset_id: int, start: str, end: str, api_key: str, page_size: int = 20000) -> pd.DataFrame:
    """Fetch one dataset as a DataFrame [timestamp_utc, value]."""
    rows, page = [], 1
    while True:
        r = requests.get(
            f"{API}/datasets/{dataset_id}/data",
            headers={"x-api-key": api_key},
            params={"startTime": f"{start}T00:00:00Z", "endTime": f"{end}T00:00:00Z",
                    "pageSize": page_size, "pageNumber": page},
            timeout=60,
        )
        r.raise_for_status()
        payload = r.json()
        chunk = payload.get("data", [])
        if not chunk:
            break
        rows.extend(chunk)
        if len(chunk) < page_size:
            break
        page += 1
        time.sleep(2.1)  # respect the documented API request limit
    if not rows:
        return pd.DataFrame(columns=["timestamp_utc", "value"])
    out = pd.DataFrame(rows)
    out["timestamp_utc"] = pd.to_datetime(out["startTime"], utc=True)
    return out[["timestamp_utc", "value"]].sort_values("timestamp_utc").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch a Fingrid open data dataset to data/.")
    ap.add_argument("--dataset", type=int, required=True, help="datasetId, e.g. 319 for imbalance price")
    ap.add_argument("--start", default="2026-06-01")
    ap.add_argument("--end", default="2026-11-02")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    key = os.environ.get("FINGRID_API_KEY", "").strip()
    if not key:
        sys.exit("Set FINGRID_API_KEY first (free registration, see module docstring).")

    df = fetch_dataset(args.dataset, args.start, args.end, key)
    out = Path(args.out) if args.out else DATA_DIR / f"fingrid_dataset_{args.dataset}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    print("Licence: Fingrid open data, Creative Commons Attribution. Source: data.fingrid.fi")


if __name__ == "__main__":
    main()
