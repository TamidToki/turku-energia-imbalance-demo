"""Download real Fingrid open data and build the model's input table.

Fingrid open data is free under a Creative Commons Attribution licence, but
the API requires a personal API key. Getting one takes a few minutes:

    1. Register at https://data.fingrid.fi/en/instructions
    2. Sign in, open the Control panel, then the Developer portal
    3. Subscribe to "Open Data starter" (free)
    4. Copy your API key from the developer portal profile page

Keep the key out of git. Either export it:

    export FINGRID_API_KEY="your-key"          # Windows: set FINGRID_API_KEY=your-key

or put one line in a local file named .env in this folder (git-ignored):

    FINGRID_API_KEY=your-key

Usage:
    python fingrid_data.py --download --build           # fetch + build
    python fingrid_data.py --build                      # rebuild from cache
    python fingrid_data.py --build --spot spot_fi.csv   # add a spot series

The spot price (Nord Pool day-ahead, Finland) is NOT open data. Turku
Energia licensed it for hackathon use only; do not download it from
unofficial sources and never commit it here (data/ is git-ignored for that
reason). If Turku Energia or your own Nord Pool/ENTSO-E access supplies it,
pass a CSV with columns timestamp_utc,value via --spot. Without it the
exceptional-price outputs stay empty on purpose (see README).

Datasets used (verified at data.fingrid.fi, all 15-minute, CC-BY):
    319  Imbalance price (EUR/MWh) - the forecast target
    244  Up-regulating price, balancing energy market (EUR/MWh)
    106  Down-regulation price, balancing energy market (EUR/MWh)
    375  Activated mFRR balancing regulation, upward sum (MW)
    376  Activated mFRR balancing regulation, downward sum (MW)
    246  Wind power generation forecast, updated once a day (MWh/h)
    247  Solar power generation forecast, updated once a day (MWh/h)
    165  Electricity consumption forecast, updated once a day (MW)

Why not more: the 15-minute-updated forecasts (245, 166) are only available
as latest values, not past vintages, so they would leak in a backtest
(see availability.py). Cross-border flows, aFRR and production actuals are
available on the same portal and can be added the same way.

API shape (from Fingrid's instructions and its official client):
GET https://data.fingrid.fi/api/datasets/{datasetId}/data with header
x-api-key, params startTime/endTime (UTC ISO), pageSize, page, sortBy,
sortOrder; response {"data": [...], "pagination": {...}}. Rate limits are
about 10 requests/minute and 10,000/day, so downloads are throttled.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API = "https://data.fingrid.fi/api"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
REAL_INPUTS_CSV = DATA_DIR / "real_inputs.csv"
QUALITY_REPORT_MD = DATA_DIR / "real_quality_report.md"
META_JSON = DATA_DIR / "real_meta.json"

# Download a little more than the model grid, so 24 h history windows and
# 3 h ramp features at the first training origin are complete.
DOWNLOAD_START = "2026-05-31T00:00:00Z"
DOWNLOAD_END = "2026-11-03T00:00:00Z"  # exclusive
GRID_START = "2026-06-01T00:00:00Z"
GRID_END = "2026-11-03T00:00:00Z"  # exclusive; covers the last test delivery

# Columns the feature engine needs but v2 does not fetch (not used as model
# features; kept in the schema so the same code runs on both data sources).
UNFETCHED_CONTEXT_COLUMNS = [
    "wind_actual_mw",
    "solar_actual_mw",
    "consumption_actual_mw",
    "flow_se1_fi_mw",
    "flow_se3_fi_mw",
    "flow_ee_fi_mw",
]


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: int
    column: str
    unit: str
    role: str  # target | settlement | forecast_daily


DATASETS = [
    DatasetSpec(319, "imbalance_price_eur_mwh", "EUR/MWh", "target"),
    DatasetSpec(244, "mfrr_up_price", "EUR/MWh", "settlement"),
    DatasetSpec(106, "mfrr_down_price", "EUR/MWh", "settlement"),
    DatasetSpec(375, "mfrr_activation_up", "MW", "settlement"),
    DatasetSpec(376, "mfrr_activation_down", "MW", "settlement"),
    DatasetSpec(246, "wind_forecast_mw", "MWh/h", "forecast_daily"),
    DatasetSpec(247, "solar_forecast_mw", "MWh/h", "forecast_daily"),
    DatasetSpec(165, "consumption_forecast_mw", "MW", "forecast_daily"),
]


class FingridApiError(RuntimeError):
    """Raised for API failures, with an actionable message."""


def load_api_key(env_file: Path = BASE_DIR / ".env") -> str:
    """Read the key from the environment or a local git-ignored .env file."""
    key = os.environ.get("FINGRID_API_KEY", "").strip()
    if key:
        return key
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("FINGRID_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise FingridApiError(
        "No Fingrid API key found. Register for free at "
        "https://data.fingrid.fi/en/instructions, then either "
        "export FINGRID_API_KEY=... or put FINGRID_API_KEY=... in a local "
        ".env file here (git-ignored). Never commit the key."
    )


class FingridClient:
    """Minimal throttled client for the Fingrid open data API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = API,
        min_interval_s: float = 6.1,  # ~10 requests/minute documented limit
        timeout_s: int = 60,
        session: requests.Session | None = None,
        sleeper=time.sleep,
    ):
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._min_interval = min_interval_s
        self._timeout = timeout_s
        self._session = session or requests.Session()
        self._sleep = sleeper
        self._last_call = 0.0

    def _get_page(self, dataset_id: int, start: str, end: str, page: int, page_size: int) -> dict:
        params = {
            "startTime": start,
            "endTime": end,
            "pageSize": page_size,
            "page": page,
            "sortBy": "startTime",
            "sortOrder": "asc",
        }
        url = f"{self._base}/datasets/{dataset_id}/data"
        last_error = "unknown error"
        for attempt in range(4):
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                self._sleep(wait)
            self._last_call = time.monotonic()
            try:
                resp = self._session.get(
                    url, headers={"x-api-key": self._key}, params=params, timeout=self._timeout
                )
            except requests.RequestException as exc:  # network failure: retry
                last_error = f"network error: {exc}"
                self._sleep(5 * (attempt + 1))
                continue
            if resp.status_code in (401, 403):
                raise FingridApiError(
                    "The API key was rejected (HTTP "
                    f"{resp.status_code}). Check FINGRID_API_KEY and that the "
                    "'Open Data starter' subscription is active in the "
                    "developer portal."
                )
            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}"
                self._sleep(10 * (attempt + 1))
                continue
            if resp.status_code != 200:
                raise FingridApiError(f"Unexpected HTTP {resp.status_code}: {resp.text[:200]}")
            try:
                return resp.json()
            except ValueError:
                raise FingridApiError("Response was not JSON; the API may have changed.")
        raise FingridApiError(f"Dataset {dataset_id}: request failed repeatedly ({last_error}).")

    def dataset_data(
        self, dataset_id: int, start: str, end: str, page_size: int = 20000
    ) -> pd.DataFrame:
        """Fetch one dataset as a DataFrame [timestamp_utc, value] (UTC, sorted)."""
        rows: list[dict] = []
        page = 1
        while True:
            payload = self._get_page(dataset_id, start, end, page, page_size)
            data = payload.get("data") if isinstance(payload, dict) else payload
            if not data:
                break
            rows.extend(data)
            pagination = payload.get("pagination", {}) if isinstance(payload, dict) else {}
            total_pages = pagination.get("totalPages") or pagination.get("pageCount")
            if total_pages is not None:
                if page >= int(total_pages):
                    break
            elif len(data) < page_size:
                break
            page += 1
        if not rows:
            return pd.DataFrame(columns=["timestamp_utc", "value"])
        df = pd.DataFrame(rows)
        df["timestamp_utc"] = pd.to_datetime(df["startTime"], utc=True)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.drop_duplicates(subset="timestamp_utc", keep="first")
        return df[["timestamp_utc", "value"]].sort_values("timestamp_utc").reset_index(drop=True)


def download_all(
    client: FingridClient,
    raw_dir: Path = RAW_DIR,
    start: str = DOWNLOAD_START,
    end: str = DOWNLOAD_END,
    force: bool = False,
) -> dict[int, Path]:
    """Fetch every registered dataset into raw_dir as fingrid_<id>.csv."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[int, Path] = {}
    for spec in DATASETS:
        out = raw_dir / f"fingrid_{spec.dataset_id}.csv"
        paths[spec.dataset_id] = out
        if out.exists() and not force:
            print(f"  {spec.dataset_id} {spec.column}: cached ({out.name})")
            continue
        df = client.dataset_data(spec.dataset_id, start, end)
        if df.empty:
            raise FingridApiError(
                f"Dataset {spec.dataset_id} ({spec.column}) returned no rows for "
                f"{start}..{end}. Check the dataset id at data.fingrid.fi."
            )
        df.to_csv(out, index=False)
        print(f"  {spec.dataset_id} {spec.column}: {len(df)} rows -> {out.name}")
    return paths


def _load_raw(path: Path, spec: DatasetSpec) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["timestamp_utc"])
    if df["timestamp_utc"].dt.tz is None:
        df["timestamp_utc"] = df["timestamp_utc"].dt.tz_localize("UTC")
    return df.set_index("timestamp_utc")["value"].sort_index()


def load_spot_csv(spot_path: Path) -> pd.Series:
    """Read a lawfully obtained spot series: timestamp_utc,value (EUR/MWh)."""
    df = pd.read_csv(spot_path)
    cols = {c.lower(): c for c in df.columns}
    ts_col = cols.get("timestamp_utc") or cols.get("timestamp") or cols.get("starttime")
    val_col = cols.get("value") or cols.get("price") or cols.get("spot")
    if not ts_col or not val_col:
        raise ValueError(
            f"{spot_path}: expected columns like timestamp_utc,value - found {list(df.columns)}"
        )
    ts = pd.to_datetime(df[ts_col], utc=True)
    return pd.Series(pd.to_numeric(df[val_col], errors="coerce").to_numpy(), index=ts).sort_index()


def build_inputs(
    raw_dir: Path = RAW_DIR,
    spot_path: Path | None = None,
    grid_start: str = GRID_START,
    grid_end: str = GRID_END,
    out_csv: Path = REAL_INPUTS_CSV,
    report_md: Path = QUALITY_REPORT_MD,
    meta_json: Path = META_JSON,
) -> pd.DataFrame:
    """Assemble the wide 15-minute input table the feature engine expects.

    Mirrors the starter package idea: one UTC quarter-hour grid, one column
    per series, gaps left as NaN and counted in the quality report. The
    availability rule itself is applied later, per forecast origin, by
    features.py with the conservative delays in availability.py.
    """
    grid = pd.date_range(grid_start, grid_end, freq="15min", inclusive="left", tz="UTC")
    wide = pd.DataFrame({"timestamp_utc": grid})
    report: dict[str, dict] = {}

    for spec in DATASETS:
        path = raw_dir / f"fingrid_{spec.dataset_id}.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run: python fingrid_data.py --download (needs your API key)."
            )
        series = _load_raw(path, spec)
        observed = series.loc[(series.index >= grid[0]) & (series.index <= grid[-1])]
        aligned = series.reindex(grid)
        gaps = int(aligned.isna().sum())
        report[spec.column] = {
            "dataset_id": spec.dataset_id,
            "unit": spec.unit,
            "role": spec.role,
            "rows_downloaded": int(len(series)),
            "rows_in_grid": int(len(observed)),
            "missing_quarters_in_grid": gaps,
        }
        wide[spec.column] = aligned.to_numpy()

    # Dominating mFRR direction from activation volumes: 1 up, -1 down, 0 none.
    up = wide["mfrr_activation_up"]
    down = wide["mfrr_activation_down"]
    both_missing = int((up.isna() & down.isna()).sum())
    # Dataset 376 stores down-regulation volumes as NEGATIVE numbers, so the
    # comparison must use magnitudes; a raw up > down comparison would call
    # nearly every quarter "up" (audited 2026-11-11: fixed 60/39 down/up mix).
    up_mag = up.fillna(0).abs()
    down_mag = down.fillna(0).abs()
    direction = pd.Series(0, index=wide.index, dtype=int)
    direction[up_mag > down_mag] = 1
    direction[down_mag > up_mag] = -1
    wide["mfrr_direction"] = direction
    report["mfrr_direction"] = {
        "derived_from": "375 vs 376 activation volumes",
        "quarters_with_both_missing_set_to_none": both_missing,
        "share_up": round(float((direction == 1).mean()), 4),
        "share_down": round(float((direction == -1).mean()), 4),
        "share_none": round(float((direction == 0).mean()), 4),
    }

    # Spot price and exceptional-price labels, only from a lawful source.
    has_spot = spot_path is not None
    if has_spot:
        spot = load_spot_csv(spot_path).reindex(grid)
        missing_spot = int(spot.isna().sum())
        wide["spot_fi_eur_mwh"] = spot.to_numpy()
        imb_vals = wide["imbalance_price_eur_mwh"].to_numpy(dtype=float)
        spot_vals = spot.to_numpy(dtype=float)
        labelled = ~np.isnan(imb_vals) & ~np.isnan(spot_vals)
        wide["spike_high"] = ((imb_vals > spot_vals + 50) & labelled).astype(int)
        wide["spike_low"] = ((imb_vals < spot_vals - 50) & labelled).astype(int)
        report["spot_fi_eur_mwh"] = {
            "source": str(spot_path),
            "missing_quarters_in_grid": missing_spot,
            "spike_high_share": round(float(wide["spike_high"].mean()), 4),
            "spike_low_share": round(float(wide["spike_low"].mean()), 4),
        }
    else:
        wide["spot_fi_eur_mwh"] = float("nan")
        # Zeros here mean "no labels exist", not "no spikes happened". The
        # model skips spike training and outputs NaN (see model.py).
        wide["spike_high"] = 0
        wide["spike_low"] = 0
        report["spot_fi_eur_mwh"] = {
            "source": "none supplied",
            "note": "spot is not open data; spike labels unavailable, not zero in reality",
        }

    for col in UNFETCHED_CONTEXT_COLUMNS:
        wide[col] = float("nan")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_csv, index=False)

    meta = {
        "built_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "has_spot": has_spot,
        "grid": {"start": grid_start, "end": grid_end, "quarters": int(len(grid))},
        "datasets": {str(s.dataset_id): s.column for s in DATASETS},
        "licence": "Fingrid open data, Creative Commons Attribution (data.fingrid.fi)",
        "availability_rule": "see availability.py (conservative publication delays)",
    }
    meta_json.write_text(json.dumps(meta, indent=2))

    lines = [
        "# Real data quality report",
        "",
        f"Built: {meta['built_at_utc']} | grid {grid_start} .. {grid_end} ({len(grid)} quarter-hours)",
        "",
        "| series | dataset | rows in grid | missing quarters | note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, r in report.items():
        lines.append(
            f"| {name} | {r.get('dataset_id', '-')} | {r.get('rows_in_grid', '-')} | "
            f"{r.get('missing_quarters_in_grid', '-')} | {r.get('note', r.get('derived_from', ''))} |"
        )
    lines += [
        "",
        "Direction shares: "
        f"up {report['mfrr_direction']['share_up']}, "
        f"down {report['mfrr_direction']['share_down']}, "
        f"none {report['mfrr_direction']['share_none']}.",
        "",
        "Gaps stay NaN: the model handles NaN features, forecast origins whose "
        "latest price is missing are skipped (and counted) by real_data.py, and "
        "quarters without actuals are excluded from scoring.",
    ]
    report_md.write_text("\n".join(lines))
    print(f"Built {out_csv.name}: {len(wide)} quarter-hours, spot={'yes' if has_spot else 'NO (spike metrics off)'}")
    print(f"Quality report: {report_md.name}")
    return wide


def load_real_inputs(path: Path = REAL_INPUTS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp_utc"])
    if df["timestamp_utc"].dt.tz is None:
        df["timestamp_utc"] = df["timestamp_utc"].dt.tz_localize("UTC")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch and assemble real Fingrid open data.")
    ap.add_argument("--download", action="store_true", help="fetch all datasets (needs API key)")
    ap.add_argument("--build", action="store_true", help="build data/real_inputs.csv from the cache")
    ap.add_argument("--spot", default=None, help="optional lawful spot CSV (timestamp_utc,value)")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    args = ap.parse_args()
    if not (args.download or args.build):
        ap.error("choose --download and/or --build")

    if args.download:
        client = FingridClient(load_api_key())
        print("Downloading Fingrid open data (throttled to the API rate limit)...")
        download_all(client, force=args.force)
    if args.build:
        build_inputs(spot_path=Path(args.spot) if args.spot else None)
    print("Licence: Fingrid open data, Creative Commons Attribution. Source: data.fingrid.fi")


if __name__ == "__main__":
    try:
        main()
    except (FingridApiError, FileNotFoundError, ValueError) as exc:
        sys.exit(f"ERROR: {exc}")
