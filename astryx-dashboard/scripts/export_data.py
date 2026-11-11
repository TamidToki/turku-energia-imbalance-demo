"""Export the Streamlit demo's results into JSON for the Astryx web dashboard.

Run from the repo root after `python backtest.py`:

    python astryx-dashboard/scripts/export_data.py

Reads the git-ignored results/ CSVs (synthetic sample data by default) and
writes astryx-dashboard/public/data/demo-data.json. The JSON is committed so
the web demo runs without Python; regenerate it after a new backtest run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS = BASE_DIR / "results"
REAL_RESULTS = RESULTS / "real_v4" if (RESULTS / "real_v4" / "forecast_test_real.csv").exists() else RESULTS / "real"
OUT = BASE_DIR / "astryx-dashboard" / "public" / "data" / "demo-data.json"

_real_files = ("forecast_test_real.csv", "baseline_test_real.csv", "metrics_summary_real.csv")
real_available = all((REAL_RESULTS / name).exists() for name in _real_files)

if real_available:
    fc_path, bl_path, mt_path = (REAL_RESULTS / n for n in _real_files)
    data_source = "real"
else:
    fc_path = RESULTS / "forecast_test.csv"
    bl_path = RESULTS / "baseline_test.csv"
    mt_path = RESULTS / "metrics_summary.csv"
    data_source = "synthetic"

fc = pd.read_csv(fc_path, parse_dates=["delivery_start_utc"])
bl = pd.read_csv(bl_path, parse_dates=["delivery_start_utc"])
mt = pd.read_csv(mt_path)
if "period" in mt.columns:
    mt = mt[mt["period"] == "test"].drop(columns=["period"], errors="ignore")

origins = sorted(fc["forecast_origin_utc"].unique())
# A usable selector: about 30 evenly spread origins, always including the latest.
stride = max(1, len(origins) // 30)
chosen = list(origins[::stride])
if origins[-1] not in chosen:
    chosen.append(origins[-1])


def rows_for(df: pd.DataFrame, origin: str) -> pd.DataFrame:
    return df[df["forecast_origin_utc"] == origin].sort_values("step")


def num(v, nd=2):
    if pd.isna(v):
        return None
    return round(float(v), nd)


exported = []
for o in chosen:
    fc_rows = rows_for(fc, o)
    bl_rows = rows_for(bl, o)
    label = pd.Timestamp(o).strftime("%a %d %b %Y, %H:%M UTC")
    rows = []
    for (_, f), (_, b) in zip(fc_rows.iterrows(), bl_rows.iterrows()):
        rows.append({
            "step": int(f["step"]),
            "t": pd.Timestamp(f["delivery_start_utc"]).strftime("%H:%M"),
            "p10": num(f["p10"]), "p50": num(f["p50"]), "p90": num(f["p90"]),
            "baseline": num(b["p50"]),
            "actual": num(f.get("actual_price")),
            "spot": num(f.get("actual_spot")),
            "pUp": num(f.get("p_up"), 3), "pDown": num(f.get("p_down"), 3), "pNone": num(f.get("p_none"), 3),
            "pSpikeHigh": num(f.get("p_spike_high"), 3), "pSpikeLow": num(f.get("p_spike_low"), 3),
        })
    exported.append({
        "origin": str(o),
        "label": label,
        "dataCutoff": str(fc_rows["data_cutoff_utc"].iloc[0]),
        "topDrivers": str(fc_rows["top_drivers"].iloc[0]),
        "rows": rows,
    })

metrics = []
for _, m in mt.iterrows():
    metrics.append({
        "model": m["model"],
        "MAE_p50_eur_mwh": num(m.get("MAE_p50_eur_mwh"), 3),
        "pinball_mean": num(m.get("pinball_mean"), 3),
        "coverage_p10_p90": num(m.get("coverage_p10_p90"), 3),
        "brier_spike_high": num(m.get("brier_spike_high"), 4),
        "brier_spike_low": num(m.get("brier_spike_low"), 4),
        "brier_direction": num(m.get("brier_direction"), 4),
    })

per_origin = (
    fc.assign(abs_err=(fc["p50"] - fc["actual_price"]).abs())
    .groupby("forecast_origin_utc")["abs_err"].mean()
)
per_origin_b = (
    bl.assign(abs_err=(bl["p50"] - bl["actual_price"]).abs())
    .groupby("forecast_origin_utc")["abs_err"].mean()
)
mae_stride = max(1, len(per_origin) // 200)
mae_by_origin = [
    {
        "origin": pd.Timestamp(o).strftime("%d %b %H:%M"),
        "model": num(per_origin.loc[o]),
        "baseline": num(per_origin_b.loc[o]),
    }
    for o in list(per_origin.index)[::mae_stride]
]

payload = {
    "generatedFrom": data_source,
    "originCount": len(origins),
    "origins": exported,
    "metrics": metrics,
    "maeByOrigin": mae_by_origin,
    "topDrivers": str(fc["top_drivers"].iloc[0]),
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(payload, separators=(",", ":")))
print(f"Wrote {OUT} ({OUT.stat().st_size/1024:.0f} KiB), source={data_source}, "
      f"{len(exported)} origins of {len(origins)}")
