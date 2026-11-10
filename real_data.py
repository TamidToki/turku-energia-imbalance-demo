"""Train and backtest on REAL Fingrid open data, with time-based validation.

Prerequisites (once):
    python fingrid_data.py --download --build      # needs your FINGRID_API_KEY
    # optional, for exceptional-price metrics: --spot your_lawful_spot.csv

Then:
    python real_data.py

What it does:
  - trains on forecast origins 3 Jun - 3 Oct 2026 (every 2nd hour), using
    only data published before each origin (availability.py)
  - scores the validation period 7 Sep - 4 Oct 2026 and the challenge test
    period 5 Oct - 1 Nov 2026 (673 hourly origins), model vs the same
    persistence baseline the challenge used
  - never trains on test-period data and never uses a row before its
    assumed publication time

Outputs (git-ignored, regenerated every run):
    results/real/forecast_backtest_real.csv   challenge's 13-column format
    results/real/forecast_test_real.csv       model predictions + actuals (dashboard)
    results/real/baseline_test_real.csv       baseline predictions + actuals
    results/real/metrics_summary_real.csv     model vs baseline, both periods

Honest notes:
  - without a spot series there are no exceptional-price labels, so spike
    probabilities are NaN and spike metrics are not reported (not zero!)
  - publication times are conservative assumptions (availability.py); the
    Fingrid API exposes no per-row publication timestamps or vintages
  - quarters whose actual price is missing are excluded from scoring and
    counted; origins whose latest known price is missing are skipped and
    counted
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import SeriesView, build_matrix, build_origin_frame
from fingrid_data import META_JSON, REAL_INPUTS_CSV, load_real_inputs
from model import ImbalanceForecaster, PersistenceBaseline, metric_table

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results" / "real"
MODELS_DIR = BASE_DIR / "models"

# Periods from the official briefing
TRAIN_START = pd.Timestamp("2026-06-03T00:00:00Z")
TRAIN_END = pd.Timestamp("2026-10-03T00:00:00Z")
TRAIN_ORIGIN_STRIDE_HOURS = 2
VALIDATION_ORIGINS_START = pd.Timestamp("2026-09-07T00:00:00Z")
VALIDATION_ORIGINS_END = pd.Timestamp("2026-10-03T23:00:00Z")
TEST_ORIGINS_START = pd.Timestamp("2026-10-04T21:00:00Z")  # = 5 Oct 00:00 Finnish time
TEST_ORIGINS_END = pd.Timestamp("2026-11-01T21:00:00Z")

SCORED_KEYS = [
    "MAE_p50_eur_mwh",
    "pinball_mean",
    "coverage_p10_p90",
    "brier_spike_high",
    "brier_spike_low",
    "brier_direction",
]


def predict_origins(model, baseline, view: SeriesView, origins):
    """Predict every origin; skip (and count) origins without a latest price."""
    model_rows, base_rows, frames, skipped = [], [], [], []
    for o in origins:
        frame = build_origin_frame(view, o)
        if pd.isna(frame["last_imb"].iloc[0]):
            skipped.append(o)
            continue
        model_rows.append(model.predict(frame))
        base_rows.append(baseline.predict(frame))
        frames.append(frame)
    return (
        pd.concat(model_rows, ignore_index=True),
        pd.concat(base_rows, ignore_index=True),
        pd.concat(frames, ignore_index=True),
        skipped,
    )


def attach_actuals(pred: pd.DataFrame, frames: pd.DataFrame, has_spot: bool) -> pd.DataFrame:
    act = frames[
        ["forecast_origin_utc", "step", "y_price", "y_direction", "y_spike_high", "y_spike_low", "actual_spot"]
    ].copy()
    act["forecast_origin_utc"] = act["forecast_origin_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    merged = pred.merge(act, on=["forecast_origin_utc", "step"], how="left")
    merged = merged.rename(
        columns={
            "y_price": "actual_price",
            "y_direction": "actual_direction",
            "y_spike_high": "actual_spike_high",
            "y_spike_low": "actual_spike_low",
        }
    )
    if not has_spot:
        # The 0/1 columns in the input table mean "no labels", not reality.
        merged["actual_spike_high"] = np.nan
        merged["actual_spike_low"] = np.nan
    return merged


def scored_metrics(pred_with_actuals: pd.DataFrame, has_spot: bool) -> tuple[dict, int, int]:
    """Metrics on rows with a known actual price. Returns (metrics, n_scored, n_total)."""
    df = pred_with_actuals
    n_total = len(df)
    df = df.dropna(subset=["actual_price"])
    metrics = metric_table(df)
    metrics = {k: (np.nan if pd.isna(v) else v) for k, v in metrics.items()}
    if not has_spot:
        metrics["brier_spike_high"] = np.nan
        metrics["brier_spike_low"] = np.nan
    return metrics, int(len(df)), int(n_total)


def main() -> None:
    if not REAL_INPUTS_CSV.exists():
        raise SystemExit(
            "data/real_inputs.csv not found. Run first:\n"
            "  python fingrid_data.py --download --build   (needs your Fingrid API key)"
        )
    meta = json.loads(META_JSON.read_text()) if META_JSON.exists() else {}
    has_spot = bool(meta.get("has_spot", False))
    print(f"Loading real Fingrid data (built {meta.get('built_at_utc', '?')}, spot={'yes' if has_spot else 'NO'})...")
    df = load_real_inputs()
    view = SeriesView(df)

    train_origins = list(pd.date_range(TRAIN_START, TRAIN_END, freq=f"{TRAIN_ORIGIN_STRIDE_HOURS}h"))
    val_origins = list(pd.date_range(VALIDATION_ORIGINS_START, VALIDATION_ORIGINS_END, freq="1h"))
    test_origins = list(pd.date_range(TEST_ORIGINS_START, TEST_ORIGINS_END, freq="1h"))
    print(f"Origins - training: {len(train_origins)} | validation: {len(val_origins)} | test: {len(test_origins)}")

    print("Building training features...")
    train = build_matrix(view, train_origins)
    n_before = len(train)
    train = train.dropna(subset=["y_price"])
    print(f"Training rows: {len(train)} (dropped {n_before - len(train)} with a missing actual price)")
    if train.empty:
        raise SystemExit("No usable training rows - check data/real_quality_report.md for gaps.")

    print("Training model and baseline on real data...")
    model = ImbalanceForecaster(spikes_available=has_spot).fit(train)
    baseline = PersistenceBaseline(spikes_available=has_spot).fit(train)
    print("Top global drivers (P50 model):", ", ".join(model.top_drivers_))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "forecaster_real.joblib")

    summaries = []
    period_preds = {}
    for label, origins in (("validation", val_origins), ("test", test_origins)):
        print(f"Forecasting the {label} period...")
        fc, fb, frames, skipped = predict_origins(model, baseline, view, origins)
        fc_a = attach_actuals(fc, frames, has_spot)
        fb_a = attach_actuals(fb, frames, has_spot)
        m_model, n_model, n_tot = scored_metrics(fc_a, has_spot)
        m_base, _, _ = scored_metrics(fb_a, has_spot)
        summaries.append({"period": label, "model": "gradient_boosting", "n_scored": n_model,
                          "n_rows": n_tot, **m_model})
        summaries.append({"period": label, "model": "baseline", "n_scored": n_model,
                          "n_rows": n_tot, **m_base})
        if skipped:
            print(f"  note: skipped {len(skipped)} {label} origins with no published price history")
        period_preds[label] = fc_a
        if label == "test":
            fc.to_csv(RESULTS_DIR / "forecast_backtest_real.csv", index=False)
            fc_a.assign(model="gradient_boosting").to_csv(RESULTS_DIR / "forecast_test_real.csv", index=False)
            fb_a.assign(model="baseline").to_csv(RESULTS_DIR / "baseline_test_real.csv", index=False)

    # Split-conformal calibration of the P10-P90 band, fitted on the
    # validation period only, then applied to the untouched test period.
    val_a = period_preds["validation"].dropna(subset=["actual_price"])
    s = np.maximum(val_a["p10"] - val_a["actual_price"], val_a["actual_price"] - val_a["p90"])
    k = int(np.ceil((len(s) + 1) * 0.80))
    q_shift = float(np.sort(s.to_numpy())[min(k, len(s)) - 1])
    print(f"Conformal band shift fitted on validation: {q_shift:+.2f} EUR/MWh")
    for label in ("validation", "test"):
        cal = period_preds[label].copy()
        cal["p10"] = cal["p10"] - q_shift
        cal["p90"] = cal["p90"] + q_shift
        m_cal, n_cal, n_tot_cal = scored_metrics(cal, has_spot)
        summaries.append({"period": label, "model": "gbm_calibrated", "n_scored": n_cal,
                          "n_rows": n_tot_cal, **m_cal})

    summary = pd.DataFrame(summaries)
    summary.to_csv(RESULTS_DIR / "metrics_summary_real.csv", index=False)

    pd.set_option("display.width", 160)
    cols = ["period", "model", "n_scored"] + SCORED_KEYS
    print("\n=== Metrics on REAL Fingrid data (time-based backtest) ===")
    print(summary[cols].to_string(index=False))
    if not has_spot:
        print("\nExceptional-price metrics are empty ON PURPOSE: the Nord Pool spot price")
        print("is not open data, so there are no honest spike labels yet. Supply a lawful")
        print("spot CSV (python fingrid_data.py --build --spot spot_fi.csv) to enable them.")
    print("\nReference: the challenge's official baseline on its own validation data was")
    print("MAE 66.20, pinball 24.18, coverage 0.68 - different data mix (it included the")
    print("licensed spot series), so compare model-vs-baseline above, not across tables.")


if __name__ == "__main__":
    main()
