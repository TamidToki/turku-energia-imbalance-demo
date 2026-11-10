"""Run the backtest: train on Jun-Sep, forecast the challenge test period, score it.

Usage:
    python backtest.py

Outputs (all git-ignored, regenerated every run):
    results/forecast_backtest.csv   model forecast in the challenge's 13-column format
    results/forecast_test.csv       model + baseline predictions with actuals (used by app.py)
    results/metrics_summary.csv     model vs baseline metrics
    models/forecaster.joblib        trained model, for future real-data use

On the sample data the metrics only show that the pipeline works. Run it on
real data before quoting any number to Turku Energia.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from features import SeriesView, build_matrix, build_origin_frame
from model import ImbalanceForecaster, PersistenceBaseline, metric_table
from sample_data import load_or_create_sample_data

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

# Challenge periods from the official briefing
TRAIN_START = pd.Timestamp("2026-06-03T00:00:00Z")  # 24h+ of history exists by then
TRAIN_END = pd.Timestamp("2026-10-03T00:00:00Z")
TEST_ORIGINS_START = pd.Timestamp("2026-10-04T21:00:00Z")  # = 5 Oct 00:00 Finnish time
TEST_ORIGINS_END = pd.Timestamp("2026-11-01T21:00:00Z")
TRAIN_ORIGIN_STRIDE_HOURS = 2  # every 2nd hour keeps training fast


def main() -> None:
    print("Loading data (synthetic sample data is generated on first run)...")
    df = load_or_create_sample_data()
    view = SeriesView(df)

    train_origins = list(
        pd.date_range(TRAIN_START, TRAIN_END, freq=f"{TRAIN_ORIGIN_STRIDE_HOURS}h")
    )
    test_origins = list(pd.date_range(TEST_ORIGINS_START, TEST_ORIGINS_END, freq="1h"))
    print(f"Training origins: {len(train_origins)} | test origins: {len(test_origins)} "
          f"(challenge used 673)")

    print("Building training features...")
    train = build_matrix(view, train_origins)
    print(f"Training rows: {len(train)}")

    print("Training model and baseline...")
    model = ImbalanceForecaster().fit(train)
    baseline = PersistenceBaseline().fit(train)
    print("Top global drivers (P50 model):", ", ".join(model.top_drivers_))

    print("Forecasting the test period...")
    model_rows, base_rows = [], []
    for o in test_origins:
        frame = build_origin_frame(view, o)
        model_rows.append(model.predict(frame))
        base_rows.append(baseline.predict(frame))
    fc = pd.concat(model_rows, ignore_index=True)
    fb = pd.concat(base_rows, ignore_index=True)

    # Attach actuals for scoring and the dashboard
    actuals = pd.concat(
        [
            build_origin_frame(view, o)[
                ["forecast_origin_utc", "step", "y_price", "y_direction", "y_spike_high", "y_spike_low", "actual_spot"]
            ]
            for o in test_origins
        ],
        ignore_index=True,
    )
    actuals["forecast_origin_utc"] = actuals["forecast_origin_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    def with_actuals(pred: pd.DataFrame) -> pd.DataFrame:
        m = pred.merge(actuals, on=["forecast_origin_utc", "step"], how="left")
        return m.rename(
            columns={
                "y_price": "actual_price",
                "y_direction": "actual_direction",
                "y_spike_high": "actual_spike_high",
                "y_spike_low": "actual_spike_low",
            }
        )

    fc_a, fb_a = with_actuals(fc), with_actuals(fb)
    m_model, m_base = metric_table(fc_a), metric_table(fb_a)

    RESULTS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    fc.to_csv(RESULTS_DIR / "forecast_backtest.csv", index=False)
    fc_a.assign(model="gradient_boosting").to_csv(RESULTS_DIR / "forecast_test.csv", index=False)
    fb_a.assign(model="baseline").to_csv(RESULTS_DIR / "baseline_test.csv", index=False)
    summary = pd.DataFrame([{"model": "gradient_boosting", **m_model}, {"model": "baseline", **m_base}])
    summary.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)
    joblib.dump(model, MODELS_DIR / "forecaster.joblib")

    print("\n=== Metrics on the TEST period (synthetic sample data!) ===")
    print(summary.to_string(index=False))
    print("\nReminder: these numbers describe the code on synthetic data,")
    print("not real forecasting skill. forecast_backtest.csv is in results/.")


if __name__ == "__main__":
    main()
