"""Situational view for the imbalance price forecast (the challenge's Level 2).

Run:
    python backtest.py      # once, builds results/ on the sample data
    streamlit run app.py

Audience: an energy expert (shift supervisor / electricity trading
specialist), not a data scientist. Two time spans, as the briefing asks:
what to check now / in the next 10 hours, and how the forecast performed in
the past, with the viewing time movable into the past.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
RESULTS = BASE_DIR / "results"
REAL_RESULTS = RESULTS / "real_v4" if (RESULTS / "real_v4" / "forecast_test_real.csv").exists() else RESULTS / "real"

st.set_page_config(page_title="Imbalance price forecast - demo", layout="wide")

_real_files = ("forecast_test_real.csv", "baseline_test_real.csv", "metrics_summary_real.csv")
real_available = all((REAL_RESULTS / name).exists() for name in _real_files)

if real_available:
    source = st.sidebar.radio(
        "Data source",
        ["Real Fingrid open data", "Synthetic sample data"],
        index=0,
        help="Real: Fingrid open data (CC-BY), backtested with publication-delay rules. Synthetic: generated numbers for pipeline checks.",
    )
else:
    source = "Synthetic sample data"

if source == "Real Fingrid open data":
    st.warning(
        "Data source: **REAL FINGRID OPEN DATA** (Creative Commons Attribution, data.fingrid.fi) - "
        "a retrospective benchmark on historical data, with availability assumptions "
        "(availability.py). Not a live forecast. ENTSO-E supplies authorised day-ahead spot data. "
        "Publication timestamps and historical forecast vintages are not verified. The nominal "
        "80% band covered about 90% in October, so it was too wide."
    )
else:
    st.warning(
        "Data source: **SYNTHETIC SAMPLE DATA** - generated numbers that imitate the "
        "shape of the real series. Metrics below measure the code on that data, not "
        "real forecasting skill."
    )

st.title("Imbalance price forecast - situational view (demo)")
st.caption(
    "Draft for Turku Energia's imbalance price challenge, Team Predictable. "
    "Forecast: P50 with P10-P90 interval, 40 quarter-hours ahead from each full-hour origin."
)


@st.cache_data
def load_results(source: str):
    if source == "Real Fingrid open data":
        fc_path = REAL_RESULTS / "forecast_test_real.csv"
        bl_path = REAL_RESULTS / "baseline_test_real.csv"
        mt_path = REAL_RESULTS / "metrics_summary_real.csv"
    else:
        fc_path = RESULTS / "forecast_test.csv"
        bl_path = RESULTS / "baseline_test.csv"
        mt_path = RESULTS / "metrics_summary.csv"
    if not (fc_path.exists() and bl_path.exists() and mt_path.exists()):
        return None, None, None
    fc = pd.read_csv(fc_path, parse_dates=["delivery_start_utc"])
    bl = pd.read_csv(bl_path, parse_dates=["delivery_start_utc"])
    mt = pd.read_csv(mt_path)
    if "period" in mt.columns:
        mt = mt[mt["period"] == "test"].drop(columns=["period"], errors="ignore")
    return fc, bl, mt


fc, bl, metrics = load_results(source)
if fc is None:
    st.info("No results yet for this data source. Run `python backtest.py` (synthetic) or "
            "`python spot_experiment.py` (real, after `python fingrid_data.py --download --build`), then reload.")
    st.stop()

origins = sorted(fc["forecast_origin_utc"].unique())
origin_labels = {o: pd.Timestamp(o).strftime("%a %d %b %Y, %H:%M UTC") for o in origins}

tab_now, tab_alerts, tab_past, tab_about = st.tabs(
    ["Next 10 hours", "What to check now", "Past performance", "About"]
)

# ---------------------------------------------------------------- helpers
def forecast_chart(fc_rows: pd.DataFrame, bl_rows: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    x = fc_rows["delivery_start_utc"]
    fig.add_trace(
        go.Scatter(
            x=list(x) + list(x[::-1]),
            y=list(fc_rows["p90"]) + list(fc_rows["p10"][::-1]),
            fill="toself",
            fillcolor="rgba(0,116,217,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="P10-P90 interval",
            hoverinfo="skip",
        )
    )
    fig.add_trace(go.Scatter(x=x, y=fc_rows["p50"], name="P50 forecast", line=dict(color="#0074D9", width=2)))
    fig.add_trace(
        go.Scatter(x=x, y=bl_rows["p50"], name="Baseline (last price)", line=dict(color="#999999", dash="dot"))
    )
    fig.add_trace(
        go.Scatter(x=x, y=fc_rows["actual_price"], name="Actual (known later)", line=dict(color="#2ECC40", dash="dash"))
    )
    if fc_rows["actual_spot"].notna().any():
        fig.add_trace(
            go.Scatter(x=x, y=fc_rows["actual_spot"], name="Spot", line=dict(color="#FF851B", dash="dashdot"))
        )
    fig.update_layout(title=title, yaxis_title="EUR/MWh", height=420, legend_orientation="h")
    return fig


def pick_origin(key: str) -> str:
    return st.selectbox(
        "Forecast origin (move the viewing time into the past)",
        origins,
        index=len(origins) - 1,
        format_func=lambda o: origin_labels[o],
        key=key,
    )


def rows_for(df: pd.DataFrame, origin: str) -> pd.DataFrame:
    return df[df["forecast_origin_utc"] == origin].sort_values("step")


# ---------------------------------------------------------------- tab: now
with tab_now:
    if source == "Real Fingrid open data":
        dl_label, dl_path, dl_name = "Download real-data forecast CSV", REAL_RESULTS / "forecast_backtest_real.csv", "real_forecast_backtest.csv"
    else:
        dl_label, dl_path, dl_name = "Download synthetic forecast CSV", RESULTS / "forecast_backtest.csv", "synthetic_forecast_backtest.csv"
    if dl_path.exists():
        st.download_button(dl_label, dl_path.read_bytes(), file_name=dl_name, mime="text/csv")
    origin = pick_origin("origin_now")
    fc_rows, bl_rows = rows_for(fc, origin), rows_for(bl, origin)
    st.plotly_chart(forecast_chart(fc_rows, bl_rows, f"Forecast from {origin_labels[origin]}"), use_container_width=True)

    spike_cols = ["p_spike_high", "p_spike_low"]
    spikes_shown = fc_rows["p_spike_high"].notna().any()
    c1, c2, c3 = st.columns(3)
    if spikes_shown:
        c1.metric("Highest spike risk (> spot + 50)", f"{fc_rows['p_spike_high'].max():.0%}")
        c2.metric("Highest low-price risk (< spot - 50)", f"{fc_rows['p_spike_low'].max():.0%}")
    else:
        c1.metric("Highest spike risk", "n/a")
        c2.metric("Highest low-price risk", "n/a")
        st.caption("Exceptional-price probabilities need the Nord Pool spot price, which is not open data - "
                   "not supplied for this run, so they are empty on purpose.")
    width = (fc_rows["p90"] - fc_rows["p10"]).mean()
    c3.metric("Mean P10-P90 width", f"{width:.0f} EUR/MWh")
    _ = spike_cols

    st.subheader("Direction of the balancing need")
    dir_df = fc_rows[["delivery_start_utc", "p_up", "p_down", "p_none"]].set_index("delivery_start_utc")
    direction_plot = go.Figure()
    for field, label, color in [('p_up', 'Up', '#EF553B'), ('p_down', 'Down', '#0074D9'), ('p_none', 'None', '#A0CBE8')]:
        direction_plot.add_trace(go.Bar(x=dir_df.index.strftime('%d %b %H:%M UTC'),
                                        y=dir_df[field], name=label, marker_color=color))
    direction_plot.update_layout(barmode='stack', height=330, yaxis_title='Probability',
                                 xaxis_title='Delivery quarter-hour (UTC)',
                                 xaxis=dict(tickmode='array', tickvals=dir_df.index.strftime('%d %b %H:%M UTC')[::8], tickangle=0),
                                 legend=dict(orientation='h', y=1.15))
    st.plotly_chart(direction_plot, use_container_width=True)
    st.caption(
        f"Data cutoff: {fc_rows['data_cutoff_utc'].iloc[0]} - "
        f"main drivers (global): {fc_rows['top_drivers'].iloc[0]}"
    )

# ------------------------------------------------------------- tab: alerts
with tab_alerts:
    origin_a = pick_origin("origin_alerts")
    fc_rows = rows_for(fc, origin_a)
    drivers = fc_rows["top_drivers"].iloc[0]
    st.subheader(f"Situation at {origin_labels[origin_a]}")

    fired = False
    hi = fc_rows[fc_rows["p_spike_high"] >= 0.15] if fc_rows["p_spike_high"].notna().any() else fc_rows.iloc[0:0]
    if len(hi):
        fired = True
        w0 = hi["delivery_start_utc"].min().strftime("%H:%M")
        w1 = (hi["delivery_start_utc"].max() + pd.Timedelta(minutes=15)).strftime("%H:%M")
        st.error(
            f"Demo high-price risk exceeds the threshold for {w0}-{w1} UTC. "
            f"Global model associations: {drivers} (not a local explanation). "
            "Uncertainty is shown by the P10-P90 band; if it is wide, the production "
            "plan and market position should be checked."
        )
    lo = fc_rows[fc_rows["p_spike_low"] >= 0.15] if fc_rows["p_spike_low"].notna().any() else fc_rows.iloc[0:0]
    if len(lo):
        fired = True
        w0 = lo["delivery_start_utc"].min().strftime("%H:%M")
        w1 = (lo["delivery_start_utc"].max() + pd.Timedelta(minutes=15)).strftime("%H:%M")
        st.warning(
            f"Demo low-price risk exceeds the threshold for {w0}-{w1} UTC. "
            f"Global model associations: {drivers} (not a local explanation). Worth checking whether "
            "flexible consumption or production can be shifted into the window."
        )
    dom = fc_rows.iloc[0]
    dir_txt = max((("up-regulation", dom["p_up"]), ("down-regulation", dom["p_down"]), ("no dominating direction", dom["p_none"])), key=lambda t: t[1])
    if not fired:
        st.success(
            f"No spike alerts at the demo threshold in the next 10 hours. Most likely direction: "
            f"{dir_txt[0]} ({dir_txt[1]:.0%})."
        )
    st.info(
        "Suggestion logic (demo): alert when a spike probability reaches 15% anywhere in "
        "the horizon, then attach the model's main global drivers and an action hint. "
        "Thresholds and wording are placeholders to tune with Turku Energia's experts."
    )

# --------------------------------------------------------------- tab: past
with tab_past:
    st.subheader("Retrospective benchmark: 5 Oct - 1 Nov 2026")
    wanted = ["MAE_p50_eur_mwh", "pinball_mean", "coverage_p10_p90", "brier_spike_high", "brier_spike_low", "brier_direction"]
    show = metrics.set_index("model")
    cols = [c for c in wanted if c in show.columns and show[c].notna().any()]
    st.dataframe(show[cols].style.format("{:.3f}"), use_container_width=True)
    caption = "Coverage target is about 0.80. The ML point forecast does not beat the spot-only baseline on the October retrospective benchmark."
    if source == "Real Fingrid open data":
        caption += " Real Fingrid open data, time-based backtest; exceptional-price metrics appear only with a lawful spot series."
    else:
        caption += " All figures on synthetic sample data - they validate the pipeline, not real skill."
    st.caption(caption)

    per_origin = (
        fc.assign(abs_err=(fc["p50"] - fc["actual_price"]).abs())
        .groupby("forecast_origin_utc")["abs_err"]
        .mean()
    )
    per_origin_b = (
        bl.assign(abs_err=(bl["p50"] - bl["actual_price"]).abs())
        .groupby("forecast_origin_utc")["abs_err"]
        .mean()
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=per_origin.index, y=per_origin.values, name="Model MAE / origin"))
    fig.add_trace(go.Scatter(x=per_origin_b.index, y=per_origin_b.values, name="Baseline MAE / origin"))
    fig.update_layout(yaxis_title="MAE (EUR/MWh)", height=350, legend_orientation="h")
    st.plotly_chart(fig, use_container_width=True)

    origin_p = pick_origin("origin_past")
    st.plotly_chart(
        forecast_chart(rows_for(fc, origin_p), rows_for(bl, origin_p), f"Replay: {origin_labels[origin_p]}"),
        use_container_width=True,
    )

# --------------------------------------------------------------- tab: about
with tab_about:
    st.markdown(
        """
**What this is.** A working draft of the imbalance price forecast and
situational view from Turku Energia's challenge at Since AI 2026, prepared for
Team Predictable after the event as a basis for discussion.

**Data.** Two sources, chosen in the sidebar. Synthetic sample data
(generated locally by `sample_data.py`) exists because the hackathon starter
package was only on sinceai.app during the event and the Nord Pool spot
series in it was licensed for hackathon use only. The real source is Fingrid
open data (CC-BY): `fingrid_data.py` downloads it with your own free API key,
`spot_experiment.py` trains and backtests it with disjoint selection and calibration periods. ENTSO-E provides authorised FI day-ahead spot. October is a retrospective benchmark, already examined in earlier experiments.

**Method.** Quantile gradient boosting for P10/P50/P90 (step number as a
feature, one model for the whole 10-hour horizon), gradient boosting
classifiers for direction and exceptional prices, persistence baseline for
comparison. Availability is approximated with conservative rules in `availability.py`. Without historical publication timestamps and vintages, this cannot prove production-time availability. The price model is selected by September MAE; probability shrinkage and interval expansion use separate data.

**Challenge output.** `results/forecast_backtest.csv` follows the official
13-column format (673 origins x 40 steps, P10-P90, direction and spike
probabilities, optional data cutoff and top drivers).
"""
    )
