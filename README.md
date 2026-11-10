# Imbalance price forecast - demo for Turku Energia

A working draft of the imbalance price forecast and situational view from
**Turku Energia's challenge at the Since AI 2026 hackathon** (Turku, 6-8 Nov
2026), prepared as a draft for **Team Predictable** (Toki Mohammad Tahmid, Viknapergash
Guraiah, Osman Berkay Guvenc) after the event as a basis for discussion with
Turku Energia.

It implements the challenge's core shape:

- **Level 1**: from every full-hour forecast origin, a 10-hour forecast
  (40 quarter-hours): P50 price with a P10-P90 interval, probabilities for
  the balancing-need direction (up / down / no dominating direction), and
  probabilities for exceptional prices (> spot + 50 and < spot - 50 EUR/MWh),
  written in the briefing-derived 13-column `forecast_backtest.csv` format.
- **Level 2**: a Streamlit situational view for an energy expert - what to
  check now and in the next 10 hours, alerts with plain-language reasoning,
  and past performance with the viewing time movable into the past.
- **Data availability rule**: every forecast uses only information that
  would have been published at that forecast origin (simplified publication
  delays in `features.py`; the real package used the binding per-row column
  `first_usable_origin_utc`).

## What data this demo uses (please read)

**By default: synthetic sample data, generated locally by `sample_data.py`.**
It imitates the shape of the real series (daily patterns, wind/solar
behaviour, price spikes) but every number is invented. Any metric the
backtest prints says something about the *code*, not about real forecasting
skill. The dashboard shows a permanent banner about this.

Why not real data:

1. **Nord Pool day-ahead spot price** was licensed to Turku Energia for
   hackathon use only. The rules said: don't share it, don't put it in
   repos, delete it after the event. So it is not here and must not be
   added. Turku Energia would need to supply it (or a licence) for real runs.
2. **The hackathon starter package** (Fingrid + ENTSO-E extracts prepared by
   Turku Energia) was only available inside sinceai.app during the event.
3. **Fingrid open data** is free (Creative Commons Attribution) but needs a
   personal API key: register at <https://data.fingrid.fi/en/instructions>,
   then see `fingrid_data.py`. Verified dataset: **319 = imbalance price**
   (15 min). Other series (mFRR/aFRR, consumption, production, cross-border
   flows, day-ahead forecasts 165/246/247) are on the same portal.

Real-data operation is not implemented yet. It needs licensed spot data, source
mapping into the feature schema, archived publication/forecast vintages,
row-level availability checks and independent out-of-sample validation.
`fingrid_data.py` is a downloader scaffold, not a tested live integration.

## Quick start

Windows: install Python 3.11+, download and unzip this repo, then double-click
`start_windows.bat`. It installs dependencies, runs the demo, checks the CSV
and opens the dashboard. Internet is needed for dependency installation.
The app runs locally, not on a public website.


```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python sample_data.py            # generates the synthetic data (data/, git-ignored)
python backtest.py               # trains, forecasts the test period, scores vs baseline
python -m unittest -v            # checks horizon, publication boundary and future-actual isolation
python validate_demo.py          # local format check, not the official validator
streamlit run app.py             # situational view at http://localhost:8501
```

`backtest.py` mirrors the official set-up: training origins 3 Jun - 3 Oct 2026 (every second hour),
then 673 hourly forecast origins over the test period 5 Oct - 1 Nov 2026,
and writes `results/forecast_backtest.csv` in the briefing-derived format (not checked with the unavailable official validator) plus
`results/metrics_summary.csv` (MAE, mean pinball loss for P10/P90, interval
coverage, Brier scores for exceptional prices and direction - model vs the
persistence baseline, the same comparison the challenge used).

## Repository layout

| File | Purpose |
| --- | --- |
| `sample_data.py` | Synthetic data generator (clearly labelled, reproducible, seed fixed) |
| `features.py` | Feature engineering + the data availability rule |
| `model.py` | Quantile gradient boosting (P10/P50/P90), direction & spike classifiers, persistence baseline, challenge metrics |
| `backtest.py` | Trains, forecasts 673 origins, scores model vs baseline |
| `app.py` | Streamlit situational view (next 10 h, alerts, past performance, time travel) |
| `fingrid_data.py` | Optional: fetch real Fingrid open data with your own API key |

`data/`, `results/`, `models/` are git-ignored on purpose: generated at run
time, and a guard so no real market data (which may be licence-restricted)
ever gets committed.

## Honest limits

- Metrics on the sample data are pipeline checks, not accuracy claims.
- `top_drivers` are training-sample permutation importances, not
  per-forecast explanations or causal drivers.
- Alert thresholds and wording in the dashboard are placeholders to tune
  with Turku Energia's experts.
- The imbalance price method changed on 1 June 2026 (volume-weighted
  mFRR+aFRR). Older data is not directly comparable; this generator is only illustrative,
  not an implementation of the settlement formula.

## Licence

Private development draft. No open-source licence has been selected. No market data is included or redistributed.
Fingrid open data: Creative Commons Attribution, source data.fingrid.fi.
This project is not affiliated with or endorsed by Turku Energia or Since AI.
