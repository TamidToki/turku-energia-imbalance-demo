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

Two data sources, chosen in the dashboard sidebar.

**1. Real Fingrid open data (Creative Commons Attribution)** - the default
once downloaded. `fingrid_data.py` fetches eight verified 15-minute datasets
from data.fingrid.fi with your own free API key and builds
`data/real_inputs.csv`; `real_data.py` trains and backtests with the
challenge's time-based setup: training origins 3 Jun - 3 Oct 2026, then the
validation period 7 Sep - 4 Oct and the test period 5 Oct - 1 Nov (673
hourly origins), model vs the persistence baseline. Every forecast uses only
data published before its origin (conservative delays in `availability.py`).

**2. Synthetic sample data**, generated locally by `sample_data.py`. It
imitates the shape of the real series but every number is invented. Metrics
on it say something about the *code*, not about real forecasting skill.

Still unavailable, on purpose:

1. **Nord Pool day-ahead spot price** was licensed to Turku Energia for
   hackathon use only. It is not here and must not be added from unofficial
   sources. Without it there are no honest exceptional-price labels, so
   `p_spike_high` / `p_spike_low` stay EMPTY rather than faked. Supply a
   lawful spot CSV (`timestamp_utc,value`) once via
   `python fingrid_data.py --build --spot your_spot.csv` to enable them
   (`data/` is git-ignored, so it cannot be committed by accident).
2. **15-minute-updated forecasts** (Fingrid 166/245) and **archived forecast
   vintages**: the public API returns only the latest value per
   quarter-hour, so backtesting with them would leak future information.
   Vintages would have to come from Turku Energia or Fingrid.

The imbalance price method changed on 1 June 2026 (volume-weighted
mFRR+aFRR); the real pipeline trains only on data from 1 June 2026 onward.

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

Real data (one extra, one-time step): register for a free Fingrid API key
(the numbered steps are in the `fingrid_data.py` docstring), then:

```bash
export FINGRID_API_KEY="your-key"   # Windows: set FINGRID_API_KEY=your-key
                                     # or one line in a local .env (git-ignored)
python fingrid_data.py --download --build   # fetch 8 datasets, build real_inputs.csv
python real_data.py                          # real backtest, model vs baseline
streamlit run app.py                         # sidebar now offers the real source
```

Never commit the key or any real market data: `data/` and `.env` are
git-ignored on purpose.

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
| `availability.py` | The data availability rule: conservative publication delays shared by both data sources |
| `features.py` | Feature engineering; applies the availability rule per forecast origin |
| `model.py` | Quantile gradient boosting (P10/P50/P90), direction & spike classifiers, persistence baseline, challenge metrics |
| `backtest.py` | Synthetic-data backtest: trains, forecasts 673 origins, scores model vs baseline |
| `fingrid_data.py` | Real Fingrid open data: throttled API client, dataset registry, `real_inputs.csv` builder + quality report |
| `real_data.py` | Real-data backtest: trains on 1 Jun - 4 Oct, scores validation and test periods vs baseline |
| `app.py` | Streamlit situational view (next 10 h, alerts, past performance, time travel, data-source switch) |

`data/`, `results/`, `models/` are git-ignored on purpose: generated at run
time, and a guard so no real market data (which may be licence-restricted)
ever gets committed.

## Honest limits

- Metrics on the sample data are pipeline checks, not accuracy claims.
- Real-data metrics are a genuine time-based backtest on published Fingrid
  history, but the publication times are conservative ASSUMPTIONS
  (`availability.py`): the public API exposes neither per-row publication
  timestamps nor forecast vintages, so they cannot be exact. The starter
  package's binding `first_usable_origin_utc` was only inside sinceai.app.
- Without a lawful spot series there are no exceptional-price labels:
  spike probabilities are empty (NaN), not zero, and spike metrics are not
  reported.
- `top_drivers` are training-sample permutation importances, not
  per-forecast explanations or causal drivers.
- Alert thresholds and wording in the dashboard are placeholders to tune
  with Turku Energia's experts.
- The imbalance price method changed on 1 June 2026 (volume-weighted
  mFRR+aFRR). Older data is not directly comparable; the real pipeline
  trains from 1 June 2026 only, and the synthetic generator is only
  illustrative, not an implementation of the settlement formula.

## Licence

Private development draft. No open-source licence has been selected. No market data is included or redistributed.
Fingrid open data: Creative Commons Attribution, source data.fingrid.fi.
This project is not affiliated with or endorsed by Turku Energia or Since AI.

