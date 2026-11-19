# Imbalance price forecast: research demo

A 10-hour forecast of Finland's imbalance price, with P10/P50/P90,
direction and exceptional-price probabilities, plus an operator dashboard.
Research working draft by Toki Mohammad Tahmid. No hackathon starter-package
Nord Pool data is used or stored here.

## What the current result really says

The spot-enabled v4 experiment was run on 11 November 2026 using Fingrid
public history and authorised ENTSO-E Finland day-ahead prices.

| October retrospective benchmark | LightGBM gap model, calibrated | Persistence | Spot-only anchor |
| --- | ---: | ---: | ---: |
| P50 MAE, EUR/MWh (lower is better) | 30.69 | 41.40 | **30.25** |
| Mean P10/P90 pinball (lower) | **8.02** | 14.77 | 9.41 |
| P10-P90 coverage (nominal 80%) | 90.1% | 77.2% | 78.1% |
| High-spike Brier (lower) | 0.07512 | 0.07548 | 0.07548 |
| Low-spike Brier (lower) | 0.05095 | 0.06275 | 0.06275 |
| Direction Brier, 3-class sum (lower) | 0.43639 | 0.45815 | 0.45815 |

**The ML point forecast beats persistence by about 26%, but does not beat
simply forecasting the known spot price on October MAE.** Its quantile
loss is better than both baselines. High-spike skill is essentially absent;
low-spike and direction skill are modest. These are not trading results.

The nominal 80% interval is too wide overall (90.1% coverage), yet catches
only 37.8% of extreme-price outcomes. Extreme-outcome MAE is 109.57 EUR/MWh.
So a good overall average does not mean reliable protection against spikes.
When future spot is not yet published, MAE rises to 41.24 (272 forecast
rows), compared with 30.58 when spot is known (26,648 rows).

October was examined in earlier versions. It is a **retrospective
benchmark**, not an untouched final test. No claim of future accuracy,
production readiness, competition rank or profitability is made.

## Reproduce locally

Python 3.10+:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Free Fingrid key: register at https://data.fingrid.fi/en/instructions
# Set FINGRID_API_KEY in your environment or a local .env (git-ignored).
python fingrid_data.py --download

# ENTSO-E: register, request REST API access, generate token under My Account.
# Put the token in a private local file OUTSIDE this repo. Never commit it.
# Set ENTSOE_TOKEN_FILE to that file's path. The downloader does not print it.
python entsoe_data.py
python fingrid_data.py --build --spot data/entsoe/spot_fi.csv
python spot_experiment.py
python -m unittest -v
streamlit run app.py
```

The ENTSO-E token is kept in a private local file outside this
repository and is never committed. Downloaded data, trained models and result CSVs are
ignored by git and regenerated locally. Authorised source data is not
redistributed. The Windows launcher remains untested on Windows.

For a synthetic pipeline check only: `python backtest.py`, then
`streamlit run app.py`. Synthetic metrics are not real forecasting skill.
The older `real_data.py` path remains for comparison, but v4 is the
recommended research experiment and the dashboard prefers its outputs.

## Method and chronological split

- Training origins: 3 June through 30 August 2026 00:00 UTC, every 2 hours.
- Model selection: 1 September through 18 September 23:00 UTC, hourly.
- Calibration: 21 September through 2 October 23:00 UTC, hourly.
- Retrospective benchmark: 4 October 21:00 UTC through 1 November 21:00 UTC,
  673 hourly origins, each with 40 quarter-hours (26,920 forecast rows).
- Assertions require over 24 hours between each period's final target
  delivery and the next period's first origin. No random splits.

Compare HGB absolute-price, HGB gap-to-spot, LightGBM gap-to-spot, a fixed
50/50 HGB blend, and an HGB gap variant excluding all daily forecasts.
LightGBM gap wins September selection (MAE 43.10 versus spot-only 47.26
and persistence 68.51). Differences between ML variants are small, not
proof of one library's universal superiority.

The target is imbalance minus the **available anchor**. Where delivery
spot is known, that is spot. Otherwise it is the latest historical spot,
used consistently in both training and inference. Quantiles use native
pinball loss. Crossing quantiles are ordered. Direction and spike models
are HGB classifiers. Selection-period Brier chooses probability shrinkage
toward training priors. A separate calibration period supplies a scalar
band expansion. It is a heuristic, not a finite-sample conformal guarantee:
serial dependence and overlapping targets violate ordinary IID assumptions.

No classifier probabilities are fed into the quantile models, avoiding
in-sample stacking leakage. The model is NOT refit on calibration or October.

## Availability audit and limits

- Historical settlement features use a 90-minute delay from delivery start.
  This is a conservative assumption, not proof that final revised values
  existed at that time. Final-history revision bias remains unmeasured.
- Day-ahead spot becomes usable at D-1 14:00 Finnish local time, based on the **Finnish
  local delivery date**, not the UTC date. Daily forecasts use D-1 19:00 Finnish local time.
  These are assumptions; exceptional publication delays are not archived.
- All indexing is a continuous UTC quarter-hour grid. Local dates are used
  only for day-ahead availability. DST and late-evening boundaries are tested.
- Fingrid's public history lacks per-row publication timestamps and past
  forecast vintages. Even once-daily forecast series may have revisions.
  Excluding them is a sensitivity check, not a complete cure for final-history
  bias. The strict variant's October metrics are in `strict_sensitivity.json`;
  it was NOT promoted based on October results.
- Perturbation tests change unavailable future actuals and unpublished
  next-local-day spot/forecasts. Available features must remain unchanged.
  These tests check code behavior, not unknowable historic publication facts.
- Fingrid 376 down-regulation volumes are negative. Direction compares
  activation magnitudes. The earlier almost-perfect direction result was a
  label bug and is withdrawn. One missing activation quarter remains; price
  and spot have no missing quarters on the 14,880-row model input grid.
- Comparing these scores with teams' official scores is not apples-to-apples:
  sources, splits, publication rules and baselines differ.

**Needed before a production claim:** archive as-of data in real time,
measure revisions and publication delays, freeze a model prospectively,
then evaluate a genuinely new period. No live trading or automated action.

## Files

- `entsoe_data.py`: authenticated, cached FI day-ahead price downloader;
  sparse change-point curves are expanded at their declared resolution.
- `fingrid_data.py`: throttled public-data downloader and input quality report.
- `availability.py`, `features.py`: availability assumptions and features.
- `spot_model.py`, `spot_experiment.py`: current v4 model and evaluation.
- `model.py`, `real_data.py`, `bakeoff.py`: earlier models and comparisons.
- `app.py`: retrospective operator dashboard with source and caveat banners.
- `test_demo.py`, `test_real_data.py`, `test_spot.py`: 18 passing checks.
- `results/real_v4/` (local, ignored): selection table, baseline comparisons,
  metrics, split metadata, daily errors, sensitivity and stratified results,
  plus the 13-column forecast CSV and 673x40 output checks.

## Sources and review

Fingrid: https://data.fingrid.fi (CC-BY, attribution required).
ENTSO-E: https://transparency.entsoe.eu, API https://web-api.tp.entsoe.eu/api.
REST API guide: https://transparencyplatform.zendesk.com/hc/en-us/articles/15692855254548.
Data remain private and source terms still apply. Access is not permission
to redistribute a licensed starter package.

Gemini was consulted on the problem description and generic metrics only,
not credentials or source data. Useful review points included historical
revisions, DST boundaries, stratified errors and split embargoes. Advice was
checked independently: the models already used native quantile loss, and
training/inference already shared the fallback anchor. Ideas also align
with public team descriptions; no unlicensed team code was copied.

## Additional pre-October research (Argon / High review)

`python pre_october_cv.py` runs three expanding-window folds with >24h
embargoes after target deliveries. The October benchmark is physically
excluded. Mean-fold MAE: 75%-weighted LightGBM gap 31.83; spot-only 33.58;
simple 2-hour decaying-gap baseline 32.16; persistence 42.85. ML beats
spot-only in all three folds but loses to decaying-gap in August. These
are selection results, not new final-test accuracy. The small difference
from the simple baseline is a reason to be cautious about complexity.

`FRESH_EVALUATION_PROTOCOL.md` freezes a separate, eight-day Nov3-10
point-forecast check before downloading those Fingrid outcomes. Run
`fresh_evaluation.py` with `FINGRID_KEY_FILE` pointing to a secure local
API-key file. It leaves v4 inputs and results untouched. Do not promote
whichever model wins this new window or infer spike/interval skill from
point MAE. A positive eight-day result would still not establish annual
or production performance.

### Frozen November check: result

After the protocol was committed, the new Fingrid outcomes were downloaded
and scored once. Nov3-10, 192 hourly origins / 7,680 rows:

| Fixed point forecast | MAE EUR/MWh |
| --- | ---: |
| Selected LightGBM gap, 75% weight | **21.23** |
| Spot-only anchor | 24.92 |
| Simple 2h decaying gap | 24.07 |
| Persistence | 41.14 |

ML error was 14.8% lower than spot-only, 11.8% lower than decaying gap,
and 48.4% lower than persistence. It beat spot-only on 7/8 days, losing on
7 November. This positive new-window result does not erase October's
failure to beat spot-only, and eight days are too few to establish stable
performance. Only P50 point forecasts were evaluated. The existing
October dashboard remains unchanged; the separate new-window tab shows
this check when its local outputs exist.

No new model was selected after seeing these results. Further tuning must
use the pre-October research folds, with another fresh period reserved.
Publication/vintage and revision bias still limit every retrospective run.
