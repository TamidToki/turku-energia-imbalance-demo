# Astryx dashboard (UI demo)

A web front-end for the imbalance price forecast demo, built with
[Astryx](https://github.com/meta/astryx) (Meta's open-source design system,
MIT licence, React 19). It lives on the `astryx-dashboard` branch only; the
original Streamlit app on `main` is untouched and remains the reference
implementation.

It re-presents the same backtest results as `../app.py` (same pipeline, same
metrics, same alert logic at the 15% spike-probability demo threshold). It
changes nothing about the model or the data. This is a demo, not a production
tool.

## Run

```bash
cd astryx-dashboard
npm install
npm run dev        # http://localhost:5173
```

`npm run build` + `npm run preview` serves the production build. The build is
static (base `./`), so `dist/` can be hosted on any static host.

## Data

The app reads `public/data/demo-data.json`, committed so the demo runs without
Python. Regenerate it after a fresh backtest:

```bash
# from the repo root
python backtest.py                                # synthetic sample pipeline
python astryx-dashboard/scripts/export_data.py    # writes the JSON snapshot
```

If the real-data results (`results/real_v4/`) exist locally, the exporter uses
them instead and the app says so on its banner. Real market data stays
git-ignored as before; only aggregate numbers and the sampled origins land in
the JSON. By default (and in the committed snapshot) the data is the synthetic
sample data, and the app labels it as such.

## Scope and honesty notes

- The committed snapshot is **synthetic sample data** - it validates the
  pipeline and the UI, not real forecasting skill.
- 32 of the 673 test origins are sampled into the snapshot to keep it small.
- Charts use recharts; everything else is Astryx components and the neutral
  theme.
