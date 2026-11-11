# Frozen new-window evaluation protocol

Frozen 11 November 2026, before downloading or inspecting Fingrid outcomes
for 3-10 November. This is a new RETROSPECTIVE check, not prospective or
production validation. Historical revision/publication limits still apply.

## Model selection already completed

Three expanding-window folds, all before the October benchmark:

| Fold | Train origins end (00 UTC) | Selection origins |
| --- | --- | --- |
| July | 14 July | 17-31 July |
| August | 14 August | 17-31 August |
| September | 14 September | 17 September-2 October |

Training starts 3 June, every 2 hours. Each validation starts >24 hours
after the final training DELIVERY label. October benchmark outcomes are
physically excluded by the CV program before feature construction.

Mean of fold MAEs: LightGBM gap weighted 75% toward ML / 25% toward spot
anchor = 31.83; spot-only = 33.58; simple 2-hour half-life decaying gap =
32.16; persistence = 42.85 EUR/MWh. ML beats spot-only in each fold, but
loses to the decaying-gap baseline in August. These are SELECTION scores.

## Frozen evaluation

- Train the fixed LightGBM gap specification on 3 June-2 October 00 UTC,
  every 2 hours. Fixed 160 trees, 15 leaves, min-child 100, learning rate .04.
- P50 = spot anchor + .75 * (ML P50 - spot anchor).
- Same v4 available-anchor definition. No tuning using October outcomes.
- Compare four point forecasts: the fixed ML recipe, spot anchor,
  persistence, and 2-hour half-life decaying last gap with the available
  previous-day diurnal proxy on unpublished-spot rows.
- Evaluate origins 3 November 00 UTC-10 November 23 UTC, hourly. Each has
  40 quarter-hour predictions. 192 origins / 7,680 forecast rows. The final
  delivery is 11 November 08:45 UTC. Download targets through 09:00 UTC.
- Primary metric: MAE across all forecast rows. Also show each day's MAE,
  each horizon bucket, and known vs unpublished spot. Do not select a new
  recipe based on this result. Report failures and missing rows explicitly.
- This bounded experiment is point-forecast validation only. Do not infer
  better spike, direction or interval calibration from a lower P50 MAE.
- v4 dashboard and reported October results remain frozen. Do not silently
  replace them with whichever new-window result looks best.

The independent new window has just eight days. Even a positive result
would be limited evidence, not a reliable annual performance estimate.
