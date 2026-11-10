# Showing the draft

This is a local working prototype, not a live market forecast or a hackathon submission.

## A two-minute walkthrough

1. Start with the yellow banner: all data is synthetic. No real accuracy has been measured.
2. Next 10 hours: choose a past forecast origin. The blue line is P50, the shaded band is P10-P90. Forty points cover ten hours. The dashed actual line is shown for replay only, never used as a future input.
3. Show the three balancing-direction probabilities, high/low-price risk and CSV download.
4. What to check now: example threshold-based alerts. Global feature associations are not explanations of an individual alert. Operational wording needs an energy expert's review.
5. Past performance: the model and persistence baseline are scored on generated data. These figures test the code, not business value.
6. Ask Turku Energia about a lawful development dataset, publication timestamps, archived forecast vintages and the decisions they want supported.

## What is not ready

- Live real-data operation: the Fingrid pipeline is built and tested against
  recorded API fixtures; the first live download runs when you add your own
  free API key (see `fingrid_data.py`).
- Exceptional-price probabilities: they need a lawful Nord Pool spot series
  (from Turku Energia or your own licence); without it they stay empty.
- Real-time forecast production, monitoring or automated trading.
- The official package's row-level availability, validator and scoring scripts.
- Archived forecast vintages (the public API serves latest values only).
- Company endorsement, competition score, submitted slides or video.

The private repo has no invited collaborators. Do not make it public or share data without checking permissions and licences.

