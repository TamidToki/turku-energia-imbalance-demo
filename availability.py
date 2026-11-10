"""Publication-delay assumptions behind the data availability rule.

The binding rule in the hackathon starter package was the per-row column
``first_usable_origin_utc``. The public Fingrid API does not expose
publication timestamps or archived forecast vintages, so this project
applies the conservative constants below instead: a row is used at a
forecast origin only when it is very likely to have been published by then.

ACTUALS_DELAY (90 min)
    Settlement series (imbalance price, mFRR regulation prices, mFRR
    activation volumes). The briefing's example: values for quarter-hour
    09:00-09:15 are published 60 minutes after the quarter ends (10:15),
    and the first full-hour origin that may use them is 11:00. A quarter
    that started at least 90 minutes before the origin was published at
    least 15 minutes before it, so this satisfies the rule with margin.

SPOT_PUBLISH_OFFSET (13 h)
    Nord Pool day-ahead spot for delivery day D is published about 14:00
    Finnish time on D-1 (briefing example: 2026-05-31T11:00:00Z for
    delivery on 1 June, summer time). Modelled as delivery-day midnight
    UTC minus 13 h. Only relevant when a lawfully obtained spot series is
    supplied (spot is not open data).

FORECAST_PUBLISH_OFFSET (8 h)
    Fingrid's once-a-day forecasts (datasets 165, 246, 247) are published
    once per day; the API does not say when. Conservatively assumed usable
    from D-1 16:00 UTC for delivery day D. If the true publication time is
    earlier, some usable hours are wasted, but nothing is used before
    publication.

Note on vintages: the 15-minute-updated forecast series (Fingrid datasets
166, 245) are NOT used. The API returns only the latest value per
quarter-hour, not the value that was current at a past forecast origin, so
backtesting with them would leak future information. Archived vintages
would have to come from Turku Energia or Fingrid directly.
"""

from __future__ import annotations

import pandas as pd

ACTUALS_DELAY = pd.Timedelta(minutes=90)
SPOT_PUBLISH_OFFSET = pd.Timedelta(hours=13)
FORECAST_PUBLISH_OFFSET = pd.Timedelta(hours=8)
