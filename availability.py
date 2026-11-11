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

SPOT_PUBLISH_OFFSET (10 h)
    Use the Finnish LOCAL delivery date at midnight, represented in UTC,
    minus 10 hours: D-1 14:00 UTC. This is a conservative operational
    assumption, not an observed publication timestamp. It fixes the v3
    UTC-date boundary error for late-evening next-local-day delivery.

FORECAST_PUBLISH_OFFSET (5 h)
    Same local-date convention, minus 5 hours: D-1 19:00 UTC.
    The public API does not provide publication/vintage metadata. These
    once-daily series may still contain revisions. Excluding all of them
    is evaluated as a sensitivity check; delays alone cannot prove no leak.

Note on vintages: the 15-minute-updated forecast series (Fingrid datasets
166, 245) are NOT used. The API returns only the latest value per
quarter-hour, not the value that was current at a past forecast origin, so
backtesting with them would leak future information. Archived vintages
would have to come from Turku Energia or Fingrid directly.
"""

from __future__ import annotations

import pandas as pd

ACTUALS_DELAY = pd.Timedelta(minutes=90)
SPOT_PUBLISH_OFFSET = pd.Timedelta(hours=10)
FORECAST_PUBLISH_OFFSET = pd.Timedelta(hours=5)
