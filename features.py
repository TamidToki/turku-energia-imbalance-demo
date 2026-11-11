"""Feature engineering and the data availability rule.

Every feature for a forecast origin uses only information that would have
been available at that moment (the challenge's "back to the future" rule):

  - actuals (imbalance price, regulation prices, activations):
    usable ACTUALS_DELAY after the fact (availability.py - 90 min,
    a conservative reading of the briefing's publication example)
  - spot price and day-ahead forecasts (wind/solar/consumption):
    assumed published the day before the Finnish delivery day (14:00 / 19:00 UTC);
    if the origin is earlier than that, the feature is left as NaN
    (the model handles NaN natively)

Real hackathon data used the binding per-row column first_usable_origin_utc;
this simplified version keeps the same idea so the habit carries over.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from availability import (
    ACTUALS_DELAY,
    FORECAST_PUBLISH_OFFSET,
    SPOT_PUBLISH_OFFSET,
)

FEATURES = [
    "step",
    "hour",
    "dow",
    "month",
    "is_weekend",
    "spot",
    "spot_ramp_3h",
    "wind_f",
    "solar_f",
    "cons_f",
    "wind_ramp_3h",
    "cons_ramp_3h",
    "last_gap",
    "gap_mean_24h",
    "residual_load",
    "last_imb",
    "imb_mean_24h",
    "imb_std_24h",
    "dir_up_share_4h",
    "dir_down_share_4h",
    "spike_hi_24h",
    "spike_lo_24h",
]

# Human-readable labels, used for the optional top_drivers column and the
# alert text in the dashboard.
FEATURE_LABELS = {
    "spot": "spot price level",
    "spot_ramp_3h": "spot price trend",
    "wind_f": "wind power forecast",
    "wind_ramp_3h": "wind power forecast change",
    "solar_f": "solar power forecast",
    "cons_f": "consumption forecast",
    "cons_ramp_3h": "consumption forecast change",
    "last_imb": "latest imbalance price",
    "imb_mean_24h": "recent imbalance price level",
    "imb_std_24h": "recent imbalance price volatility",
    "dir_up_share_4h": "recent up-regulation share",
    "dir_down_share_4h": "recent down-regulation share",
    "spike_hi_24h": "recent high-price spikes",
    "spike_lo_24h": "recent low-price spikes",
    "hour": "time of day",
    "dow": "day of week",
    "month": "month",
    "is_weekend": "weekend",
    "step": "forecast step",
}


class SeriesView:
    """Fast position-based access to the wide dataset."""

    def __init__(self, df: pd.DataFrame):
        self.idx = pd.DatetimeIndex(df["timestamp_utc"])
        self.t0 = self.idx[0]
        self.cols = {c: df[c].to_numpy(dtype=float) for c in df.columns if c != "timestamp_utc"}

    def pos(self, ts: pd.Timestamp) -> int:
        return int((ts - self.t0).total_seconds() // 900)

    def ts(self, pos: int) -> pd.Timestamp:
        return self.t0 + pd.Timedelta(minutes=15 * pos)

    def __len__(self) -> int:
        return len(self.idx)


def _day_ahead_available(delivery_ts: pd.DatetimeIndex, origin: pd.Timestamp, offset: pd.Timedelta) -> np.ndarray:
    """True where a day-ahead series for the delivery moment is published by the origin."""
    published_at = delivery_ts.tz_convert("Europe/Helsinki").tz_localize(None).normalize().tz_localize("UTC") - offset
    # UTC date-based formula was early for late-evening next-local-day delivery.
    # Date is the Finnish calendar date, represented at midnight UTC.
    return np.asarray(published_at <= origin)


def build_origin_frame(view: SeriesView, origin: pd.Timestamp, steps: int = 40) -> pd.DataFrame:
    """Features and targets for one forecast origin (40 quarter-hour steps).

    Step 1 starts at the forecast origin, as in the official briefing.
    """
    p = view.pos(origin)
    k = p - int(ACTUALS_DELAY.total_seconds() // 900)  # latest usable actuals position
    step_arr = np.arange(1, steps + 1)
    d = p + step_arr - 1  # delivery positions
    delivery_ts = view.idx[d]

    # Latest known actuals (scalars per origin)
    imb = view.cols["imbalance_price_eur_mwh"]
    direction = view.cols["mfrr_direction"]
    s_hi = view.cols["spike_high"]
    s_lo = view.cols["spike_low"]
    win_24h = slice(max(k - 95, 0), k + 1)
    win_4h = slice(max(k - 15, 0), k + 1)
    last_imb = imb[k]
    imb_mean_24h = np.nanmean(imb[win_24h])
    imb_std_24h = np.nanstd(imb[win_24h])
    dir_up_share_4h = np.mean(direction[win_4h] == 1)
    dir_down_share_4h = np.mean(direction[win_4h] == -1)
    spike_hi_24h = np.nansum(s_hi[win_24h])
    spike_lo_24h = np.nansum(s_lo[win_24h])

    # Day-ahead series, NaN where not yet published at the origin
    spot = view.cols["spot_fi_eur_mwh"][d].copy()
    wind_f = view.cols["wind_forecast_mw"][d].copy()
    solar_f = view.cols["solar_forecast_mw"][d].copy()
    cons_f = view.cols["consumption_forecast_mw"][d].copy()
    spot_ok = _day_ahead_available(delivery_ts, origin, SPOT_PUBLISH_OFFSET)
    fc_ok = _day_ahead_available(delivery_ts, origin, FORECAST_PUBLISH_OFFSET)
    spot[~spot_ok] = np.nan
    for arr in (wind_f, solar_f, cons_f):
        arr[~fc_ok] = np.nan

    def ramp(arr_full: np.ndarray, positions: np.ndarray, back: int = 12) -> np.ndarray:
        now = arr_full[positions].copy()
        past = arr_full[positions - back]
        return now - past

    spot_ramp = ramp(view.cols["spot_fi_eur_mwh"], d)
    spot_ramp[~spot_ok] = np.nan
    wind_ramp = ramp(view.cols["wind_forecast_mw"], d)
    cons_ramp = ramp(view.cols["consumption_forecast_mw"], d)
    wind_ramp[~fc_ok] = np.nan
    cons_ramp[~fc_ok] = np.nan

    hour = delivery_ts.hour + delivery_ts.minute / 60.0
    out = pd.DataFrame(
        {
            "step": step_arr,
            "hour": hour,
            "dow": delivery_ts.dayofweek,
            "month": delivery_ts.month,
            "is_weekend": (delivery_ts.dayofweek >= 5).astype(int),
            "spot": spot,
            "spot_ramp_3h": spot_ramp,
            "wind_f": wind_f,
            "solar_f": solar_f,
            "cons_f": cons_f,
            "wind_ramp_3h": wind_ramp,
            "cons_ramp_3h": cons_ramp,
            "last_gap": last_imb - view.cols["spot_fi_eur_mwh"][k],
            "gap_mean_24h": np.nanmean(imb[win_24h] - view.cols["spot_fi_eur_mwh"][win_24h]),
            "residual_load": cons_f-wind_f-solar_f,
            "last_imb": last_imb,
            "imb_mean_24h": imb_mean_24h,
            "imb_std_24h": imb_std_24h,
            "dir_up_share_4h": dir_up_share_4h,
            "dir_down_share_4h": dir_down_share_4h,
            "spike_hi_24h": spike_hi_24h,
            "spike_lo_24h": spike_lo_24h,
        }
    )
    out["forecast_origin_utc"] = origin
    out["delivery_start_utc"] = delivery_ts
    out["y_price"] = view.cols["imbalance_price_eur_mwh"][d]
    out["y_direction"] = view.cols["mfrr_direction"][d].astype(int)
    out["y_spike_high"] = view.cols["spike_high"][d].astype(int)
    out["y_spike_low"] = view.cols["spike_low"][d].astype(int)
    out["actual_spot"] = view.cols["spot_fi_eur_mwh"][d]
    return out


def build_matrix(view: SeriesView, origins: list[pd.Timestamp]) -> pd.DataFrame:
    return pd.concat([build_origin_frame(view, o) for o in origins], ignore_index=True)
