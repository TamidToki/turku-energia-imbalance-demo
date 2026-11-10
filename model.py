"""Forecast models: quantile price forecast, direction and spike probabilities.

Deliberately simple and readable (the audience is an energy expert and a team
that uses coding assistants, not ML specialists):

  - price P10/P50/P90: histogram gradient boosting with quantile loss,
    one model per quantile; step number is a feature, so one model covers
    the whole 10-hour horizon
  - direction (up / down / none) and exceptional prices (> spot +/- 50):
    gradient boosting classifiers, probabilities from predict_proba

The challenge's reference baseline is included: latest known price persists
for all 40 steps, interval from historical persistence errors, direction and
spike probabilities from historical shares.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from features import FEATURES, FEATURE_LABELS

GB_KW = dict(max_iter=100, learning_rate=0.06, early_stopping=False, validation_fraction=0.1, random_state=0)


def _hgb_quantile(q: float) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, **GB_KW)


@dataclass
class ImbalanceForecaster:
    q10: HistGradientBoostingRegressor = field(default_factory=lambda: _hgb_quantile(0.1))
    q50: HistGradientBoostingRegressor = field(default_factory=lambda: _hgb_quantile(0.5))
    q90: HistGradientBoostingRegressor = field(default_factory=lambda: _hgb_quantile(0.9))
    direction_clf: HistGradientBoostingClassifier = field(
        default_factory=lambda: HistGradientBoostingClassifier(**GB_KW)
    )
    spike_hi_clf: HistGradientBoostingClassifier = field(
        default_factory=lambda: HistGradientBoostingClassifier(**GB_KW)
    )
    spike_lo_clf: HistGradientBoostingClassifier = field(
        default_factory=lambda: HistGradientBoostingClassifier(**GB_KW)
    )
    top_drivers_: list[str] = field(default_factory=list)

    def fit(self, train: pd.DataFrame) -> "ImbalanceForecaster":
        X = train[FEATURES]
        self.q10.fit(X, train["y_price"])
        self.q50.fit(X, train["y_price"])
        self.q90.fit(X, train["y_price"])
        self.direction_clf.fit(X, train["y_direction"])
        self.spike_hi_clf.fit(X, train["y_spike_high"])
        self.spike_lo_clf.fit(X, train["y_spike_low"])
        from sklearn.inspection import permutation_importance
        sample = train.iloc[::100]
        importance = permutation_importance(self.q50, sample[FEATURES], sample['y_price'],
                                            n_repeats=1, random_state=0, scoring='neg_mean_absolute_error')
        imp = pd.Series(importance.importances_mean, index=FEATURES).sort_values(ascending=False)
        self.top_drivers_ = [FEATURE_LABELS.get(f, f) for f in imp.index[:3]]
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Predict one origin frame; returns the 13 challenge columns."""
        X = frame[FEATURES]
        a = self.q10.predict(X)
        b = self.q50.predict(X)
        c = self.q90.predict(X)
        p10 = np.minimum(np.minimum(a, b), c)
        p90 = np.maximum(np.maximum(a, b), c)
        p50 = np.clip(b, p10, p90)

        dir_proba = self.direction_clf.predict_proba(X)
        classes = list(self.direction_clf.classes_)
        p_up = dir_proba[:, classes.index(1)] if 1 in classes else np.zeros(len(X))
        p_down = dir_proba[:, classes.index(-1)] if -1 in classes else np.zeros(len(X))
        p_none = dir_proba[:, classes.index(0)] if 0 in classes else np.zeros(len(X))
        s = p_up + p_down + p_none
        p_up, p_down, p_none = p_up / s, p_down / s, p_none / s

        p_hi = np.clip(self.spike_hi_clf.predict_proba(X)[:, 1], 0, 1)
        p_lo = np.clip(self.spike_lo_clf.predict_proba(X)[:, 1], 0, 1)
        over = p_hi + p_lo > 1
        total = p_hi[over] + p_lo[over]
        p_hi[over] /= total
        p_lo[over] /= total

        origin = frame["forecast_origin_utc"].iloc[0]
        drivers = "; ".join(self.top_drivers_)
        return pd.DataFrame(
            {
                "forecast_origin_utc": origin.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "delivery_start_utc": frame["delivery_start_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "step": frame["step"].astype(int),
                "p10": np.round(p10, 2),
                "p50": np.round(p50, 2),
                "p90": np.round(p90, 2),
                "p_up": np.round(p_up, 4),
                "p_down": np.round(p_down, 4),
                "p_none": 1 - np.round(p_up, 4) - np.round(p_down, 4),
                "p_spike_high": np.round(p_hi, 4),
                "p_spike_low": np.round(p_lo, 4),
                "data_cutoff_utc": (origin - pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "top_drivers": drivers,
            }
        )


@dataclass
class PersistenceBaseline:
    """Challenge baseline: latest known price continues; interval and
    probabilities from historical behaviour (fitted on training origins)."""

    q10_err: np.ndarray | None = None
    q90_err: np.ndarray | None = None
    dir_shares: np.ndarray | None = None
    spike_hi_share: float = 0.0
    spike_lo_share: float = 0.0

    def fit(self, train: pd.DataFrame) -> "PersistenceBaseline":
        err = train["y_price"] - train["last_imb"]
        self.q10_err = train.groupby("step").apply(lambda g: np.nanquantile(g["y_price"] - g["last_imb"], 0.1)).to_numpy()
        self.q90_err = train.groupby("step").apply(lambda g: np.nanquantile(g["y_price"] - g["last_imb"], 0.9)).to_numpy()
        counts = train["y_direction"].value_counts(normalize=True)
        self.dir_shares = np.array([counts.get(1, 0.0), counts.get(-1, 0.0), counts.get(0, 0.0)])
        self.spike_hi_share = float(train["y_spike_high"].mean())
        self.spike_lo_share = float(train["y_spike_low"].mean())
        _ = err
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        steps = frame["step"].astype(int).to_numpy()
        p50 = frame["last_imb"].to_numpy()
        origin = frame["forecast_origin_utc"].iloc[0]
        return pd.DataFrame(
            {
                "forecast_origin_utc": origin.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "delivery_start_utc": frame["delivery_start_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "step": steps,
                "p10": np.round(p50 + self.q10_err[steps - 1], 2),
                "p50": np.round(p50, 2),
                "p90": np.round(p50 + self.q90_err[steps - 1], 2),
                "p_up": self.dir_shares[0],
                "p_down": self.dir_shares[1],
                "p_none": self.dir_shares[2],
                "p_spike_high": round(self.spike_hi_share, 4),
                "p_spike_low": round(self.spike_lo_share, 4),
                "data_cutoff_utc": (origin - pd.Timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "top_drivers": "",
            }
        )


def pinball_loss(actual: np.ndarray, pred: np.ndarray, q: float) -> float:
    diff = actual - pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def metric_table(df: pd.DataFrame) -> dict:
    """Challenge metrics for a prediction frame that carries actuals."""
    actual = df["actual_price"].to_numpy()
    out = {
        "MAE_p50_eur_mwh": float(np.mean(np.abs(df["p50"] - actual))),
        "pinball_p10": pinball_loss(actual, df["p10"].to_numpy(), 0.1),
        "pinball_p90": pinball_loss(actual, df["p90"].to_numpy(), 0.9),
        "coverage_p10_p90": float(np.mean((df["p10"] <= actual) & (actual <= df["p90"]))),
        "brier_spike_high": float(np.mean((df["p_spike_high"] - df["actual_spike_high"]) ** 2)),
        "brier_spike_low": float(np.mean((df["p_spike_low"] - df["actual_spike_low"]) ** 2)),
    }
    out["pinball_mean"] = (out["pinball_p10"] + out["pinball_p90"]) / 2
    onehot = np.stack(
        [
            (df["actual_direction"] == 1).astype(float),
            (df["actual_direction"] == -1).astype(float),
            (df["actual_direction"] == 0).astype(float),
        ],
        axis=1,
    )
    probs = df[["p_up", "p_down", "p_none"]].to_numpy()
    out["brier_direction"] = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    return out
