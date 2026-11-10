"""Model bake-off on the real validation window (7 Sep - 4 Oct 2026).

Compares quantile regressors (P10/P50/P90) from three libraries plus blends,
all on identical features/splits as real_data.py. Selection is by validation
metrics only; the test period stays untouched here. Also reports conformal
calibration of the P10-P90 band (fitted on validation, so validation coverage
after calibration is ~0.80 by construction - the honest check comes when the
same factor is applied to the untouched test period in the final run).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from features import FEATURES, SeriesView, build_matrix
from fingrid_data import load_real_inputs
from model import GB_KW, pinball_loss
from real_data import (TRAIN_START, TRAIN_ORIGIN_STRIDE_HOURS,
                       VALIDATION_ORIGINS_START, VALIDATION_ORIGINS_END)

# Clean model-selection split: the challenge's validation window (7 Sep - 4 Oct)
# OVERLAPS its training window (ends 3 Oct), so picking a model on it rewards
# memorisation. v3 selects on a truly out-of-time split instead: train ends
# 31 Aug, selection window 7 Sep - 4 Oct. The winner is then retrained on the
# full training window and scored once on the untouched test period.
TRAIN_END = pd.Timestamp("2026-08-31T00:00:00Z")

import lightgbm as lgb
import xgboost as xgb

def hgb(q):
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, **GB_KW)

def lgbm(q):
    return lgb.LGBMRegressor(objective="quantile", alpha=q, n_estimators=400,
                             learning_rate=0.05, num_leaves=63, min_child_samples=40,
                             subsample=0.9, colsample_bytree=0.9, random_state=0,
                             verbosity=-1, n_jobs=-1)

def xgbr(q):
    return xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=q,
                            n_estimators=400, learning_rate=0.05, max_depth=8,
                            subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                            random_state=0, n_jobs=-1)

LIBS = {"hgb": hgb, "lightgbm": lgbm, "xgboost": xgbr}

def fit_quantiles(factory, X, y):
    return {q: factory(q).fit(X, y) for q in (0.1, 0.5, 0.9)}

def predict_band(models, X):
    a = models[0.1].predict(X); b = models[0.5].predict(X); c = models[0.9].predict(X)
    p10 = np.minimum(np.minimum(a, b), c)
    p90 = np.maximum(np.maximum(a, b), c)
    p50 = np.clip(b, p10, p90)
    return p10, p50, p90

def band_metrics(y, p10, p50, p90):
    return {
        "MAE_p50": float(np.mean(np.abs(p50 - y))),
        "pinball_mean": (pinball_loss(y, p10, 0.1) + pinball_loss(y, p90, 0.9)) / 2,
        "coverage": float(np.mean((p10 <= y) & (y <= p90))),
    }

def conformal_q(y, p10, p90, target=0.80):
    s = np.maximum(p10 - y, y - p90)  # nonconformity: >0 means outside band
    n = len(s)
    k = int(np.ceil((n + 1) * target))
    return float(np.sort(s)[min(k, n) - 1])

def main():
    print("Loading real data and building matrices (this reuses real_data.py splits)...")
    view = SeriesView(load_real_inputs())
    train_origins = list(pd.date_range(TRAIN_START, TRAIN_END, freq=f"{TRAIN_ORIGIN_STRIDE_HOURS}h"))
    val_origins = list(pd.date_range(VALIDATION_ORIGINS_START, VALIDATION_ORIGINS_END, freq="1h"))
    train = build_matrix(view, train_origins).dropna(subset=["y_price"])
    val = build_matrix(view, val_origins).dropna(subset=["y_price"])
    print(f"train rows: {len(train)} | validation rows: {len(val)}")

    Xtr, ytr = train[FEATURES], train["y_price"].to_numpy()
    Xv, yv = val[FEATURES], val["y_price"].to_numpy()

    rows, bands = [], {}
    for name, factory in LIBS.items():
        print(f"fitting {name}...")
        models = fit_quantiles(factory, Xtr, ytr)
        p10, p50, p90 = predict_band(models, Xv)
        bands[name] = (p10, p50, p90)
        rows.append({"model": name, **band_metrics(yv, p10, p50, p90)})
        print(" ", rows[-1])

    # 50/50 blends of pairs with different biases
    for a, b in (("lightgbm", "hgb"), ("lightgbm", "xgboost"), ("hgb", "xgboost")):
        p10 = (bands[a][0] + bands[b][0]) / 2
        p50 = (bands[a][1] + bands[b][1]) / 2
        p90 = (bands[a][2] + bands[b][2]) / 2
        rows.append({"model": f"blend_{a}+{b}", **band_metrics(yv, p10, p50, p90)})
        print(" ", rows[-1])

    # conformal calibration of each band on validation
    for row in rows:
        name = row["model"]
        if name in bands:
            p10, p50, p90 = bands[name]
        else:
            a, b = name.replace("blend_", "").split("+")
            p10 = (bands[a][0] + bands[b][0]) / 2; p50 = (bands[a][1] + bands[b][1]) / 2; p90 = (bands[a][2] + bands[b][2]) / 2
        q = conformal_q(yv, p10, p90)
        row["conformal_shift"] = round(q, 2)
        row["coverage_calibrated_val"] = round(float(np.mean((p10 - q <= yv) & (yv <= p90 + q))), 4)

    out = pd.DataFrame(rows).sort_values("MAE_p50")
    pd.set_option("display.width", 160)
    print("\n=== Bake-off on the REAL validation window (chosen by validation only) ===")
    print(out.to_string(index=False))
    out.to_csv("results/real/bakeoff_validation.csv", index=False)
    print("saved results/real/bakeoff_validation.csv")

if __name__ == "__main__":
    main()
