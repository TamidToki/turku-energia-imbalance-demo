"""Spot-aware models. Selection is chronological, never random-split.
No classifier outputs are fed back as price features.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from model import ImbalanceForecaster,GB_KW
from features import FEATURES

@dataclass
class SpotForecaster(ImbalanceForecaster):
    family: str = 'hgb'
    gap: bool = True
    strict: bool = False
    @staticmethod
    def anchor(frame):
        return frame.spot.fillna(frame.last_imb-frame.last_gap).to_numpy()
    def fit(self,train):
        self.feature_names=[c for c in FEATURES if not (self.strict and c in ['wind_f','solar_f','cons_f','wind_ramp_3h','cons_ramp_3h','residual_load'])]
        X=train[self.feature_names]
        y=train.y_price.to_numpy()-(self.anchor(train) if self.gap else 0)
        for q,name in [(0.1,'q10'),(0.5,'q50'),(0.9,'q90')]:
            if self.family=='lightgbm':
                from lightgbm import LGBMRegressor
                m=LGBMRegressor(objective='quantile',alpha=q,n_estimators=160,num_leaves=15,min_child_samples=100,learning_rate=.04,verbosity=-1,n_jobs=2,random_state=0)
            else: m=HistGradientBoostingRegressor(loss='quantile',quantile=q,**GB_KW)
            setattr(self,name,m.fit(X,y))
        self.direction_clf.fit(X,train.y_direction)
        self.spike_hi_clf.fit(X,train.y_spike_high)
        self.spike_lo_clf.fit(X,train.y_spike_low)
        self.top_drivers_=['spot anchor' if self.gap else 'spot feature','recent imbalance','recent balancing direction']
        return self
    def predict(self,frame):
        # Parent uses FEATURES; temporarily reindex the same feature order via local prediction.
        X=frame[self.feature_names]
        anchor=self.anchor(frame) if self.gap else 0
        a=self.q10.predict(X)+anchor;b=self.q50.predict(X)+anchor;c=self.q90.predict(X)+anchor
        d=self.direction_clf.predict_proba(X);classes=list(self.direction_clf.classes_)
        out=pd.DataFrame({'forecast_origin_utc':frame.forecast_origin_utc.dt.strftime('%Y-%m-%dT%H:%M:%SZ'),'delivery_start_utc':frame.delivery_start_utc.dt.strftime('%Y-%m-%dT%H:%M:%SZ'),'step':frame.step,'p10':np.minimum(np.minimum(a,b),c),'p50':b,'p90':np.maximum(np.maximum(a,b),c)})
        for label,col in [(1,'p_up'),(-1,'p_down'),(0,'p_none')]:out[col]=d[:,classes.index(label)] if label in classes else 0
        for clf,col in [(self.spike_hi_clf,'p_spike_high'),(self.spike_lo_clf,'p_spike_low')]:
            probs=clf.predict_proba(X);cl=list(clf.classes_);out[col]=probs[:,cl.index(1)] if 1 in cl else 0
        total=out.p_spike_high+out.p_spike_low;over=total>1
        out.loc[over,['p_spike_high','p_spike_low']]=out.loc[over,['p_spike_high','p_spike_low']].div(total[over],axis=0)
        out['data_cutoff_utc']=(frame.forecast_origin_utc-pd.Timedelta(minutes=90)).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        out['top_drivers']='spot and recent balancing history (not a causal explanation)'
        return out.reset_index(drop=True)
