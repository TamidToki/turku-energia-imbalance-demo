"""Pre-October expanding-window research. Does NOT read/score October benchmark.
These are model-selection results, not an untouched final performance estimate.
All source data still have the historical-vintage limits described in README.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from features import SeriesView
from fingrid_data import load_real_inputs
from spot_experiment import matrix,predict
from spot_model import SpotForecaster

FOLDS=[('july','2026-07-14','2026-07-17','2026-07-31T23:00'),
       ('august','2026-08-14','2026-08-17','2026-08-31T23:00'),
       ('september','2026-09-14','2026-09-17','2026-10-02T23:00')]

def main():
    out=Path('results/pre_october_cv');out.mkdir(exist_ok=True,parents=True)
    df=load_real_inputs()
    # Physically discard October benchmark data before feature construction.
    df=df[df.timestamp_utc<pd.Timestamp('2026-10-04',tz='UTC')].copy()
    view=SeriesView(df);rows=[]
    for fold,end,start,stop in FOLDS:
        train=matrix(view,'2026-06-03',end,'2h');val=matrix(view,start,stop)
        assert train.delivery_start_utc.max()+pd.Timedelta(hours=24)<val.forecast_origin_utc.min()
        anchor=SpotForecaster.anchor(val);y=val.y_price.to_numpy();steps=val.step.to_numpy()
        known=val.spot.notna().to_numpy()
        previous_day=view.cols['spot_fi_eur_mwh'][[view.pos(t)-96 for t in val.delivery_start_utc]]
        proxy=np.where(known,anchor,previous_day)
        candidates={'persistence':val.last_imb.to_numpy(),'spot_anchor':anchor,'spot_diurnal_proxy':proxy}
        for half_life in [.5,1,2,4]:
            candidates[f'decaying_gap_{half_life}h']=proxy+np.exp(-np.log(2)*(1.5+(steps-1)/4)/half_life)*val.last_gap.to_numpy()
        for name,family,strict in [('lgb_gap','lightgbm',False),('hgb_gap_strict','hgb',True)]:
            print('fit',fold,name,flush=True)
            m=SpotForecaster(family=family,gap=True,strict=strict).fit(train)
            pred=predict(m,val).p50.to_numpy()
            for w in [0.25,0.5,0.75,1]:candidates[f'{name}_weight_{w}']=anchor+w*(pred-anchor)
            weights=np.select([steps<=8,steps<=20],[.75,.5],default=.25)
            candidates[f'{name}_horizon_damped']=anchor+weights*(pred-anchor)
        for name,p in candidates.items():
            for bucket,mask in [('all',np.ones(len(y),dtype=bool)),('spot_known',known),('spot_unpublished',~known)]:
                rows.append(dict(fold=fold,model=name,bucket=bucket,n=int(mask.sum()),MAE=float(np.mean(abs(p[mask]-y[mask]))),bias=float(np.mean(p[mask]-y[mask]))))
        pd.DataFrame(rows).to_csv(out/'fold_metrics.csv',index=False)
    detail=pd.DataFrame(rows);overall=detail[detail.bucket=='all'].copy()
    ranking=overall.groupby('model').agg(mean_fold_MAE=('MAE','mean'),worst_fold_MAE=('MAE','max')).sort_values('mean_fold_MAE')
    ranking.to_csv(out/'ranking.csv');print(ranking.to_string(),flush=True)
    (out/'protocol.json').write_text(json.dumps({'folds':FOLDS,'warning':'Selection only, folds share expanding training history. No untouched performance estimate. October benchmark physically excluded. Candidate list fixed before runs. v4 remains frozen.'},indent=2))
if __name__=='__main__':main()
