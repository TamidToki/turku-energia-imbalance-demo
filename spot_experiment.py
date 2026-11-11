"""Reproducible v4 experiment with disjoint training, selection, calibration, test.
All times UTC. Test was examined in earlier versions, so it is a retrospective
benchmark, not a pristine final holdout or evidence of future profitability.
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from features import SeriesView,build_matrix
from fingrid_data import load_real_inputs
from model import PersistenceBaseline
from real_data import attach_actuals,scored_metrics
from spot_model import SpotForecaster

OUT=Path('results/real_v4')
PROBS=['p_up','p_down','p_none','p_spike_high','p_spike_low']

def matrix(view,start,end,stride='1h'):
    return build_matrix(view,list(pd.date_range(start,end,freq=stride,tz='UTC'))).dropna(subset=['y_price'])

def predict(model,frame):
    p=model.predict(frame)
    # PersistenceBaseline's API was originally single-origin only.
    p['forecast_origin_utc']=frame.forecast_origin_utc.dt.strftime('%Y-%m-%dT%H:%M:%SZ').to_numpy()
    return p

def score(p,f):return scored_metrics(attach_actuals(p,f,True),True)[0]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    view=SeriesView(load_real_inputs())
    train=matrix(view,'2026-06-03','2026-08-30','2h')
    sel=matrix(view,'2026-09-01','2026-09-18T23:00')
    cal=matrix(view,'2026-09-21','2026-10-02T23:00')
    test=matrix(view,'2026-10-04T21:00','2026-11-01T21:00')
    assert train.delivery_start_utc.max()+pd.Timedelta(hours=24)<sel.forecast_origin_utc.min()
    # Purge the final 10h from selection so delivery labels do not overlap calibration.
    sel=sel[sel.delivery_start_utc<cal.forecast_origin_utc.min()].reset_index(drop=True)
    assert cal.delivery_start_utc.max()+pd.Timedelta(hours=24)<test.forecast_origin_utc.min()
    assert sel.delivery_start_utc.max()+pd.Timedelta(hours=24)<cal.forecast_origin_utc.min()
    base=PersistenceBaseline().fit(train)
    bp={k:predict(base,f) for k,f in [('selection',sel),('calibration',cal),('test',test)]}
    # A stronger naive comparison once day-ahead spot is available.
    spot_rows=[]
    err=train.y_price-SpotForecaster.anchor(train)
    for period,f in [('selection',sel),('calibration',cal),('test',test)]:
        p=bp[period].copy();p['p50']=SpotForecaster.anchor(f)
        p['p10']=p.p50+err.quantile(.1);p['p90']=p.p50+err.quantile(.9)
        spot_rows.append({'period':period,'model':'spot_anchor_only',**score(p,f)})
    pd.DataFrame(spot_rows).to_csv(OUT/'spot_anchor_baseline.csv',index=False)
    models={};pred={};rows=[]
    # Freeze all candidates before checking the retrospective test.
    candidates=[('hgb_absolute','hgb',False,False),('hgb_gap','hgb',True,False),('lgb_gap','lightgbm',True,False),('hgb_gap_strict','hgb',True,True)]
    for name,family,gap,strict in candidates:
        print('Fitting',name,flush=True)
        model=SpotForecaster(family=family,gap=gap,strict=strict).fit(train)
        models[name]=model;pred[name]=predict(model,sel)
        rows.append({'model':name,**score(pred[name],sel)})
        print(rows[-1],flush=True)
    # Fixed 50/50 blend, no learned weights.
    blend=pred['hgb_gap'].copy()
    for c in ['p10','p50','p90']+PROBS:blend[c]=(pred['hgb_gap'][c]+pred['hgb_absolute'][c])/2
    pred['blend_hgb']=blend;rows.append({'model':'blend_hgb',**score(blend,sel)})
    table=pd.DataFrame(rows).sort_values('MAE_p50_eur_mwh');table.to_csv(OUT/'selection.csv',index=False)
    winner=table.iloc[0]['model'];print('Selection winner:',winner,flush=True)
    def winning(f):
        if winner!='blend_hgb':return predict(models[winner],f)
        a=predict(models['hgb_gap'],f);b=predict(models['hgb_absolute'],f)
        for c in ['p10','p50','p90']+PROBS:a[c]=(a[c]+b[c])/2
        return a
    # Probability shrinkage selected solely on selection-period Brier, separately by outcome.
    shrink={}
    for cols,key in [(['p_up','p_down','p_none'],'brier_direction'),(['p_spike_high'],'brier_spike_high'),(['p_spike_low'],'brier_spike_low')]:
        best=(float('inf'),0)
        for w in [0,.25,.5,.75,1]:
            p=pred[winner].copy();p[cols]=w*p[cols]+(1-w)*bp['selection'][cols]
            v=score(p,sel)[key]
            if v<best[0]:best=(v,w)
        shrink[key]=best[1]
    cp=winning(cal)
    for cols,key in [(['p_up','p_down','p_none'],'brier_direction'),(['p_spike_high'],'brier_spike_high'),(['p_spike_low'],'brier_spike_low')]:
        w=shrink[key];cp[cols]=w*cp[cols]+(1-w)*bp['calibration'][cols]
    # Heuristic band expansion on calibration rows. Serial dependence prevents exact conformal guarantees.
    err=np.maximum(cp.p10.to_numpy()-cal.y_price.to_numpy(),cal.y_price.to_numpy()-cp.p90.to_numpy())
    shift=max(0,float(np.quantile(err,.8)))
    summary=list(spot_rows)
    # Sensitivity only, not a new test-driven selection.
    strict_p=predict(models['hgb_gap_strict'],test)
    strict_score=score(strict_p,test)
    (OUT/'strict_sensitivity.json').write_text(json.dumps(strict_score,indent=2))
    for period,f in [('selection',sel),('calibration',cal),('test',test)]:
        p=winning(f)
        summary.append({'period':period,'model':winner+'_raw',**score(p,f)})
        for cols,key in [(['p_up','p_down','p_none'],'brier_direction'),(['p_spike_high'],'brier_spike_high'),(['p_spike_low'],'brier_spike_low')]:
            w=shrink[key];p[cols]=w*p[cols]+(1-w)*bp[period][cols]
        p.p10-=shift;p.p90+=shift
        summary.append({'period':period,'model':winner+'_calibrated',**score(p,f)})
        summary.append({'period':period,'model':'persistence',**score(bp[period],f)})
        if period=='test':
            a=attach_actuals(p,f,True);a.to_csv(OUT/'forecast_test_real.csv',index=False)
            p.to_csv(OUT/'forecast_backtest_real.csv',index=False)
            attach_actuals(bp[period],f,True).to_csv(OUT/'baseline_test_real.csv',index=False)
            daily=a.assign(day=pd.to_datetime(a.forecast_origin_utc).dt.date).groupby('day').apply(lambda x:pd.Series({'mae':np.mean(abs(x.p50-x.actual_price)),'coverage':np.mean((x.p10<=x.actual_price)&(x.actual_price<=x.p90))}))
            daily.to_csv(OUT/'daily_test.csv')
            flags=f.spot.notna().to_numpy()
            strata=[]
            for label,mask in [('spot_known',flags),('spot_unpublished',~flags),('step_1_8',f.step.to_numpy()<=8),('step_9_40',f.step.to_numpy()>8),('extreme_actual',(f.y_spike_high.to_numpy()+f.y_spike_low.to_numpy())>0)]:
                strata.append({'bucket':label,'rows':int(mask.sum()),**scored_metrics(a.loc[mask],True)[0]})
            pd.DataFrame(strata).to_csv(OUT/'strata_test.csv',index=False)
    pd.DataFrame(summary).to_csv(OUT/'metrics_summary_real.csv',index=False)
    metadata={'winner':winner,'probability_weights':shrink,'band_expansion':shift,'train_origins':['2026-06-03','2026-08-30'],'selection':['2026-09-01','2026-09-18 (24h+ embargo)'],'calibration':['2026-09-21','2026-10-02'],'test':['2026-10-04T21:00Z','2026-11-01T21:00Z'],'warning':'Retrospective test previously examined; public data has no exact historical publication/vintage records. Not production verified. No finite-sample conformal guarantee for overlapping serial data.'}
    (OUT/'experiment.json').write_text(json.dumps(metadata,indent=2))
    joblib.dump({'models':models,'metadata':metadata},'models/forecaster_real_v4.joblib')
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    print(json.dumps(metadata,indent=2),flush=True)
if __name__=='__main__':main()
