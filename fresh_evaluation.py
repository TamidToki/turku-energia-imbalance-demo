"""Frozen Nov3-10 point-forecast check. See FRESH_EVALUATION_PROTOCOL.md.
Requires FINGRID_KEY_FILE for the first download, never logs that key.
Original v4 data/results remain untouched.
"""
import os,json
from pathlib import Path
import numpy as np
import pandas as pd
from fingrid_data import DATASETS,FingridClient,build_inputs
from features import SeriesView
from spot_model import SpotForecaster
from spot_experiment import matrix,predict

def main():
    raw=Path('data/fresh_raw');raw.mkdir(exist_ok=True)
    client=None
    for spec in DATASETS:
        dest=raw/f'fingrid_{spec.dataset_id}.csv'
        if dest.exists():continue
        if client is None:client=FingridClient(Path(os.environ['FINGRID_KEY_FILE']).read_text().strip())
        fresh=client.dataset_data(spec.dataset_id,'2026-11-03T00:00:00Z','2026-11-11T09:00:00Z')
        old=pd.read_csv(f'data/raw/fingrid_{spec.dataset_id}.csv')
        df=pd.concat([old,fresh]).drop_duplicates('timestamp_utc',keep='last')
        df.to_csv(dest,index=False)
        print('Downloaded dataset',spec.dataset_id,'new rows',len(fresh),flush=True)
    out=Path('results/fresh_november');out.mkdir(parents=True,exist_ok=True)
    wide=build_inputs(raw_dir=raw,spot_path=Path('data/entsoe/spot_fi.csv'),grid_end='2026-11-11T09:00:00Z',out_csv=Path('data/fresh_inputs.csv'),report_md=out/'quality.md',meta_json=out/'data_meta.json')
    view=SeriesView(wide)
    train=matrix(view,'2026-06-03','2026-10-02','2h')
    test=matrix(view,'2026-11-03','2026-11-10T23:00')
    assert len(test)==192*40,'Missing target rows: fail rather than silently score fewer'
    assert test.delivery_start_utc.max()==pd.Timestamp('2026-11-11T08:45Z')
    assert test.y_price.notna().all()
    print('Fitting frozen model',flush=True)
    m=SpotForecaster(family='lightgbm',gap=True).fit(train)
    p=predict(m,test).p50.to_numpy();anchor=SpotForecaster.anchor(test);steps=test.step.to_numpy()
    previous_day=view.cols['spot_fi_eur_mwh'][[view.pos(t)-96 for t in test.delivery_start_utc]]
    proxy=np.where(test.spot.notna().to_numpy(),anchor,previous_day)
    preds={'lgb_gap_75':anchor+.75*(p-anchor),'spot_anchor':anchor,'persistence':test.last_imb.to_numpy(),'decaying_gap_2h':proxy+np.exp(-np.log(2)*(1.5+(steps-1)/4)/2)*test.last_gap.to_numpy()}
    rows=[];daily=[]
    for name,a in preds.items():
        for bucket,mask in [('all',np.ones(len(test),dtype=bool)),('known_spot',test.spot.notna().to_numpy()),('unpublished_spot',test.spot.isna().to_numpy()),('steps_1_8',steps<=8),('steps_9_40',steps>8)]:
            rows.append({'model':name,'bucket':bucket,'rows':int(mask.sum()),'MAE':float(np.mean(abs(a[mask]-test.y_price.to_numpy()[mask])))})
        day=pd.DataFrame({'day':test.forecast_origin_utc.dt.date,'error':abs(a-test.y_price.to_numpy())}).groupby('day').error.mean()
        daily.extend({'model':name,'day':str(d),'MAE':float(v)} for d,v in day.items())
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(daily).to_csv(out/'daily.csv',index=False)
    pd.DataFrame({'origin':test.forecast_origin_utc,'step':test.step,'actual':test.y_price,**preds}).to_csv(out/'predictions.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)
    print('Eight-day retrospective check only. No new model selected from these results.',flush=True)
if __name__=='__main__':main()
