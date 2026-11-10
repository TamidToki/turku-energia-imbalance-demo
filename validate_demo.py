"""Local format check derived from the briefing, NOT the official validator."""
import numpy as np
import pandas as pd

COLUMNS = ['forecast_origin_utc','delivery_start_utc','step','p10','p50','p90',
           'p_up','p_down','p_none','p_spike_high','p_spike_low','data_cutoff_utc','top_drivers']

def validate(df):
    assert list(df.columns) == COLUMNS, 'Column order differs'
    assert len(df) == 673 * 40, 'Expected 26,920 rows'
    assert not df.isna().any().any(), 'Missing values'
    origin = pd.to_datetime(df.forecast_origin_utc, utc=True)
    delivery = pd.to_datetime(df.delivery_start_utc, utc=True)
    expected = pd.date_range('2026-10-04T21:00:00Z','2026-11-01T21:00:00Z',freq='h')
    assert set(origin) == set(expected), 'Origin grid differs'
    assert ((delivery-origin) == pd.to_timedelta((df.step-1)*15, unit='min')).all()
    assert (df.groupby('forecast_origin_utc').step.apply(list).apply(lambda x:x==list(range(1,41)))).all()
    numeric = df[['p10','p50','p90','p_up','p_down','p_none','p_spike_high','p_spike_low']]
    assert np.isfinite(numeric.to_numpy()).all()
    assert ((df.p10<=df.p50)&(df.p50<=df.p90)).all()
    assert numeric.iloc[:,3:].ge(0).all().all() and numeric.iloc[:,3:].le(1).all().all()
    assert np.allclose(df[['p_up','p_down','p_none']].sum(axis=1),1,atol=1e-6)
    assert (df.p_spike_high + df.p_spike_low <= 1.000001).all()

if __name__ == '__main__':
    validate(pd.read_csv('results/forecast_backtest.csv'))
    print('Local demo format check passed (not the official validator).')
