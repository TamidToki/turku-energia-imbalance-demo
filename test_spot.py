"""Spot parser, availability and v4 output checks. No network or secrets."""
import unittest
import pandas as pd
import numpy as np
from entsoe_data import parse_prices
from features import SeriesView,build_origin_frame,FEATURES,_day_ahead_available
from availability import SPOT_PUBLISH_OFFSET
from fingrid_data import load_real_inputs

class SpotTests(unittest.TestCase):
    def test_sparse_curve_expansion(self):
        xml='''<Publication_MarketDocument><TimeSeries><Period><timeInterval><start>2026-10-01T00:00Z</start><end>2026-10-01T01:00Z</end></timeInterval><resolution>PT15M</resolution><Point><position>1</position><price.amount>10</price.amount></Point><Point><position>3</position><price.amount>20</price.amount></Point></Period></TimeSeries></Publication_MarketDocument>'''
        p=parse_prices(xml);self.assertEqual(p.value.tolist(),[10,10,20,20])
    def test_local_day_boundary_spot_unavailable(self):
        # 22:00 UTC is already next Finnish day. Prior UTC-date rule leaked this.
        delivery=pd.DatetimeIndex(['2026-07-01T22:00Z'])
        self.assertFalse(_day_ahead_available(delivery,pd.Timestamp('2026-07-01T10:00Z'),SPOT_PUBLISH_OFFSET)[0])
        self.assertTrue(_day_ahead_available(delivery,pd.Timestamp('2026-07-01T16:00Z'),SPOT_PUBLISH_OFFSET)[0])
    def test_dst_repeated_hour_and_winter_midnight(self):
        # Both repeated local 03:00 hours share the same local delivery date.
        d=pd.DatetimeIndex(['2026-10-25T00:00Z','2026-10-25T01:00Z'])
        self.assertEqual(d.tz_convert('Europe/Helsinki').hour.tolist(),[3,3])
        self.assertFalse(_day_ahead_available(d,pd.Timestamp('2026-10-24T13:59Z'),SPOT_PUBLISH_OFFSET).any())
        self.assertTrue(_day_ahead_available(d,pd.Timestamp('2026-10-24T14:00Z'),SPOT_PUBLISH_OFFSET).all())
        d=pd.DatetimeIndex(['2026-11-01T22:00Z']) # 2 Nov midnight in Finland
        self.assertFalse(_day_ahead_available(d,pd.Timestamp('2026-11-01T13:59Z'),SPOT_PUBLISH_OFFSET)[0])
        self.assertTrue(_day_ahead_available(d,pd.Timestamp('2026-11-01T14:00Z'),SPOT_PUBLISH_OFFSET)[0])
    def test_gap_anchor_never_reads_actual_spot(self):
        from spot_model import SpotForecaster
        a=pd.DataFrame({'spot':[10.,np.nan],'last_imb':[50.,50.],'last_gap':[20.,20.],'actual_spot':[9999.,9999.]})
        np.testing.assert_allclose(SpotForecaster.anchor(a),[10,30])
    def test_future_perturbation_real(self):
        try:df=load_real_inputs()
        except FileNotFoundError:self.skipTest('Real input not downloaded')
        origin=pd.Timestamp('2026-09-10T10:00Z')
        a=build_origin_frame(SeriesView(df),origin)[FEATURES]
        b=df.copy();future=b.timestamp_utc>origin-pd.Timedelta(minutes=90)
        for col in ['imbalance_price_eur_mwh','mfrr_direction','spike_high','spike_low']:
            b.loc[future,col]=999999
        pd.testing.assert_frame_equal(a,build_origin_frame(SeriesView(b),origin)[FEATURES])
        # Unpublished next-local-day spot/forecasts cannot change feature values.
        origin=pd.Timestamp('2026-09-10T13:00Z')
        a=build_origin_frame(SeriesView(df),origin)[FEATURES]
        b=df.copy();mask=b.timestamp_utc>=pd.Timestamp('2026-09-10T21:00Z')
        for col in ['spot_fi_eur_mwh','wind_forecast_mw','solar_forecast_mw','consumption_forecast_mw']:b.loc[mask,col]=999999
        pd.testing.assert_frame_equal(a,build_origin_frame(SeriesView(b),origin)[FEATURES])
    def test_v4_predictions(self):
        from pathlib import Path
        p=Path('results/real_v4/forecast_backtest_real.csv')
        if not p.exists():self.skipTest('Run spot_experiment.py first')
        x=pd.read_csv(p);self.assertEqual(len(x),673*40);self.assertEqual(x.shape[1],13)
        self.assertFalse(x.isna().any().any());self.assertTrue((x.p10<=x.p50).all());self.assertTrue((x.p50<=x.p90).all())
        self.assertTrue(np.allclose(x[['p_up','p_down','p_none']].sum(axis=1),1))
        self.assertTrue((x.p_spike_high+x.p_spike_low<=1+1e-10).all())

if __name__=='__main__':unittest.main()
