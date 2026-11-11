"""Run: python -m unittest -v. No market data or network needed."""
import unittest
import numpy as np
import pandas as pd
from features import SeriesView, build_origin_frame, FEATURES, _day_ahead_available
from sample_data import make_sample_data, SPOT_PUBLISH_OFFSET
from model import pinball_loss

class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.df=make_sample_data()
    def test_full_horizon(self):
        f=build_origin_frame(SeriesView(self.df),pd.Timestamp('2026-11-01T21:00:00Z'))
        self.assertEqual(len(f),40)
        self.assertEqual(f.delivery_start_utc.iloc[-1],pd.Timestamp('2026-11-02T06:45:00Z'))
    def test_future_actuals_do_not_change_features(self):
        origin=pd.Timestamp('2026-10-05T00:00:00Z')
        a=build_origin_frame(SeriesView(self.df),origin)[FEATURES]
        altered=self.df.copy()
        mask=altered.timestamp_utc > origin-pd.Timedelta(hours=1)
        for col in ['imbalance_price_eur_mwh','mfrr_direction','spike_high','spike_low']:
            altered.loc[mask,col]=99999
        b=build_origin_frame(SeriesView(altered),origin)[FEATURES]
        pd.testing.assert_frame_equal(a,b)
    def test_publication_boundary(self):
        d=pd.DatetimeIndex(['2026-10-06T00:00:00Z'])
        self.assertFalse(_day_ahead_available(d,pd.Timestamp('2026-10-05T10:00:00Z'),SPOT_PUBLISH_OFFSET)[0])
        self.assertTrue(_day_ahead_available(d,pd.Timestamp('2026-10-05T14:00:00Z'),SPOT_PUBLISH_OFFSET)[0])
    def test_pinball(self):
        self.assertAlmostEqual(pinball_loss(np.array([10]),np.array([0]),.9),9)
        self.assertAlmostEqual(pinball_loss(np.array([0]),np.array([10]),.9),1)

if __name__=='__main__': unittest.main()
