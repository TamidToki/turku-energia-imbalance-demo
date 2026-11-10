"""Tests for the real Fingrid data path. Run: python -m unittest -v.

No network and no API key needed: the HTTP layer is faked with the
documented API response shape.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import fingrid_data as fd
from features import FEATURES, SeriesView, build_origin_frame
from fingrid_data import FingridApiError, FingridClient, build_inputs
from model import ImbalanceForecaster


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class FakeSession:
    """Returns queued responses; records every call for assertions."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "params": params})
        return self.responses.pop(0)


def _rows(dataset_id: int, start: pd.Timestamp, n: int) -> list[dict]:
    return [
        {
            "datasetId": dataset_id,
            "startTime": (start + pd.Timedelta(minutes=15 * i)).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "endTime": (start + pd.Timedelta(minutes=15 * (i + 1))).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "value": float(i),
        }
        for i in range(n)
    ]


class ClientTests(unittest.TestCase):
    def test_pagination_and_header(self):
        # Page 1 full (pageSize rows), page 2 partial -> stop after 2 calls.
        session = FakeSession(
            [
                FakeResponse(200, {"data": _rows(319, pd.Timestamp("2026-06-01", tz="UTC"), 3), "pagination": {}}),
                FakeResponse(200, {"data": _rows(319, pd.Timestamp("2026-06-01T00:45", tz="UTC"), 1), "pagination": {}}),
            ]
        )
        client = FingridClient("secret-key", session=session, sleeper=lambda s: None)
        df = client.dataset_data(319, "2026-06-01T00:00:00Z", "2026-06-02T00:00:00Z", page_size=3)
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(len(df), 4)
        self.assertEqual(session.calls[0]["headers"]["x-api-key"], "secret-key")
        self.assertEqual(session.calls[0]["params"]["page"], 1)
        self.assertEqual(session.calls[1]["params"]["page"], 2)
        self.assertEqual(session.calls[0]["params"]["sortBy"], "startTime")
        self.assertTrue(df["timestamp_utc"].is_monotonic_increasing)

    def test_pagination_total_pages(self):
        session = FakeSession(
            [
                FakeResponse(200, {"data": _rows(319, pd.Timestamp("2026-06-01", tz="UTC"), 2), "pagination": {"totalPages": 1}}),
            ]
        )
        client = FingridClient("k", session=session, sleeper=lambda s: None)
        df = client.dataset_data(319, "2026-06-01T00:00:00Z", "2026-06-02T00:00:00Z", page_size=2)
        self.assertEqual(len(session.calls), 1)  # totalPages=1 stops the loop
        self.assertEqual(len(df), 2)

    def test_rejected_key_is_actionable(self):
        session = FakeSession([FakeResponse(401, text="Unauthorized")])
        client = FingridClient("bad-key", session=session, sleeper=lambda s: None)
        with self.assertRaises(FingridApiError) as ctx:
            client.dataset_data(319, "2026-06-01T00:00:00Z", "2026-06-02T00:00:00Z")
        self.assertIn("API key", str(ctx.exception))
        self.assertEqual(len(session.calls), 1)  # no retry on auth failure

    def test_rate_limit_retried(self):
        ok = {"data": _rows(319, pd.Timestamp("2026-06-01", tz="UTC"), 1), "pagination": {}}
        session = FakeSession([FakeResponse(429, text="slow down"), FakeResponse(200, ok)])
        client = FingridClient("k", session=session, sleeper=lambda s: None)
        df = client.dataset_data(319, "2026-06-01T00:00:00Z", "2026-06-02T00:00:00Z")
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(len(df), 1)


class BuildInputsTests(unittest.TestCase):
    def _write_raw(self, raw_dir: Path, start="2026-10-01T00:00:00Z", days=8):
        """Fabricate raw cache files in the real download format."""
        idx = pd.date_range(start, periods=96 * days, freq="15min", tz="UTC")
        base = pd.DataFrame({"timestamp_utc": idx})
        series = {
            319: 80 + np.sin(np.arange(len(idx)) / 8.0) * 20,          # imbalance price
            244: np.where(np.arange(len(idx)) % 3 == 0, 95.0, np.nan),  # up-reg price
            106: np.where(np.arange(len(idx)) % 3 == 1, 60.0, np.nan),  # down-reg price
            375: np.where(np.arange(len(idx)) % 3 == 0, 120.0, 0.0),    # mFRR up MW
            376: np.where(np.arange(len(idx)) % 3 == 1, 80.0, 0.0),     # mFRR down MW
            246: 2500.0,                                                # wind forecast
            247: 300.0,                                                 # solar forecast
            165: 7000.0,                                                # consumption forecast
        }
        for dataset_id, values in series.items():
            df = base.copy()
            df["value"] = values
            df.to_csv(raw_dir / f"fingrid_{dataset_id}.csv", index=False)
        return idx

    def test_build_schema_direction_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw = tmp / "raw"
            raw.mkdir()
            idx = self._write_raw(raw)
            out_csv, report_md, meta_json = tmp / "real_inputs.csv", tmp / "report.md", tmp / "meta.json"
            wide = build_inputs(
                raw_dir=raw,
                grid_start=idx[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
                grid_end=(idx[-1] + pd.Timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                out_csv=out_csv,
                report_md=report_md,
                meta_json=meta_json,
            )
            self.assertEqual(len(wide), len(idx))
            for col in ("imbalance_price_eur_mwh", "mfrr_direction", "spike_high", "spike_low",
                        "wind_forecast_mw", "solar_forecast_mw", "consumption_forecast_mw"):
                self.assertIn(col, wide.columns)
            # direction follows the fabricated activation volumes (3-quarter cycle)
            self.assertEqual(wide["mfrr_direction"].iloc[0], 1)
            self.assertEqual(wide["mfrr_direction"].iloc[1], -1)
            self.assertEqual(wide["mfrr_direction"].iloc[2], 0)
            # no spot supplied -> labels zeroed and flagged unavailable
            self.assertTrue((wide["spike_high"] == 0).all())
            meta = json.loads(meta_json.read_text())
            self.assertFalse(meta["has_spot"])
            self.assertIn("not open data", report_md.read_text())

    def test_spot_enables_spike_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw = tmp / "raw"
            raw.mkdir()
            idx = self._write_raw(raw)
            spot = pd.DataFrame({"timestamp_utc": idx, "value": 10.0})  # far below imbalance price
            spot_path = tmp / "spot.csv"
            spot.to_csv(spot_path, index=False)
            wide = build_inputs(
                raw_dir=raw,
                spot_path=spot_path,
                grid_start=idx[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
                grid_end=(idx[-1] + pd.Timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                out_csv=tmp / "real_inputs.csv",
                report_md=tmp / "report.md",
                meta_json=tmp / "meta.json",
            )
            self.assertTrue((wide["spike_high"] == 1).all())  # 80+sin*20 > 10+50 everywhere
            self.assertTrue((wide["spike_low"] == 0).all())

    def test_availability_rule_on_real_built_data(self):
        """Altering not-yet-published rows must not change an origin's features."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            raw = tmp / "raw"
            raw.mkdir()
            idx = self._write_raw(raw)
            wide = build_inputs(
                raw_dir=raw,
                grid_start=idx[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
                grid_end=(idx[-1] + pd.Timedelta(minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                out_csv=tmp / "real_inputs.csv",
                report_md=tmp / "report.md",
                meta_json=tmp / "meta.json",
            )
            origin = idx[0] + pd.Timedelta(days=2)
            a = build_origin_frame(SeriesView(wide), origin)[FEATURES]
            altered = wide.copy()
            mask = altered["timestamp_utc"] > origin - pd.Timedelta(minutes=75)
            for col in ("imbalance_price_eur_mwh", "mfrr_direction", "mfrr_activation_up", "mfrr_activation_down"):
                if col in altered:
                    altered.loc[mask, col] = 99999.0
            b = build_origin_frame(SeriesView(altered), origin)[FEATURES]
            pd.testing.assert_frame_equal(a, b)


class NoSpotModelTests(unittest.TestCase):
    def test_spike_outputs_are_nan_not_fake(self):
        from sample_data import make_sample_data

        df = make_sample_data()
        view = SeriesView(df)
        origins = list(pd.date_range("2026-10-05T00:00:00Z", "2026-10-05T08:00:00Z", freq="2h"))
        train = pd.concat([build_origin_frame(view, o) for o in origins], ignore_index=True)
        model = ImbalanceForecaster(spikes_available=False).fit(train)
        pred = model.predict(build_origin_frame(view, pd.Timestamp("2026-10-06T00:00:00Z")))
        self.assertTrue(pred["p_spike_high"].isna().all())
        self.assertTrue(pred["p_spike_low"].isna().all())
        self.assertTrue(np.isfinite(pred["p50"]).all())


if __name__ == "__main__":
    unittest.main()
