"""Daily OHLCV provider. Mirrors src/qlab_rl/data/provider.py. No network."""

import json
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from qlab_rl.data.provider import (
    Bar,
    YahooCsvProvider,
    _epoch,
    fetch_yahoo,
    read_csv,
    write_csv,
)

from tests.helpers import make_ramp_bars


class TestBar(unittest.TestCase):
    def test_slots_and_repr(self):
        b = Bar("2020-01-01", 1.0, 2.0, 0.5, 1.5, 100.0)
        self.assertEqual(b.close, 1.5)
        self.assertIn("2020-01-01", repr(b))
        self.assertEqual(set(Bar.__slots__), {"date", "open", "high", "low", "close", "volume"})


class TestEpoch(unittest.TestCase):
    def test_epoch_known_date(self):
        self.assertEqual(_epoch("1970-01-01"), 0)
        self.assertGreater(_epoch("2020-01-02"), _epoch("2020-01-01"))


class TestCsvRoundTrip(unittest.TestCase):
    def test_write_then_read_preserves_bars(self):
        import tempfile
        bars = make_ramp_bars(5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "SPY.csv"
            write_csv(bars, path)
            self.assertTrue(path.exists())
            back = read_csv(path)
            self.assertEqual(len(back), len(bars))
            for a, b in zip(bars, back):
                self.assertEqual(a.date, b.date)
                self.assertAlmostEqual(a.close, b.close)


def stub_urlopen(payload):
    class Resp:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return json.dumps(payload).encode()
    def fake(req, timeout=None):
        fake.last_url = req.full_url
        return Resp()
    return fake


class TestFetchYahoo(unittest.TestCase):
    def test_parses_quotes_and_skips_none_close(self):
        payload = {"chart": {"result": [{
            "timestamp": [1577836800, 1577923200, 1578009600],
            "indicators": {"quote": [{
                "open": [100.0, 101.0, None],
                "high": [101.0, 102.0, None],
                "low": [99.0, 100.0, None],
                "close": [100.5, None, 103.0],
                "volume": [1000, 2000, None],
            }]}}], "error": None}}
        orig = urllib.request.urlopen
        urllib.request.urlopen = stub_urlopen(payload)
        try:
            bars = fetch_yahoo("SPY", "2020-01-01", "2020-01-04")
        finally:
            urllib.request.urlopen = orig
        self.assertEqual(len(bars), 2)  # None close skipped
        self.assertEqual(bars[0].close, 100.5)
        self.assertEqual(bars[1].close, 103.0)
        self.assertEqual(bars[1].open, 103.0)  # missing OHLC falls back to close
        self.assertEqual(bars[1].volume, 0.0)

    def test_yahoo_error_raises(self):
        payload = {"chart": {"error": {"code": "Not Found"}, "result": None}}
        orig = urllib.request.urlopen
        urllib.request.urlopen = stub_urlopen(payload)
        try:
            with self.assertRaises(RuntimeError):
                fetch_yahoo("BAD", "2020-01-01", "2020-01-02")
        finally:
            urllib.request.urlopen = orig

    def test_empty_result_raises(self):
        payload = {"chart": {"result": [], "error": None}}
        orig = urllib.request.urlopen
        urllib.request.urlopen = stub_urlopen(payload)
        try:
            with self.assertRaises(RuntimeError):
                fetch_yahoo("SPY", "2020-01-01", "2020-01-02")
        finally:
            urllib.request.urlopen = orig


class TestProviderCache(unittest.TestCase):
    def test_cache_hit_avoids_network(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            bars = make_ramp_bars(3)
            write_csv(bars, cache / "SPY.csv")
            p = YahooCsvProvider("SPY", "2020-01-01", None, cache)
            def boom(req, timeout=None):
                raise AssertionError("network should not be hit on cache hit")
            orig = urllib.request.urlopen
            urllib.request.urlopen = boom
            try:
                back = p.load(refresh=False)
            finally:
                urllib.request.urlopen = orig
            self.assertEqual(len(back), 3)

    def test_refresh_downloads_and_writes_cache(self):
        import tempfile
        payload = {"chart": {"result": [{
            "timestamp": [1577836800],
            "indicators": {"quote": [{
                "open": [100.0], "high": [101.0], "low": [99.0],
                "close": [100.5], "volume": [1000],
            }]}}], "error": None}}
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            p = YahooCsvProvider("SPY", "2020-01-01", "2020-01-02", cache)
            orig = urllib.request.urlopen
            urllib.request.urlopen = stub_urlopen(payload)
            try:
                bars = p.load(refresh=True)
            finally:
                urllib.request.urlopen = orig
            self.assertEqual(len(bars), 1)
            self.assertTrue((cache / "SPY.csv").exists())


if __name__ == "__main__":
    unittest.main()
