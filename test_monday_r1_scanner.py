"""
Unit Tests for monday_r1_rejection_scanner.py
Validates full compliance with:
- Monday Weekly R1 1st-Hour Rejection & Resistance Confluence Scanner Documentation
- Core Rejection Criteria (Test/Pierce, Close Below, Bearish Structure)
- Multi-Timeframe Resistance Level Engine (9 derived levels)
- Resistance Confluence Clustering (<= 0.75% tolerance)
- Output Record Schema
"""

import unittest
from unittest import mock
import numpy as np
import pandas as pd
import monday_r1_rejection_scanner as mr1


class TestMondayR1Scanner(unittest.TestCase):

    def setUp(self):
        # 240 business days of daily data ending Friday 2026-11-20
        dates = pd.date_range("2026-01-01", periods=240, freq="B")
        np.random.seed(101)
        base = 2500.0 + np.cumsum(np.random.randn(len(dates)) * 8)
        self.df_d = pd.DataFrame({
            "Open": base + np.random.uniform(-4, 4, len(dates)),
            "High": base + np.random.uniform(5, 15, len(dates)),
            "Low": base - np.random.uniform(5, 15, len(dates)),
            "Close": base
        }, index=dates)

        self.target_monday = pd.Timestamp("2026-11-23 09:15:00", tz="Asia/Kolkata")
        self.levels = mr1.get_daily_and_weekly_levels(self.df_d, self.target_monday)
        self.w_r1 = self.levels["Weekly_R1"]

    def test_level_engine_output(self):
        """Verifies derivation of all 9 multi-timeframe resistance levels."""
        expected_keys = [
            "Weekly_R1", "PWH", "PDH", "Weekly_Cam_H3", "Weekly_Cam_H4",
            "Monthly_PP", "Monthly_R1", "EMA_100", "EMA_200"
        ]
        for k in expected_keys:
            self.assertIn(k, self.levels, f"Missing level key: {k}")
            self.assertIsInstance(self.levels[k], float, f"Level {k} should be float")
            self.assertFalse(np.isnan(self.levels[k]), f"Level {k} is NaN")

    def test_valid_rejection_red_candle(self):
        """High tests R1, candle closes red below R1 -> Valid rejection."""
        df_1h = pd.DataFrame({
            "Open": [self.w_r1 * 0.999],
            "High": [self.w_r1 * 1.002],  # Pierces R1
            "Low": [self.w_r1 * 0.985],
            "Close": [self.w_r1 * 0.990]  # Red: Close < Open, Close < R1
        }, index=[self.target_monday])

        res = mr1.scan_monday_r1_from_data("TEST_RED", self.df_d, df_1h, mr1.CONFIG)
        self.assertIsNotNone(res)
        self.assertEqual(res["symbol"], "TEST_RED")
        self.assertEqual(res["entry_trigger_below"], round(self.w_r1 * 0.985, 2))
        self.assertEqual(res["stop_loss_above"], round(self.w_r1 * 1.002, 2))

    def test_valid_rejection_pinbar_green(self):
        """High tests R1, candle closes green below R1 but upper wick >= body -> Valid rejection."""
        # Open: 2500, Close: 2502 (body = 2)
        # High: 2510 (wick = 2510 - 2502 = 8 >= 2)
        r1 = 2505.0
        df_1h = pd.DataFrame({
            "Open": [2500.0],
            "High": [2510.0],
            "Low": [2495.0],
            "Close": [2502.0]
        }, index=[self.target_monday])

        # Patch levels
        mock_levels = self.levels.copy()
        mock_levels["Weekly_R1"] = r1

        with unittest.mock.patch("monday_r1_rejection_scanner.get_daily_and_weekly_levels", return_value=mock_levels):
            res = mr1.scan_monday_r1_from_data("TEST_PINBAR", self.df_d, df_1h, mr1.CONFIG)
            self.assertIsNotNone(res)
            self.assertIn("Upper Rejection Wick", res["pattern_tag"])

    def test_fail_close_above_r1(self):
        """Candle closes above R1 -> Not a bearish rejection."""
        df_1h = pd.DataFrame({
            "Open": [self.w_r1 * 0.999],
            "High": [self.w_r1 * 1.005],
            "Low": [self.w_r1 * 0.995],
            "Close": [self.w_r1 * 1.002]  # Closes above R1
        }, index=[self.target_monday])

        res = mr1.scan_monday_r1_from_data("TEST_FAIL_CLOSE", self.df_d, df_1h, mr1.CONFIG)
        self.assertIsNone(res)

    def test_fail_touch_tolerance(self):
        """High does not reach within 0.2% of R1 -> Fails touch test."""
        df_1h = pd.DataFrame({
            "Open": [self.w_r1 * 0.990],
            "High": [self.w_r1 * 0.995],  # 0.5% below R1 (exceeds 0.2% tolerance)
            "Low": [self.w_r1 * 0.980],
            "Close": [self.w_r1 * 0.985]
        }, index=[self.target_monday])

        res = mr1.scan_monday_r1_from_data("TEST_FAIL_TOUCH", self.df_d, df_1h, mr1.CONFIG)
        self.assertIsNone(res)

    def test_fail_green_small_wick(self):
        """Green candle where upper wick < real body -> Fails rejection structure."""
        # Open: 2490, Close: 2502 (body = 12), High: 2504 (wick = 2)
        r1 = 2503.0
        df_1h = pd.DataFrame({
            "Open": [2490.0],
            "High": [2504.0],
            "Low": [2488.0],
            "Close": [2502.0]
        }, index=[self.target_monday])

        mock_levels = self.levels.copy()
        mock_levels["Weekly_R1"] = r1

        with unittest.mock.patch("monday_r1_rejection_scanner.get_daily_and_weekly_levels", return_value=mock_levels):
            res = mr1.scan_monday_r1_from_data("TEST_FAIL_STRUCTURE", self.df_d, df_1h, mr1.CONFIG)
            self.assertIsNone(res)

    def test_confluence_clustering(self):
        """Checks that levels within 0.75% are clustered and sorted closest first."""
        mock_levels = {
            "Weekly_R1": 1000.0,
            "PWH": 1005.0,        # 0.5% -> Match
            "PDH": 1003.0,        # 0.3% -> Match (closer than PWH)
            "Weekly_Cam_H4": 1015.0,  # 1.5% -> Excluded (> 0.75%)
            "EMA_100": 998.0,     # 0.2% -> Match (closest)
            "EMA_200": 950.0      # 5.0% -> Excluded
        }
        confs = mr1.find_confluences(mock_levels, tol_pct=0.0075)
        self.assertEqual(len(confs), 3)
        # Should be ordered closest first: EMA_100 (0.2%), PDH (0.3%), PWH (0.5%)
        self.assertEqual(confs[0]["level_name"], "EMA_100")
        self.assertEqual(confs[1]["level_name"], "PDH")
        self.assertEqual(confs[2]["level_name"], "PWH")

    def test_schema_compliance(self):
        """Verifies Section 6 output schema compliance."""
        df_1h = pd.DataFrame({
            "Open": [self.w_r1 * 0.999],
            "High": [self.w_r1 * 1.002],
            "Low": [self.w_r1 * 0.985],
            "Close": [self.w_r1 * 0.990]
        }, index=[self.target_monday])

        res = mr1.scan_monday_r1_from_data("SCHEMA_TEST", self.df_d, df_1h, mr1.CONFIG)
        required_fields = [
            "symbol", "monday_date", "weekly_r1", "1h_open", "1h_high", "1h_low", "1h_close",
            "entry_trigger_below", "stop_loss_above", "confluence_count", "confluences"
        ]
        for field in required_fields:
            self.assertIn(field, res, f"Output record missing required field: {field}")


if __name__ == "__main__":
    unittest.main()
