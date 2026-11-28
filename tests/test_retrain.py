import unittest

import pandas as pd

from src.mp167.retrain import fill_findings, promote, split_later


class RetrainTests(unittest.TestCase):
    def test_split_orders_by_date_and_keeps_ties_in_input_order(self):
        days = [8, 2, 5, 1, 9, 4, 5, 7, 3, 6]
        df = pd.DataFrame({
            "date": [f"2026-11-{d:02}" for d in days],
            "record_id": list("abcdefghij"),
        }, index=range(20, 30))
        early, check = split_later(df)
        self.assertEqual(early["record_id"].tolist(), list("dbifcgj"))
        self.assertEqual(check["record_id"].tolist(), list("hae"))
        self.assertEqual([*early.index, *check.index], list(range(10)))

    def test_split_rounds_down(self):
        df = pd.DataFrame({"date": pd.date_range("2026-11-01", periods=7)})
        early, check = split_later(df)
        self.assertEqual((len(early), len(check)), (4, 3))

    def test_promotion_threshold(self):
        for mae2, expected in [(89, True), (90, True), (90.1, False), (120, False)]:
            with self.subTest(mae2=mae2):
                self.assertEqual(promote(100, mae2), expected)

    def test_findings_preserve_other_entries_and_comments(self):
        template = (
            "tracking:\n  champion_version: 1\n"
            "monitoring:\n  worst_served_group: 'Tier-2'\n"
            "  mae_v1_on_check: null  # last 30%\n"
            "  mae_v2_on_check: null\n  decision: null  # promote or keep\n"
        )
        for mae2, decision, version in [(90, "promote", 2), (91, "keep", 1)]:
            with self.subTest(mae2=mae2):
                expected = (
                    f"tracking:\n  champion_version: {version}\n"
                    "monitoring:\n  worst_served_group: 'Tier-2'\n"
                    "  mae_v1_on_check: 100  # last 30%\n"
                    f"  mae_v2_on_check: {mae2}\n  decision: {decision}  # promote or keep\n"
                )
                self.assertEqual(fill_findings(template, 100, mae2), expected)
