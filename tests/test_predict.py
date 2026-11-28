import unittest

import pandas as pd

from src.mp167.predict import predict_holdout


class FakeModel:
    def predict(self, x):
        assert list(x.columns) == ["weight", "distance"]
        return (x["weight"] + x["distance"]).to_numpy()


class PredictTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "record_id": ["008", "003", "019"],
            "date": ["2026-11-02", "2026-11-01", "2026-11-01"],
            "distance": [7, 3, 1],
            "weight": [1, 2, 3],
        }, index=[9, 4, 6])

    def test_output_matches_input_rows(self):
        out = predict_holdout(FakeModel(), self.df, ["weight", "distance"])
        self.assertEqual(out.columns.tolist(), ["record_id", "prediction"])
        self.assertEqual(len(out), len(self.df))
        self.assertEqual(out["record_id"].tolist(), ["008", "003", "019"])
        self.assertEqual(out["prediction"].tolist(), [8, 5, 4])

    def test_missing_prediction_is_rejected(self):
        self.df.loc[9, "weight"] = float("nan")
        with self.assertRaises(ValueError):
            predict_holdout(FakeModel(), self.df, ["weight", "distance"])
