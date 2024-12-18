import unittest
import numpy as np
import os
from models.time_series import (
    prepare_sequence_data,
    time_series_split,
    fetch_stock_data,
    StockPredictor
)

class TestTimeSeriesModel(unittest.TestCase):
    def test_prepare_sequence_data(self):
        data = np.array([
            [i, i+1, i+2] for i in range(10)
        ])

        look_back = 3
        look_forward = 1
        target_idx = 2

        X, y = prepare_sequence_data(data, look_back, look_forward, target_idx)

        self.assertEqual(X.shape, (7, 3, 3))
        self.assertEqual(y.shape, (7,))
        expected_X_first = data[:3]
        expected_y_first = data[3, target_idx]
        np.testing.assert_array_equal(X[0], expected_X_first)
        self.assertEqual(y[0], expected_y_first)

    def test_fetch_stock_data(self):
        ticker = "AAPL"
        start_date = "2023-01-01"
        end_date = "2023-01-10"

        df = fetch_stock_data(ticker, start_date, end_date, interval="1d")
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)

        ticker_inv = "INVALIDTICKER"
        df_inv = fetch_stock_data(ticker_inv, start_date, end_date, interval="1d")
        self.assertTrue(df_inv.empty)

    def test_model_save_load(self):
        predictor = StockPredictor(n_features=5)
        predictor.setup_model(hidden_size=10, num_layers=2, normalize=True)
        predictor.scaler = {"mean": 0, "std": 1}  # Mocked scaler
        predictor.y_scaler = {"mean": 0, "std": 1}  # Mocked scaler

        model_path = "test_model.pt"
        scaler_path = "test_scalers.pkl"

        predictor.save_model(model_path, scaler_path)
        self.assertTrue(os.path.exists(model_path))
        self.assertTrue(os.path.exists(scaler_path))

        predictor.load_model(model_path, scaler_path)
        self.assertIsNotNone(predictor.model)
        self.assertIsNotNone(predictor.scaler)
        self.assertIsNotNone(predictor.y_scaler)

        # Cleanup
        os.remove(model_path)
        os.remove(scaler_path)