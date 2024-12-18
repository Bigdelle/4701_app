import unittest
import pandas as pd
import os
from src.data_collection.stockinfo import (
    fetch_stock_data,
    calculate_moving_averages,
    calculate_volatility,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_momentum,
    collect_all_features,
    collect_reddit_data,
    load_sentiment_data,
    collect_sentiment_analysis,
    mock_time_series_prediction,
    extract_tickers_from_directory
)


class TestStockFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sentiment_csv_path = "src/data_collection/test_sentiment_results.csv"
        sentiment_data = """0,text,positive,0.85
1,text,neutral,0.5
2,text,negative,0.2"""
        with open(cls.sentiment_csv_path, "w") as f:
            f.write(sentiment_data)

    def test_fetch_stock_data(self):
        df = fetch_stock_data("AAPL", "2022-01-01", "2022-01-31")
        self.assertFalse(df.empty)
        self.assertIn("Close", df.columns)
        self.assertIn("Volume", df.columns)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.sentiment_csv_path)

    def test_calculate_moving_averages(self):
        data = {"Close": [100, 101, 102, 103, 104, 105, 106]}
        df = pd.DataFrame(data)
        df = calculate_moving_averages(df, periods=[3])
        self.assertIn("MA_3", df.columns)
        self.assertAlmostEqual(df["MA_3"].iloc[-1], (104 + 105 + 106) / 3)

    def test_calculate_volatility(self):
        data = {"Close": [100, 102, 104, 103, 101, 105, 107]}
        df = pd.DataFrame(data)
        df = calculate_volatility(df, window=3)
        self.assertIn("Volatility", df.columns)
        self.assertIsNotNone(df["Volatility"].iloc[-1])

    def test_calculate_rsi(self):
        data = {"Close": [100, 101, 102, 101, 100, 99, 98]}
        df = pd.DataFrame(data)
        df = calculate_rsi(df, period=3)
        self.assertIn("RSI", df.columns)
        self.assertTrue((df["RSI"] >= 0).all() and (df["RSI"] <= 100).all())

    def test_calculate_macd(self):
        data = {"Close": [100, 101, 102, 103, 104, 105, 106]}
        df = pd.DataFrame(data)
        df = calculate_macd(df)
        self.assertIn("MACD", df.columns)
        self.assertIn("Signal_Line", df.columns)

    def test_calculate_bollinger_bands(self):
        data = {"Close": [100, 101, 102, 103, 104, 105, 106]}
        df = pd.DataFrame(data)
        df = calculate_bollinger_bands(df, window=3)
        self.assertIn("BB_Middle", df.columns)
        self.assertIn("BB_Upper", df.columns)
        self.assertIn("BB_Lower", df.columns)

    def test_calculate_momentum(self):
        data = {"Close": [100, 101, 102, 103, 104, 105, 106]}
        df = pd.DataFrame(data)
        df = calculate_momentum(df, period=2)
        self.assertIn("Momentum", df.columns)
        self.assertEqual(df["Momentum"].iloc[-1], 106 - 104)

    def test_load_sentiment_data(self):
        sentiment_dict = load_sentiment_data(self.sentiment_csv_path)
        self.assertIn(0, sentiment_dict)
        self.assertIn("overall_sentiment", sentiment_dict[0])
        self.assertEqual(sentiment_dict[0]["overall_sentiment"], "positive")
        self.assertEqual(sentiment_dict[0]["intensity_score"], 0.85)

    def test_collect_sentiment_analysis_valid(self):
        result = collect_sentiment_analysis(
            0, sentiment_loc=self.sentiment_csv_path)
        self.assertIsNotNone(result)
        self.assertEqual(result["overall_sentiment"], "positive")
        self.assertEqual(result["intensity_score"], 0.85)

    def test_collect_sentiment_analysis_invalid(self):
        result = collect_sentiment_analysis(
            99, sentiment_loc=self.sentiment_csv_path)
        self.assertIsNone(result)

    def test_collect_all_features(self):
        def mock_reddit_func(ticker):
            return {"mentions": 10, "sentiment": "positive"}

        def mock_sentiment_func(ticker):
            return {"overall_sentiment": "neutral", "intensity_score": 0.5}

        def mock_time_series_func(ticker):
            return {"predicted_price": 150}

        df = collect_all_features(
            "AAPL",
            "2022-01-01",
            "2022-01-31",
            mock_reddit_func,
            mock_sentiment_func,
            mock_time_series_func,
        )
        self.assertFalse(df.empty)
        self.assertIn("MA_7", df.columns)
        self.assertIn("RSI", df.columns)

    def test_collect_reddit_data(self):
        data = {"body": ["AAPL is great!", "I think AAPL is overvalued."]}
        reddit_df = pd.DataFrame(data)
        folder_path = "data/reddit_data/AAPL"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        reddit_df.to_csv(
            "data/reddit_data/AAPL/reddit_comments_test.csv", index=False)
        result = collect_reddit_data(
            "AAPL", query="AAPL OR Apple stock", filename="reddit_comments_test.csv")
        self.assertIn("mentions", result)
        self.assertIn("sentiment", result)
        os.remove("data/reddit_data/AAPL/reddit_comments_test.csv")
        os.rmdir("data/reddit_data/AAPL")

    def test_mock_time_series_prediction(self):
        result = mock_time_series_prediction("AAPL")
        self.assertTrue(100 <= result["predicted_price"] <= 500)

    def test_extract_tickers_from_directory(self):
        os.makedirs("data/test_directory/20220101_AAPL", exist_ok=True)
        os.makedirs("data/test_directory/20220101_GOOG", exist_ok=True)
        tickers = extract_tickers_from_directory("data/test_directory")
        self.assertEqual(len(tickers), 2)
        self.assertEqual(tickers[0]["ticker"], "GOOG")
        self.assertEqual(tickers[0]["start_date"], "20220101")
        os.rmdir("data/test_directory/20220101_AAPL")
        os.rmdir("data/test_directory/20220101_GOOG")
        os.rmdir("data/test_directory")


if __name__ == "__main__":
    unittest.main()
