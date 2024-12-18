import torch
import pandas as pd
import os
import unittest
from models.sentiment import *
import shutil

class TestFinBertModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        cls.tokenizer = AutoTokenizer.from_pretrained(
            'yiyanghkust/finbert-tone')
        cls.model = AutoModelForSequenceClassification.from_pretrained(
            'yiyanghkust/finbert-tone')
        cls.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        cls.model.to(cls.device)

        cls.sample_data = pd.DataFrame({
            'ticker': ['TEST'],
            'text': ["The company's performance this quarter was exceptional, exceeding all expectations."]
        })

        cls.sample_text = "This is a test text file for load_data function."
        cls.sample_file_path = 'test_sample.txt'
        with open(cls.sample_file_path, 'w', encoding='utf-8') as f:
            f.write(cls.sample_text)

        cls.sample_dir = 'test_sample_dir'
        os.makedirs(cls.sample_dir, exist_ok=True)
        cls.sample_file_path_in_dir = os.path.join(
            cls.sample_dir, 'test_file_in_dir.txt')
        with open(cls.sample_file_path_in_dir, 'w', encoding='utf-8') as f:
            f.write(cls.sample_text)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.sample_file_path)
        os.remove(cls.sample_file_path_in_dir)
        shutil.rmtree('test_sample_dir')

    def test_load_data_with_dataframe(self):
        df = load_data(self.sample_data)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertIn('ticker', df.columns)
        self.assertIn('text', df.columns)
        self.assertEqual(df.iloc[0]['ticker'], 'TEST')
        self.assertEqual(df.iloc[0]['text'], self.sample_data.iloc[0]['text'])

    def test_load_data_with_file(self):
        df = load_data(self.sample_file_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.iloc[0]['ticker'], 'test_sample')
        self.assertEqual(df.iloc[0]['text'], self.sample_text)

    def test_load_data_with_directory(self):
        df = load_data(self.sample_dir)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.iloc[0]['ticker'], 'test_file_in_dir')
        self.assertEqual(df.iloc[0]['text'], self.sample_text)

    def test_preprocess_text(self):
        raw_text = "This is   a test   text with  extra spaces and [unwanted content]  a URL: http://example.com."
        expected_text = "This is a test text with extra spaces and a URL:"
        processed_text = preprocess_text(raw_text)
        self.assertEqual(processed_text, expected_text.strip())

    def test_split_into_sentences(self):
        text = "This is the first sentence. This is the second sentence! And the third one?"
        sentences = split_into_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertEqual(sentences[0], "This is the first sentence.")
        self.assertEqual(sentences[1], "This is the second sentence!")
        self.assertEqual(sentences[2], "And the third one?")

    def test_finbert_sentiment_intensity_batch(self):
        texts = [
            "The company's profits have increased significantly.",  # Expected: Positive
            "We are facing unexpected losses.",                    # Expected: Negative
        ]
        expected_sentiments = ['positive', 'negative', 'neutral']
        intensities = finbert_sentiment_intensity_batch(
            texts, self.model, self.tokenizer, self.device, batch_size=2
        )
        self.assertEqual(len(intensities), 2)
        for i, intensity in enumerate(intensities):
            self.assertIsInstance(intensity, list)
            self.assertEqual(len(intensity), 3)
            self.assertTrue(all(0 <= prob <= 1 for prob in intensity))
            self.assertAlmostEqual(sum(intensity), 1.0, places=5)

            predicted_sentiment, _, _, _, _ = aggregate_intensity_scores([intensity])
            self.assertEqual(predicted_sentiment, expected_sentiments[i])

    def test_aggregate_intensity_scores(self):
        # Create dummy intensities
        intensities = [
            [0.2, 0.5, 0.3],  # [prob_neutral, prob_positive, prob_negative]
            [0.1, 0.7, 0.2],
            [0.6, 0.3, 0.1]
        ]
        overall_sentiment, avg_intensity, _, _, _ = aggregate_intensity_scores(
            intensities)
        self.assertIsInstance(overall_sentiment, str)
        self.assertIn(overall_sentiment, ['positive', 'negative', 'neutral'])
        self.assertIsInstance(avg_intensity, float)
        self.assertEqual(overall_sentiment, 'positive')

    def test_finbert_handler_with_dataframe(self):
        results_df = finbert_handler(self.sample_data, batch_size=2)
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertIn('ticker', results_df.columns)
        self.assertIn('finbert_sentiment', results_df.columns)
        self.assertIn('sentiment_intensity', results_df.columns)
        self.assertEqual(len(results_df), 1)
        self.assertEqual(results_df.iloc[0]['ticker'], 'TEST')
        # Can't predict the exact sentiment, but we can check the types
        self.assertIsInstance(results_df.iloc[0]['finbert_sentiment'], str)
        self.assertIn(results_df.iloc[0]['finbert_sentiment'], [
                      'positive', 'negative', 'neutral'])
        self.assertIsInstance(results_df.iloc[0]['sentiment_intensity'], float)

    def test_finbert_handler_with_file(self):
        results_df = finbert_handler(self.sample_file_path, batch_size=2)
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertEqual(len(results_df), 1)
        self.assertEqual(results_df.iloc[0]['ticker'], 'test_sample')
        self.assertIn('finbert_sentiment', results_df.columns)
        self.assertIn('sentiment_intensity', results_df.columns)

    def test_finbert_handler_with_directory(self):
        results_df = finbert_handler(self.sample_dir, batch_size=2)
        self.assertIsInstance(results_df, pd.DataFrame)
        self.assertEqual(results_df.iloc[0]['ticker'], 'test_file_in_dir')
        self.assertIn('finbert_sentiment', results_df.columns)
        self.assertIn('sentiment_intensity', results_df.columns)

    def test_validate_ticker(self):
        self.assertTrue(validate_ticker("AAPL"))
        self.assertFalse(validate_ticker("INVALIDTICKER"))

    def test_validate_ticker_with_delay(self):
        self.assertTrue(validate_ticker_with_delay("MSFT"))
        self.assertFalse(validate_ticker_with_delay("FAKETICKER"))

    def test_fetch_price_changes(self):
        call_date = "2023-01-03"
        result = fetch_price_changes("AAPL", call_date)
        self.assertIn('change_1d', result)
        self.assertIn('change_2d', result)
        self.assertIn('change_5d', result)
        self.assertIsInstance(result['change_1d'], (float, type(None)))
        self.assertIsInstance(result['change_2d'], (float, type(None)))
        self.assertIsInstance(result['change_5d'], (float, type(None)))

    def test_find_txt_files(self):
        txt_files = find_txt_files(self.sample_dir)
        self.assertEqual(txt_files[0], self.sample_file_path_in_dir)

    def test_load_txt_data(self):
        data = load_txt_data(self.sample_dir)
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(data.iloc[0]['ticker'], 'test')
        self.assertEqual(data.iloc[0]['text'], self.sample_text)

    def test_process_with_yfinance(self):
        valid_df, invalid_tickers = process_with_yfinance(self.sample_dir, max_files=1)
        self.assertIsInstance(valid_df, pd.DataFrame)
        self.assertIsInstance(invalid_tickers, list)
        self.assertIn('ticker', valid_df.columns)

    def test_compute_decile_analysis(self):
        sample_valid_df = pd.DataFrame({
            'sentiment_intensity': [0.1, 0.2, -0.1, -0.3, 0.5],
            'change_1d': [1.2, -0.5, 0.3, -1.0, 2.0],
            'change_2d': [1.5, -0.3, 0.6, -0.8, 2.5],
            'change_5d': [1.8, 0.0, 0.8, -0.5, 3.0]
        })
        decile_analysis = compute_decile_analysis(sample_valid_df)
        self.assertIsInstance(decile_analysis, pd.DataFrame)
        self.assertIn('sentiment_range', decile_analysis.columns)
        self.assertIn('change_1d', decile_analysis.columns)
        self.assertIn('change_2d', decile_analysis.columns)
        self.assertIn('change_5d', decile_analysis.columns)

    def test_classify_and_explain(self):
        sample_decile_analysis = pd.DataFrame({
            'sentiment_range': [-0.975, -0.925, 0.025, 0.075, 0.125],
            'change_1d': [-1.0, -0.5, 0.0, 0.5, 1.0],
            'change_2d': [-1.5, -1.0, 0.0, 0.8, 1.5],
            'change_5d': [-2.0, -1.5, 0.0, 1.0, 2.0]
        })
        save_decile_analysis(sample_decile_analysis, filename='test_decile_analysis.pkl')
        explanation = classify_and_explain(0.03, file_path='test_decile_analysis.pkl')
        self.assertIsInstance(explanation, str)
        self.assertIn("This stock is classified this way", explanation)
        os.remove('test_decile_analysis.pkl')

if __name__ == '__main__':
    unittest.main()
