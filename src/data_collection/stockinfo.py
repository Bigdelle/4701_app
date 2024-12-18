from io import StringIO
import yfinance as yf
import pandas as pd
import numpy as np
import re
from collections import Counter
from datetime import datetime, timedelta
import math
import os
import requests

# from models.sentiment import finbert_handler
# from models.time_series import predict_future_price

def clean_transcript(transcript):
    """
    Cleans the earnings call transcript by removing boilerplate text and non-natural language components.
    """
    transcript = re.sub(r"Prepared Remarks:.*?Operator", "", transcript, flags=re.DOTALL)
    transcript = re.sub(r"Operator.*?Thank you", "", transcript, flags=re.DOTALL)
    transcript = transcript.strip()
    return transcript




def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url).text
    sp500_table = pd.read_html(StringIO(html))[0]
    tickers = sp500_table['Symbol'].tolist()
    return tickers

def fetch_stock_data(ticker, start_date, end_date, interval="1d"):
    """
    Fetch stock price data using yfinance.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date, interval=interval)
    return df


def calculate_moving_averages(df, periods=[7, 14, 50]):
    """
    Calculate moving averages for given periods.
    """
    for period in periods:
        df[f"MA_{period}"] = df['Close'].rolling(window=period).mean()
    return df


def calculate_volatility(df, window=14):
    """
    Calculate volatility (standard deviation of returns) over a rolling window.
    """
    df['Daily_Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Returns'].rolling(window=window).std()
    return df


def calculate_rsi(df, period=14):
    """
    Calculate the Relative Strength Index (RSI).
    """
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(0)
    return df


def calculate_macd(df, short_window=12, long_window=26, signal_window=9):
    """
    Calculate MACD (Moving Average Convergence Divergence).
    """
    df['MACD'] = df['Close'].ewm(span=short_window, adjust=False).mean() - \
        df['Close'].ewm(span=long_window, adjust=False).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
    return df


def calculate_bollinger_bands(df, window=20):
    """
    Calculate Bollinger Bands.
    """
    df['BB_Middle'] = df['Close'].rolling(window=window).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2 * \
        df['Close'].rolling(window=window).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * \
        df['Close'].rolling(window=window).std()
    return df


def calculate_momentum(df, period=10):
    """
    Calculate Momentum (Rate of Change - ROC).
    """
    df['Momentum'] = df['Close'].diff(periods=period)
    return df


def add_earnings_call_sentiment(df, sentiment_score):
    """
    Add sentiment data to the dataframe.
    """
    df['Earnings_Call_Sentiment'] = sentiment_score
    return df


def load_sentiment_data(file_path):
    sentiment_df = pd.read_csv(file_path, header=None, names=[
                               "index", "text", "overall_sentiment", "intensity_score"])
    sentiment_dict = sentiment_df.set_index(
        "index")[["overall_sentiment", "intensity_score"]].to_dict(orient="index")
    return sentiment_dict


def collect_sentiment_analysis(ticker_index, sentiment_loc="sentiment_results.csv"):
    sentiment_dict = load_sentiment_data(sentiment_loc)
    if ticker_index in sentiment_dict:
        return {
            'overall_sentiment': sentiment_dict[ticker_index]['overall_sentiment'],
            'intensity_score': float(sentiment_dict[ticker_index]['intensity_score'])
        }
    return None


def extract_tickers_from_directory(directory):
    """
    Extracts tickers and dates from folder names in the given directory.

    Args:
        directory (str): Path to the directory containing earnings call folders.

    Returns:
        list: A list of dictionaries with 'ticker' and 'date' as keys.
    """
    extracted_data = []

    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        if not os.path.isdir(folder_path):
            continue

        try:
            date, ticker = folder_name.split('_', 1)
            extracted_data.append({'ticker': ticker, 'start_date': date})
        except ValueError:
            print(f"Skipping folder: {folder_name} (unexpected format)")

    return extracted_data


def collect_all_features(ticker, start_date, end_date, reddit_data_func, sentiment_score, time_series_func):
    """
    Collect all features (price data, technical indicators, sentiment, Reddit metrics, etc.).
    """
    max_window_size = 50
    max_window_size = 50
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
    # 50 trading days of data, accounting for weekends
    adj_start_dt = start_date_dt - \
        timedelta(days=math.ceil(max_window_size * (5)))
    adj_start_str = adj_start_dt.strftime('%Y-%m-%d')
    # print(f"Fetching stock data for {ticker}...")
    df = fetch_stock_data(ticker, adj_start_str, end_date)
    features = [column for column in df.columns if column !=
                'Dividends' and column != 'Stock Splits']
    df = df[features]

    # print("Calculating technical indicators...")
    df = calculate_moving_averages(df)
    df = calculate_volatility(df)
    df = calculate_rsi(df)
    df = calculate_macd(df)
    df = calculate_bollinger_bands(df)
    df = calculate_momentum(df)
    df = add_earnings_call_sentiment(df, sentiment_score)

    mask = (df.index >= start_date)
    return df.loc[mask]


def collect_reddit_data(ticker, query="", filename="reddit_comments.csv"):

    reddit_data = pd.read_csv(
        f"data/reddit_data/{ticker}/{filename}")

    keywords = [keyword.lower() for keyword in query.split(' OR ')]

    reddit_data['tokens'] = reddit_data['body'].str.lower().apply(
        lambda x: re.findall(r'\b\w+\b', str(x)))
    all_tokens = [token for tokens in reddit_data['tokens']
                  for token in tokens]

    word_counts = Counter(all_tokens)

    mentions = sum([word_counts[keyword] for keyword in keywords])
    return {
        'mentions': mentions,
        'sentiment': np.random.choice(['positive', 'neutral', 'negative'])
    }


def mock_time_series_prediction(ticker):
    return {
        'predicted_price': np.random.uniform(100, 500)
    }


def calculate_correlation(stock_data):

    correlation_results = []
    tickers = stock_data.keys()

    for ticker in tickers:

        ticker_stock = stock_data[ticker]

        if ticker not in stock_data:
            print(f"No stock data found for {ticker}. Skipping...")
            continue

        c = 0
        for date, df in ticker_stock.iterrows():
            c += 1
            if c > 6:
                c = 0
                break

            sentiment_intensity = df['Earnings_Call_Sentiment']

            future_date = date + timedelta(days=7)

            dt = future_date.strftime('%A')
            if dt == "Sunday":
                future_date = future_date + timedelta(days=1)
            if dt == "Saturday":
                future_date = future_date + timedelta(days=2)

            start_price = ticker_stock.loc[date, 'Close']

            if future_date not in ticker_stock.index:
                break
            end_price = ticker_stock.loc[future_date, 'Close']

            percent_change = ((end_price - start_price) / start_price) * 100
            correlation_results.append({
                'ticker': ticker,
                'sentiment_intensity': sentiment_intensity,
                'percent_change': percent_change
            })

    correlation_df = pd.DataFrame(correlation_results)
    correlation_score = correlation_df[[
        'sentiment_intensity', 'percent_change']].corr().iloc[0, 1]

    return correlation_df, correlation_score
