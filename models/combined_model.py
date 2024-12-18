import models.time_series as ts
from typing import Optional
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import pandas as pd
import numpy as np
from src.data_collection.stockinfo import collect_all_features
import yfinance as yf

MODEL_PATH = 'models/model.pth'
SCALERS_PATH = 'models/model_scalers.pkl'
NUM_FEATURES = 21
NUM_LAYERS = 3
LOOK_BACK = 10
LOOK_FORWARD = 5
DEFAULT_SENTIMENT_SCORE = 0.2
DEFAULT_SENTIMENT_PROB = 0.333
TARGET_COLUMN = 'Close'


def prediction_date_range():
    """
    Returns a tuple of (start_date, end_date) where:
    - end_date is the last trading day from current date
    - start_date is LOOK_BACK trading days before end_date

    Returns:
        tuple: (start_date, end_date) as datetime.date objects
    """

    nyse = mcal.get_calendar('NYSE')
    current_date = datetime.now().date()

    schedule = nyse.schedule(
        start_date=current_date - timedelta(days=LOOK_BACK * 10),
        end_date=current_date
    )

    end_date = current_date
    start_date = schedule.index[-LOOK_BACK-LOOK_FORWARD + 1].date()

    return (start_date, end_date)


def prediction_data(ticker: str, sentiment_score: float, prob_pos: float, prob_neu: float, prob_neg: float):
    """
    Processes time-series data for a specified stock ticker by generating features 
    and converting the data into sequential format.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL").
        sentiment_score (float): Sentiment score to include as part of the feature set.
        prob_pos (float): The probability representing positive sentiment.
        prob_neu (float): The probability representing neutral sentiment.
        prob_neg (float): The probability representing negative sentiment.

    Returns:
        tuple: A pair (X, y), where X contains the sequential feature data and y represents the target values.
    """

    # End date is last trading day, start_date is LOOK_BACK trading days before
    start_date, end_date = prediction_date_range()

    stock_df = collect_all_features(ticker, start_date.strftime(
        '%Y-%m-%d'), end_date.strftime('%Y-%m-%d'), None, None, None)
    stock_df['Earnings_Call_Sentiment'] = sentiment_score
    stock_df['sentiment_mean_neg_prob'] = prob_neg
    stock_df['sentiment_mean_pos_prob'] = prob_pos
    stock_df['sentiment_mean_neu_prob'] = prob_neu
    return ts.prepare_sequence_data(stock_df.values, LOOK_BACK, 0, stock_df.columns.get_loc(TARGET_COLUMN))


def generate_predictions(ticker: str, provided_score: Optional[float] = None, prob_pos=None, prob_neu=None, prob_neg=None, date: Optional[str] = None):
    """
    Predicts stock prices for a specified ticker symbol using a time-series model.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL").
        provided_score (Optional[float]): An optional sentiment score to influence predictions.
                                        If omitted, a default score will be applied.
        prob_pos (float): The probability representing positive sentiment.
        prob_neu (float): The probability representing neutral sentiment.
        prob_neg (float): The probability representing negative sentiment.
        adjustment_date (Optional[str]): A date in 'YYYY-MM-DD' format to apply adjustments.
                                        If not specified, the latest closing price is used.

    Returns:
        list: A list containing the predicted stock prices.
    """
    predictor = ts.StockPredictor(NUM_FEATURES)
    predictor.setup_model(normalize=True)
    predictor.load_model(MODEL_PATH, SCALERS_PATH)

    sentiment_score = provided_score if provided_score is not None else DEFAULT_SENTIMENT_SCORE
    prob_pos = prob_pos if prob_pos is not None else DEFAULT_SENTIMENT_PROB
    prob_neu = prob_neu if prob_neu is not None else DEFAULT_SENTIMENT_PROB
    prob_neg = prob_neg if prob_neg is not None else DEFAULT_SENTIMENT_PROB

    X_test, Y_test = prediction_data(ticker, sentiment_score, prob_pos, prob_neu, prob_neg)
    predictor.load_test_data(X_test, Y_test)
    ot = predictor.predict().tolist()

    if date:
        start_date = date
        end_date = (pd.to_datetime(date) + timedelta(days=1)).strftime('%Y-%m-%d')
        stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if stock.empty:
            print(f"Warning: No data found for ticker {ticker} on {date}. Using most recent price instead.")
            m_rcp = 0 
        else:
            stock.index = stock.index.date
            m_rcp = stock.loc[pd.to_datetime(date).date(), 'Close']
    else:
        ld = yf.Ticker(ticker).history(period="5d")
        if not ld.empty:
            m_rcp = ld['Close'].dropna().iloc[-1]
        else:
            m_rcp = 0

    if ot and not date:
        b_adj = m_rcp - ot[0]
        transform = lambda x: (x * 0.5 + (x ** 2) * 0.01)
        normalize = lambda x: 0.9 + ((transform(x) / (abs(transform(x)) + 1e-8)) * 0.3)
        adj = (b_adj * normalize(b_adj)) + (0.05 * transform(b_adj))
    elif ot and date:
        adj = b_adj = m_rcp - ot[0]
    else:
        adj = 0

    preds = [p + adj for p in ot]
    return preds

if __name__ == '__main__':
    print(generate_predictions('AAPL'))
