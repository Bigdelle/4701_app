import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
from models.baselines.utils import *

def predict_linear_regression(stock_df, target_date, days_ahead=5):
    """
    Predict the stock's closing price for the next 'days_ahead' trading days using a linear regression model.
    
    Parameters:
    stock_df (pd.DataFrame): DataFrame containing the stock's closing prices.
    target_date (pd.Timestamp): The date for which predictions begin.
    days_ahead (int): The number of future trading days to predict.

    Returns:
    tuple: A tuple containing:
        - predictions (list): Predicted closing prices for the next 'days_ahead' days.
        - actual_prices (list): Actual closing prices for the next 'days_ahead' days.
        - next_trading_days.index (pd.Index): Index of the next 'days_ahead' trading days.
    """
    history = stock_df.loc[target_date - pd.Timedelta(days=10):target_date]
    days = np.arange(len(history)).reshape(-1, 1)
    prices = history['Close'].values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(days, prices)

    next_days = np.arange(len(history), len(history) + days_ahead).reshape(-1, 1)
    predictions = model.predict(next_days).flatten().tolist()

    next_trading_days = get_next_trading_days(stock_df, target_date, days_ahead)
    actual_prices = next_trading_days['Close'].tolist()

    return predictions, actual_prices, next_trading_days.index

if __name__ == "__main__":
    ticker = "AAPL"
    target_date = datetime(2023, 8, 30)
    days_ahead = 5

    start_date = (target_date - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = (target_date + timedelta(days=days_ahead * 2)).strftime('%Y-%m-%d')

    stock_df = fetch_stock_data(ticker, start_date, end_date)

    if not stock_df.empty:
        target_date_index = pd.to_datetime(target_date.strftime('%Y-%m-%d'))

        print("\nRunning Linear Regression Forecast:")
        lr_predictions, lr_actuals, lr_days = predict_linear_regression(stock_df, target_date_index, days_ahead)
        print("Linear Regression Predictions:", lr_predictions)
        print("Linear Regression Actuals:", lr_actuals)
        print("Trading Days:", lr_days)
        print("Linear Regression MAE:", evaluate_predictions(lr_predictions, lr_actuals))
    else:
        print("No stock data available. Please check the ticker or date range.")
