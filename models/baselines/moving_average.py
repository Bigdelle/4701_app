import pandas as pd
from datetime import datetime, timedelta
from models.baselines.utils import *

def predict_moving_average(stock_df, target_date, window_size=3, days_ahead=5):
    """
    Predict the stock's closing price for the next 'days_ahead' trading days using a moving average model.
    
    Parameters:
    stock_df (pd.DataFrame): DataFrame containing the stock's closing prices.
    target_date (pd.Timestamp): The date for which predictions begin.
    window_size (int): The number of past days to include in the moving average calculation.
    days_ahead (int): The number of future trading days to predict.

    Returns:
    tuple: A tuple containing:
        - predictions (list): Predicted closing prices for the next 'days_ahead' days.
        - actual_prices (list): Actual closing prices for the next 'days_ahead' days.
        - next_trading_days.index (pd.Index): Index of the next 'days_ahead' trading days.

    Raises:
    ValueError: If there are insufficient previous data points for the moving average.
    """
    if target_date not in stock_df.index:
        raise ValueError("The target date is not in the stock data.")

    if target_date < stock_df.index[window_size]:
        raise ValueError("Not enough historical data for moving average calculations.")

    next_trading_days = get_next_trading_days(stock_df, target_date, days_ahead)
    actual_prices = next_trading_days['Close'].tolist()
    predictions = []

    for i in range(len(next_trading_days)):
        window_start = stock_df.index.get_loc(target_date) - window_size + 1 + i
        window = stock_df['Close'].iloc[window_start:window_start + window_size]
        predictions.append(window.mean())

    return predictions, actual_prices, next_trading_days.index

def run_moving_average_forecast(ticker, target_date, window_size=3, days_ahead=5):
    """
    Run the moving average forecast model for a given stock ticker and target date.
    
    This function fetches stock data, predicts future closing prices using the moving average model, and evaluates
    the predictions against the actual prices for the next 'days_ahead' trading days.
    
    Parameters:
    ticker (str): Stock ticker symbol (e.g., 'AAPL' for Apple).
    target_date (datetime): The date from which predictions start.
    window_size (int): The number of past days used for the moving average calculation.
    days_ahead (int): The number of trading days to predict.

    Returns:
    None
    """
    start_date = (target_date - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = (target_date + timedelta(days=days_ahead * 2)).strftime('%Y-%m-%d')

    print(f"Fetching data for {ticker}...")
    stock_df = fetch_stock_data(ticker, start_date, end_date)

    if stock_df.empty:
        print("No stock data available. Check the ticker or date range.")
        return

    target_date_str = target_date.strftime('%Y-%m-%d')
    target_date_index = pd.to_datetime(target_date_str)

    print(f"Stock data fetched. Running moving average forecast for {target_date.date()}...")

    try:
        predictions, actual_prices, trading_days = predict_moving_average(stock_df, target_date_index, window_size, days_ahead)
        mae = evaluate_predictions(predictions, actual_prices)

        print(f"Predicted Prices for Next {days_ahead} Trading Days: {predictions}")
        print(f"Actual Prices for Next {days_ahead} Trading Days: {actual_prices}")
        print(f"Trading Days: {trading_days}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        return predictions, actual_prices, trading_days
    except ValueError as e:
        print(e)
        return

if __name__ == "__main__":
    ticker = "AAPL"
    target_date = datetime(2024, 9, 20)
    days_ahead = 5
    window_size = 3

    run_moving_average_forecast(ticker, target_date, window_size, days_ahead)
