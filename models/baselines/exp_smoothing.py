import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from models.baselines.utils import fetch_stock_data, get_next_trading_days, evaluate_predictions
from datetime import datetime, timedelta

def predict_exponential_smoothing(stock_df, target_date, smoothing_level=0.8, days_ahead=5):
    """
    Predict the stock's closing price for the next 'days_ahead' trading days using Exponential Smoothing.
    
    Parameters:
    stock_df (pd.DataFrame): DataFrame containing the stock's closing prices.
    target_date (pd.Timestamp): The date for which predictions begin.
    smoothing_level (float): The smoothing level parameter (alpha) for exponential smoothing.
    days_ahead (int): The number of future trading days to predict.

    Returns:
    tuple: A tuple containing:
        - predictions (list): Predicted closing prices for the next 'days_ahead' days.
        - actual_prices (list): Actual closing prices for the next 'days_ahead' days.
        - next_trading_days.index (pd.Index): Index of the next 'days_ahead' trading days.
    """
    # Extract history up to and including the target_date
    history = stock_df.loc[:target_date]['Close']
    
    # Set frequency to Business Days and forward-fill missing values
    history = history.asfreq('B').ffill()
    
    # Check for any remaining NaN values
    if history.isnull().any():
        raise ValueError("History contains NaN values after forward filling. Please check the data.")
    
    # Fit the Exponential Smoothing model
    try:
        model = SimpleExpSmoothing(history).fit(smoothing_level=smoothing_level, optimized=False)
    except Exception as e:
        raise ValueError(f"Error fitting Exponential Smoothing model: {e}")
    
    # Forecast the next 'days_ahead' days
    try:
        predictions = model.forecast(days_ahead).tolist()
    except Exception as e:
        raise ValueError(f"Error during forecasting: {e}")
    
    # Retrieve actual prices for comparison
    try:
        next_trading_days = get_next_trading_days(stock_df, target_date, days_ahead)
        actual_prices = next_trading_days['Close'].tolist()
    except Exception as e:
        raise ValueError(f"Error fetching actual trading days: {e}")
    
    return predictions, actual_prices, next_trading_days.index

if __name__ == "__main__":
    ticker = "AAPL"
    target_date = datetime(2023, 9, 20)
    days_ahead = 5

    start_date = (target_date - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = (target_date + timedelta(days=days_ahead * 2)).strftime('%Y-%m-%d')

    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    stock_df = fetch_stock_data(ticker, start_date, end_date)

    if not stock_df.empty:
        target_date_index = pd.to_datetime(target_date.strftime('%Y-%m-%d'))

        print("\nRunning Exponential Smoothing Forecast:")
        try:
            es_predictions, es_actuals, es_days = predict_exponential_smoothing(
                stock_df, target_date_index, days_ahead=days_ahead
            )
            print("Exponential Smoothing Predictions:", es_predictions)
            print("Exponential Smoothing Actuals:", es_actuals)
            print("Trading Days:", es_days)
            mae = evaluate_predictions(es_predictions, es_actuals)
            print("Exponential Smoothing MAE:", mae)
        except ValueError as e:
            print(f"Error: {e}")
    else:
        print("No stock data available. Please check the ticker or date range.")
