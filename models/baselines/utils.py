import yfinance as yf
from sklearn.metrics import mean_absolute_error

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetch stock price data for a given ticker symbol and date range.
    
    Parameters:
    ticker (str): Stock ticker symbol (e.g., 'AAPL' for Apple).
    start_date (str): The start date for fetching stock data in 'YYYY-MM-DD' format.
    end_date (str): The end date for fetching stock data in 'YYYY-MM-DD' format.

    Returns:
    pd.DataFrame: A DataFrame containing the closing prices for the specified date range.
    """
    print("start date " + start_date)
    stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return stock[['Close']]

def get_next_trading_days(stock_df, target_date, days_ahead=5):
    """
    Retrieve the next available trading days after a given target date.
    
    Parameters:
    stock_df (pd.DataFrame): DataFrame containing the stock's closing prices.
    target_date (pd.Timestamp): The date from which to find the next trading days.
    days_ahead (int): The number of future trading days to retrieve.

    Returns:
    pd.DataFrame: A DataFrame containing the next 'days_ahead' trading days and their closing prices.

    Raises:
    ValueError: If there are not enough trading days available after the target date.
    """
    future_dates = stock_df.loc[target_date:].iloc[1:days_ahead + 1]
    if len(future_dates) < days_ahead:
        raise ValueError("Not enough data to fetch 5 trading days after the target date.")
    return future_dates

def evaluate_predictions(predictions, actual_prices):
    """
    Calculate the Mean Absolute Error (MAE) between predicted and actual prices.
    
    Parameters:
    predictions (list): List of predicted stock prices.
    actual_prices (list): List of actual stock prices.

    Returns:
    float: The Mean Absolute Error between predictions and actual prices.
    """
    mae = mean_absolute_error(actual_prices, predictions)
    return mae
