import pandas as pd
from datetime import datetime, timedelta
from models.baselines.moving_average import predict_moving_average
from models.baselines.naive import predict_next_days
from models.baselines.regression import predict_linear_regression
from models.baselines.exp_smoothing import predict_exponential_smoothing
from models.baselines.utils import fetch_stock_data, evaluate_predictions
import models.combined_model as cm

def compare_models(tickers, dates, days_ahead=5, window_size=3, smoothing_level=0.8):
    """
    Compare different stock prediction models for given tickers and dates, including a custom time-series model.

    Parameters:
    - tickers (list of str): List of stock ticker symbols.
    - dates (list of datetime): List of target dates for predictions.
    - days_ahead (int): Number of trading days to predict.
    - window_size (int): Window size for moving average model.
    - smoothing_level (float): Smoothing level for exponential smoothing model.

    Returns:
    - pd.DataFrame: DataFrame containing MAE for each model, ticker, and date.
    """
    results = []

    for ticker in tickers:
        for target_date in dates:
            print(f"\nProcessing Ticker: {ticker}, Date: {target_date.date()}")

            start_date = (target_date - timedelta(days=60)).strftime('%Y-%m-%d')
            end_date = (target_date + timedelta(days=30)).strftime('%Y-%m-%d')
            stock_df = fetch_stock_data(ticker, start_date, end_date)
            if stock_df.empty:
                print(f"No stock data available for {ticker} between {start_date} and {end_date}. Skipping.")
                continue

            if target_date not in stock_df.index:
                print(f"Target date {target_date.strftime('%Y-%m-%d')} not found in stock data for {ticker}. Skipping.")
                continue

            result = {
                'Ticker': ticker,
                'Date': target_date.strftime('%Y-%m-%d'),
                'Moving Average MAE': None,
                'Naive MAE': None,
                'Linear Regression MAE': None,
                'Exponential Smoothing MAE': None,
                'Custom Model MAE': None
            }

            try:
                predictions_ma, actual_ma, _ = predict_moving_average(
                    stock_df, target_date, window_size=window_size, days_ahead=days_ahead
                )
                mae_ma = evaluate_predictions(predictions_ma, actual_ma)
                result['Moving Average MAE'] = mae_ma
                print(f"  [Moving Average] MAE: {mae_ma:.2f}")
            except Exception as e:
                print(f"  [Moving Average] Error: {e}")

            try:
                predictions_naive, actual_naive, _ = predict_next_days(
                    stock_df, target_date, days_ahead=days_ahead
                )
                mae_naive = evaluate_predictions(predictions_naive, actual_naive)
                result['Naive MAE'] = mae_naive
                print(f"  [Naive] MAE: {mae_naive:.2f}")
            except Exception as e:
                print(f"  [Naive] Error: {e}")

            try:
                predictions_lr, actual_lr, _ = predict_linear_regression(
                    stock_df, target_date, days_ahead=days_ahead
                )
                mae_lr = evaluate_predictions(predictions_lr, actual_lr)
                result['Linear Regression MAE'] = mae_lr
                print(f"  [Linear Regression] MAE: {mae_lr:.2f}")
            except Exception as e:
                print(f"  [Linear Regression] Error: {e}")

            try:
                predictions_es, actual_es, _ = predict_exponential_smoothing(
                    stock_df, target_date, smoothing_level=smoothing_level, days_ahead=days_ahead
                )
                mae_es = evaluate_predictions(predictions_es, actual_es)
                result['Exponential Smoothing MAE'] = mae_es
                print(f"  [Exponential Smoothing] MAE: {mae_es:.2f}")
            except Exception as e:
                print(f"  [Exponential Smoothing] Error: {e}")

            try:
                date = (target_date - timedelta(days=60)).strftime('%Y-%m-%d')
                custom_preds = cm.generate_predictions(ticker, date=date)
                actual_custom = stock_df["Close"].iloc[:len(custom_preds)].values.tolist()
                mae_custom = evaluate_predictions(custom_preds, actual_custom)
                result['Custom Model MAE'] = mae_custom
                print(f"  [Custom Model] MAE: {mae_custom:.2f}")
            except Exception as e:
                print(f"  [Custom Model] Error: {e}")


            results.append(result)

    results_df = pd.DataFrame(results)

    mae_columns = [
        'Moving Average MAE',
        'Naive MAE',
        'Linear Regression MAE',
        'Exponential Smoothing MAE',
        'Custom Model MAE'
    ]
    average_mae = results_df[mae_columns].mean()

    best_model = average_mae.idxmin()
    best_mae = average_mae.min()

    return results_df, average_mae, best_model, best_mae

if __name__ == "__main__":
    # Anyone can define this below to run additional tests
    tickers = []# user can define any tickers they want]

    dates = [
        #user can define whatever dates they want
    ] 

    days_ahead = 5
    window_size = 10
    smoothing_level = 0.8

    comparison_results, average_mae, best_model, best_mae = compare_models(
        tickers, dates, days_ahead=days_ahead,
        window_size=window_size, smoothing_level=smoothing_level
    )

    print("\n=== Detailed Model Comparison Results ===")
    print(comparison_results)

    print("\n=== Average MAE per Model ===")
    print(average_mae)

    print(f"\nBest Model: {best_model} with an average MAE of {best_mae:.2f}")

    comparison_results.to_csv("model_comparison_results.csv", index=False)
    average_mae_df = average_mae.reset_index()
    average_mae_df.columns = ['Model', 'Average MAE']
    average_mae_df.to_csv("average_mae_per_model.csv", index=False)
    print("\nDetailed results have been saved to 'model_comparison_results.csv'.")
    print("Average MAE per model has been saved to 'average_mae_per_model.csv'.")
