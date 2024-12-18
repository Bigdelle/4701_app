import os
import re
import pandas as pd
import numpy as np
import torch
import yfinance as yf
from datetime import datetime, timedelta
import time

from nltk.tokenize import sent_tokenize
import nltk
from tqdm import tqdm

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt_tab')

def load_data(data_input):
    """
    Loads data from a directory of text files, a single text file, or a DataFrame.

    Args:
        data_input (str or pd.DataFrame): Path to a directory containing text files,
                                          path to a single text file, or a DataFrame.

    Returns:
        pd.DataFrame: DataFrame with columns 'ticker' and 'text'.
    """
    if isinstance(data_input, pd.DataFrame):
        df = data_input.copy()
        if 'ticker' not in df.columns or 'text' not in df.columns:
            raise ValueError("DataFrame must contain 'ticker' and 'text' columns.")
        return df

    data = []
    if os.path.isdir(data_input):
        for root, _, files in os.walk(data_input):
            for filename in files:
                if filename.endswith('.txt'):
                    filepath = os.path.join(root, filename)
                    with open(filepath, 'r', encoding='utf-8') as file:
                        text = file.read()
                        ticker = os.path.splitext(filename)[0]  # Use filename as ticker
                        data.append({
                            'ticker': ticker,
                            'text': text
                        })
    elif os.path.isfile(data_input) and data_input.endswith('.txt'):
        with open(data_input, 'r', encoding='utf-8') as file:
            text = file.read()
            ticker = os.path.splitext(os.path.basename(data_input))[0]
            data.append({
                'ticker': ticker,
                'text': text
            })
    else:
        raise ValueError("data_input must be a DataFrame, a directory path, or a .txt file path.")

    df = pd.DataFrame(data)
    return df

def preprocess_text(text):
    """
    Cleans the input text by removing unwanted content and normalizing whitespace.

    Args:
        text (str): The input text to preprocess.

    Returns:
        str: The cleaned and preprocessed text.
    """
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def split_into_sentences(text):
    """
    Splits the input text into sentences using NLTK's sentence tokenizer.

    Args:
        text (str): The input text to split into sentences.
    Returns:
        list: A list of sentences extracted from the input text.
    """
    sentences = sent_tokenize(text)
    return sentences

def finbert_sentiment_intensity_batch(texts, model, tokenizer, device, batch_size=8):
    """
    Processes a batch of texts through FinBERT to compute sentiment intensity scores.

    Args:
        texts (list): A list of text strings to analyze.
        model (torch.nn.Module): The FinBERT model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for FinBERT.
        device (torch.device): The device to run the model on (CPU or GPU).
        batch_size (int): The number of texts to process per batch.

    Returns:
        list: A list of sentiment intensity scores for each text.
    """
    intensities = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt',
                           truncation=True, max_length=512, padding=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        batch_intensities = probs.cpu().numpy().tolist()
        intensities.extend(batch_intensities)
    return intensities

def aggregate_intensity_scores(intensities):
    """
    Aggregates sentiment intensity scores to compute:
      - overall sentiment label
      - average sentiment intensity (existing behavior)
      - mean negative, neutral, and positive probabilities (new)
    """

    # Convert list of lists into a numpy array for easier calculations
    intensities_arr = np.array(intensities)  
    # intensities_arr[:,0] = neutral probs
    # intensities_arr[:,1] = positive probs
    # intensities_arr[:,2] = negative probs

    mean_neutral = intensities_arr[:,0].mean() if len(intensities_arr) > 0 else 0.0
    mean_positive = intensities_arr[:,1].mean() if len(intensities_arr) > 0 else 0.0
    mean_negative = intensities_arr[:,2].mean() if len(intensities_arr) > 0 else 0.0

    # Existing logic to compute avg_intensity and overall_sentiment
    sentiment_scores = {'negative': -1, 'neutral': 0, 'positive': 1}
    total_score = 0
    count = 0
    for intensity in intensities:
        score = (intensity[2] * sentiment_scores['negative'] +
                 intensity[0] * sentiment_scores['neutral'] +
                 intensity[1] * sentiment_scores['positive'])
        total_score += score
        count += 1
    avg_intensity = total_score / count if count != 0 else 0

    if avg_intensity >= 0.26:
        overall_sentiment = 'positive'
    elif avg_intensity <= 0.2:
        overall_sentiment = 'negative'
    else:
        overall_sentiment = 'neutral'

    return overall_sentiment, avg_intensity, mean_negative, mean_neutral, mean_positive

def finbert_handler(data_input, batch_size=8, max_files=None):
    df = load_data(data_input)
    if max_files:
        df = df.head(max_files)

    # preprocess text
    df['clean_text'] = df['text'].apply(preprocess_text)
    df['sentences'] = df['clean_text'].apply(split_into_sentences)

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    model = AutoModelForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    all_overall_sentiments = []
    all_intensity_scores = []
    all_neg_means = []
    all_neu_means = []
    all_pos_means = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc='Processing documents', unit='document'):
        sentences = row['sentences']
        intensities = []

        num_batches = len(range(0, len(sentences), batch_size))
        for i in tqdm(range(0, len(sentences), batch_size), desc='Processing sentences', leave=False, unit='batch', total=num_batches):
            batch_sentences = sentences[i:i+batch_size]
            batch_intensities = finbert_sentiment_intensity_batch(
                batch_sentences, model, tokenizer, device, batch_size
            )
            intensities.extend(batch_intensities)

        overall_sentiment, avg_intensity, mean_neg, mean_neu, mean_pos = aggregate_intensity_scores(intensities)
        all_overall_sentiments.append(overall_sentiment)
        all_intensity_scores.append(avg_intensity)
        all_neg_means.append(mean_neg)
        all_neu_means.append(mean_neu)
        all_pos_means.append(mean_pos)

    df['finbert_sentiment'] = all_overall_sentiments
    df['sentiment_intensity'] = all_intensity_scores
    df['mean_neg_prob'] = all_neg_means
    df['mean_neu_prob'] = all_neu_means
    df['mean_pos_prob'] = all_pos_means

    result_df = df[['ticker', 'finbert_sentiment', 'sentiment_intensity', 'mean_neg_prob', 'mean_neu_prob', 'mean_pos_prob']]
    return result_df

def validate_ticker_with_delay(ticker):
    time.sleep(.8) #avoid timeouts because of too many requests
    return validate_ticker(ticker)

def validate_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1d')
        if not hist.empty:
            return True
    except Exception:
        return False
    return False

def fetch_price_changes(ticker, call_date):
    """
    Fetches stock price changes for 1, 2, and 5 days after the earnings call.

    Args:
        ticker (str): The stock ticker symbol.
        call_date (str): The date of the earnings call in 'YYYY-MM-DD' format.

    Returns:
        dict: A dictionary with percentage changes for 1, 2, and 5 days.
    """
    try:
        stock = yf.Ticker(ticker)
        start_date = datetime.strptime(call_date, '%Y-%m-%d')
        end_date = start_date + timedelta(days=7)
        hist = stock.history(start=start_date, end=end_date)

        if hist.empty:
            return {'change_1d': None, 'change_2d': None, 'change_5d': None}

        changes = {}
        for days in [1, 2, 5]:
            target_date = start_date + timedelta(days=days)
            target_date_str = target_date.strftime('%Y-%m-%d')

            if target_date_str in hist.index:
                close_on_call_date = hist.loc[start_date.strftime('%Y-%m-%d'), 'Close']
                close_on_target_date = hist.loc[target_date_str, 'Close']
                changes[f'change_{days}d'] = ((close_on_target_date - close_on_call_date) / close_on_call_date) * 100
            else:
                changes[f'change_{days}d'] = None

        return changes
    except Exception as e:
        print(f"Error fetching price changes for {ticker}: {e}")
        return {'change_1d': None, 'change_2d': None, 'change_5d': None}

def find_txt_files(directory):
    """
    Finds all .txt files in the top-level of the specified directory.

    Args:
        directory (str): Path to the directory to search.

    Returns:
        list: A list of paths to .txt files in the top-level directory.
    """
    txt_files = []
    for file in os.listdir(directory):
        if file.endswith('.txt') and os.path.isfile(os.path.join(directory, file)):
            txt_files.append(os.path.join(directory, file))
    return txt_files

def load_txt_data(directory, max_files=200):
    """
    Loads data from a directory of text files, limiting to a maximum number of files.

    Args:
        directory (str): Path to a directory containing .txt files.
        max_files (int): Maximum number of files to process (default is 200).

    Returns:
        pd.DataFrame: DataFrame with columns 'ticker', 'call_date', and 'text'.
    """
    data = []
    txt_files = find_txt_files(directory)

    txt_files = txt_files[:max_files]
    count = 1

    for filepath in txt_files:
        count += 1
        with open(filepath, 'r', encoding='utf-8') as file:
            text = file.read()
            filename = os.path.basename(filepath)
            ticker, call_date = filename.split('_')[0], filename.split('_')[1].replace('.txt', '')
            data.append({
                'ticker': ticker,
                'call_date': call_date,
                'text': text
            })

    df = pd.DataFrame(data)
    df['call_date'] = pd.to_datetime(df['call_date'], format='%Y-%m-%d', errors='coerce')
    return df

def process_with_yfinance(directory, max_files=200):
    """
    Processes the data, validates tickers, fetches stock price changes, and computes sentiment analysis.

    Args:
        directory (str): Path to a directory containing .txt files.

    Returns:
        tuple: A tuple containing:
            - valid_df (pd.DataFrame): DataFrame with valid tickers, sentiment, and stock price changes.
            - invalid_tickers (list): List of invalid tickers that failed validation.
    """
    df = load_txt_data(directory, max_files=max_files)

    print("Validating tickers...")
    df['is_valid_ticker'] = df['ticker'].apply(validate_ticker_with_delay)

    valid_df = df[df['is_valid_ticker']].copy()
    invalid_tickers = df[~df['is_valid_ticker']]['ticker'].tolist()

    if not valid_df.empty:
        print("Fetching stock price changes...")
        price_changes = []
        for _, row in tqdm(valid_df.iterrows(), total=valid_df.shape[0], desc="Fetching stock data"):
            changes = fetch_price_changes(row['ticker'], row['call_date'].strftime('%Y-%m-%d'))
            price_changes.append(changes)

        price_changes_df = pd.DataFrame(price_changes)
        valid_df = pd.concat([valid_df.reset_index(drop=True), price_changes_df.reset_index(drop=True)], axis=1)

        sentiment_df = finbert_handler(valid_df)
        valid_df = pd.merge(
            sentiment_df,
            valid_df[['ticker', 'call_date', 'change_1d', 'change_2d', 'change_5d']],
            on='ticker'
        )
    return valid_df, invalid_tickers

def save_decile_analysis(decile_analysis, filename='decile_analysis.pkl'):
    decile_analysis.to_pickle(filename)
    print(f"Decile analysis saved to {filename}")

def load_decile_analysis(filename='decile_analysis.pkl'):
    decile_analysis = pd.read_pickle(filename)
    return decile_analysis

def compute_decile_analysis(valid_df):
    """
    Computes average stock price changes for fixed 0.05 ranges of sentiment intensity.

    Args:
        valid_df (pd.DataFrame): DataFrame with sentiment analysis and stock price changes.

    Returns:
        pd.DataFrame: Range-based analysis DataFrame with average stock price changes.
    """
    if 'sentiment_intensity' not in valid_df.columns:
        raise ValueError("The input DataFrame must contain a 'sentiment_intensity' column.")
    
    valid_df = valid_df.dropna(subset=['sentiment_intensity', 'change_1d', 'change_2d', 'change_5d'])

    bins = np.arange(-1.0, 1.05, 0.05) 
    labels = [round(b + 0.025, 3) for b in bins[:-1]]
    valid_df['sentiment_range'] = pd.cut(valid_df['sentiment_intensity'], bins=bins, labels=labels, include_lowest=True)
    range_analysis = valid_df.groupby('sentiment_range')[['change_1d', 'change_2d', 'change_5d']].mean().reset_index()

    range_analysis['sentiment_range'] = range_analysis['sentiment_range'].astype(float)

    return range_analysis

def classify_and_explain(sentiment_intensity, file_path='decile_analysis.pkl'):
    """
    Classifies the sentiment intensity by finding the closest sentiment range and explains stock price changes.

    Args:
        sentiment_intensity (float): The sentiment intensity of the analyzed text.
        file_path (str): Path to the decile_analysis.pkl file.

    Returns:
        str: Explanation of the classification and expected stock price changes.
    """
    decile_analysis = load_decile_analysis(file_path)

    if decile_analysis.empty:
        return "The decile analysis file is empty. Ensure it is generated correctly."

    decile_analysis['abs_diff'] = abs(decile_analysis['sentiment_range'] - sentiment_intensity)
    closest_row = decile_analysis.loc[decile_analysis['abs_diff'].idxmin()]

    explanation = (
        f"This stock is classified this way because stocks with sentiment intensity closest to "
        f"{sentiment_intensity:.2f} typically see an average "
        f"change of {closest_row['change_1d']:.2f}% after 1 day, {closest_row['change_2d']:.2f}% after 2 days, "
        f"and {closest_row['change_5d']:.2f}% after 5 days."
    )
    return explanation
