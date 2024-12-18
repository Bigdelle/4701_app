import os
import praw
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time


def get_comments(tickers, queries, path_to_save=None):

    load_dotenv()

    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
    REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')

    comments_df = pd.DataFrame(
        columns=['subreddit', 'author', 'created_utc', 'body'])

    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
    )

    subreddits = [
        "stocks",
        "investing",
        "wallstreetbets",
        "finance",
        "economy",
        "personalfinance",
        "options",
        "pennystocks",
        "quantitativefinance",
        "dividends",
        "financialplanning",
        "stockmarket",
        "earnings",
        "StockMarketResearch"
    ]

    lent = len(tickers)
    tnumber = 1  # currently have downloaded 124/983
    start_time = time.time()
    last_stop_time = 0
    requests_made = 0

    for ticker in tickers:
        last_stop_time, requests_made, tnumber = create_comment_csvs(
            ticker, queries[ticker], subreddits, comments_df, reddit, start_time, last_stop_time, requests_made, lent, tnumber, tickers[ticker], path_to_save)


def csv_create(ticker, last_stop_time, requests_made, tnumber, lent, comments_df, path_to_save=None):
    if path_to_save is None:
        path_to_save = ""
    folder_path = f"{path_to_save}data/reddit_data/{ticker}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    comments_df.to_csv(
        f'{folder_path}/reddit_comments.csv', index=False)
    print(f"Scraped comments for {ticker}")
    print(f"{tnumber}/{lent} % complete.")
    return last_stop_time, requests_made, tnumber + 1


def create_comment_csvs(ticker, query, subreddits, comments_df, reddit, start_time, last_stop_time, requests_made, lent, tnumber, collect_time, path_to_save=None):

    print(f"Starting data collection for {ticker}")

    time_bound_left = int(datetime.strptime(
        collect_time[0], '%Y-%m-%d').timestamp())
    time_bound_right = int(datetime.strptime(
        collect_time[1], '%Y-%m-%d').timestamp())

    timestamp = f"timestamp:{collect_time[0]}..{collect_time[1]}"
    date_constrained_query = f"{query} {timestamp}"
    counter = 0

    for subreddit_name in subreddits:
        subreddit = reddit.subreddit(subreddit_name)
        search = subreddit.search(
            date_constrained_query, time_filter='all', sort='new')
        results = list(search)
        print(f"Number of submissions found: {len(results)}")

        # added this to avoid exceeding the rate limit for free tier reddit apit
        requests_made += 1

        elapsed_time = time.time() - start_time
        window = elapsed_time - last_stop_time
        if requests_made >= 75:
            # if we hit the limit, ensure we wait until the 10-minute window is over
            if window < 600:
                wait_time = 600 - window
                print(
                    f"Rate limit reached. Sleeping for {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            requests_made = 0
            last_stop_time = time.time() - start_time
            print(f"Last stop time: {requests_made}")

        for submission in search:
            submission.comments.replace_more(limit=None)
            for comment in submission.comments.list():
                counter += 1
                # Early stop for reddits with many comments
                if counter > 250:
                    return csv_create(ticker, last_stop_time, requests_made,
                                      tnumber, lent, comments_df, path_to_save)
                # filters:
                # 1. out comments older than 7 days
                # 2. out comments from bots (check comment karma)
                # 3. account age (at least 7 days old)
                # 4. comment score

                if comment.author:
                    intime = time_bound_right >= comment.created_utc >= time_bound_left
                    if intime:
                        comments_df = pd.concat(
                            [
                                comments_df,
                                pd.DataFrame({
                                    'subreddit': [subreddit_name],
                                    'author': [comment.author.name if comment.author else None],
                                    'created_utc': [datetime.utcfromtimestamp(comment.created_utc)],
                                    'body': [comment.body]
                                })
                            ],
                            ignore_index=True
                        )

    return csv_create(ticker, last_stop_time, requests_made,
                      tnumber, lent, comments_df, path_to_save)
