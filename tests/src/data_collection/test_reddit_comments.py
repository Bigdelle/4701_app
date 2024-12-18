import unittest
from unittest.mock import patch, MagicMock
import os
import pandas as pd
from datetime import datetime
from src.data_collection.reddit_comments import get_comments, create_comment_csvs
import time
import shutil


class TestRedditDataCollection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_tickers = {
            "AAPL": ["01-01-2023", "01-01-2023"], "MSFT": ["01-01-2023", "01-01-2023"]}
        cls.sample_queries = {
            "AAPL": "AAPL OR \"Apple stock\"",
            "MSFT": "MSFT OR \"Microsoft stock\""
        }
        cls.sample_subreddits = [
            "stocks", "investing", "wallstreetbets"
        ]
        cls.sample_comments_df = pd.DataFrame(
            columns=['subreddit', 'author', 'created_utc', 'body'])
        cls.start_time = time.time()
        cls.last_stop_time = 0
        cls.requests_made = 0

    @patch("src.data_collection.reddit_comments.praw.Reddit")
    def test_get_comments(self, mock_reddit):

        mock_instance = MagicMock()
        mock_reddit.return_value = mock_instance
        with patch("src.data_collection.reddit_comments.create_comment_csvs") as mock_create_csvs:
            mock_create_csvs.return_value = (0, 0, 1)
            get_comments(self.sample_tickers, self.sample_queries)

        self.assertEqual(mock_create_csvs.call_count, 2 )

    @patch("src.data_collection.reddit_comments.praw.Reddit")
    def test_create_comment_csvs(self, mock_reddit):

        mock_instance = mock_reddit.return_value
        mock_subreddit = mock_instance.subreddit.return_value
        mock_submission = MagicMock()
        mock_comment = MagicMock()
        mock_comment.created_utc = int(
            datetime.utcnow().timestamp())
        mock_comment.author.name = "test_user"
        mock_comment.body = "Test comment body"
        mock_submission.comments.list.return_value = [mock_comment]
        mock_subreddit.search.return_value = [mock_submission]

        last_stop_time, requests_made, tnumber = create_comment_csvs(
            ticker="AAPL",
            query="AAPL",
            subreddits=["stocks"],
            comments_df=pd.DataFrame(
                columns=["subreddit", "author", "created_utc", "body"]),
            reddit=mock_instance,
            start_time=time.time(),
            last_stop_time=0,
            requests_made=0,
            lent=1,
            tnumber=0,
            collect_time=["2023-01-01", "2023-01-31"])

        mock_instance.subreddit.assert_called_with("stocks")
        self.assertTrue(last_stop_time == 0)
        self.assertTrue(requests_made == 1)

        os.remove("data/reddit_data/AAPL/reddit_comments.csv")

    def test_create_comment_csvs_folder_creation(self):
        folder_path = "data/reddit_data/AAPL"
        if os.path.exists(folder_path):
            os.rmdir(folder_path)

        self.assertFalse(os.path.exists(folder_path))

        os.makedirs(folder_path)
        self.assertTrue(os.path.exists(folder_path))
        os.rmdir(folder_path)

    def tearDown(self):
        folder_path = "data/reddit_data/AAPL"
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)


    def test_append_comment_to_df(self):
        """
        Testing create_comment_csvs() correctly appending comments to DataFrame
        """
        comments = [
            {
                "subreddit": "stocks",
                "author": "user1",
                "created_utc": int(datetime.utcnow().timestamp()),
                "body": "Test comment about AAPL stock"
            },

            {
                "subreddit": "investing",
                "author": "user2",
                "created_utc": int(datetime.utcnow().timestamp()),
                "body": "Test comment about Microsoft stock"
            }
        ]

        df = pd.DataFrame(
            columns=["subreddit", "author", "created_utc", "body"]
        )

        for comment in comments:
            df = pd.concat(
                [df, pd.DataFrame([comment])], ignore_index=True
            )

        self.assertEqual(len(df), 2)
        self.assertIn("user1", df["author"].values)
        self.assertIn("Test comment about AAPL stock", df["body"].values)
        self.assertIn("stocks", df["subreddit"].values)
