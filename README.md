# Development Setup Guide

## Initial Setup

1. Install Miniconda or Anaconda if you haven't already:
   - https://docs.anaconda.com/miniconda/install/#quick-command-line-install
   - Make sure to follow all instuctions, including running `bash conda init -all`
   - Note that you may have to restart your terminal.

## Environment Setup

1. Navigate to the project root directory:

   ```bash
   cd PC_RinehiML_brb227_gwr47_igp4_jkc97_rjc398
   ```

2. Create and activate the conda environment:

   ```bash
   conda create -n cs4701demo python=3.10
   conda activate cs4701demo
   ```

3. Install dependencies from requirements.txt:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Website

1. Navigate to the project root directory:

   ```bash
   cd PC_RinehiML_brb227_gwr47_igp4_jkc97_rjc398
   ```

2. Start the application:

   ```bash
   python -m project.app
   ```

3. The website should now be running at a link given in the terminal.

   - For example, it may say "Running on http://127.0.0.1:5000"

4. Launch the provided link in your web browser.

## Sentiment Analysis

1. After launching the website, click on "Sentiment Analysis" in the navigation bar on the top right.

2. Select "Choose File" and then select the provided demo earnings call located at demo_data/GOOG_Q32024.txt in the project root directory.

3. Now select "Analyze Sentiment"

   - Note that the website may load for some time (should be less than a minute).

4. The sentiment and intensity results should appear beneath the "Analyze Sentiment" button.

## Time Series Analysis

1. After launching the website, click on "Time Series Analysis" in the navigation bar on the top right.

2. Optional: Select "Choose File" and then select the provided demo earnings call located at demo_data/GOOG_Q32024.txt in the project root directory.

3. Enter the ticker name (in this case GOOG) in the box underneath "Stock Ticker".

4. Select "Generate Predictions"

   - Again, this may take some time.

5. A graph and chart of the closing price predictions for the next 5 days should appear below.

6. If an earnings call transcript was provided, there will also be a box displaying the sentiment intensity.

## Running Tests

Again, navigate to the project root:

1. Navigate to the project root directory:

   ```bash
   cd PC_RinehiML_brb227_gwr47_igp4_jkc97_rjc398
   ```

2. Make the test script executable:

   ```bash
   chmod +x run_tests.sh
   ```

3. Run the tests from the project root:
   ```bash
   ./run_tests.sh
   ```

### Test Output

Successful test runs will show a summary of passed/failed tests in the terminal. Details and the implementation of the tests is located in the `tests/` directory.


### Generating Model Training/Validation/Test Data

To see how the data for the LSTM model was generated, refer to `model_data.ipynb` in `src/data_collection/`. The notebook will walk you through the various steps in generating model data from this kaggle dataset: https://www.kaggle.com/datasets/tpotterer/motley-fool-scraped-earnings-call-transcripts?resource=download. Initially, we used another dataset for our earnings calls, but we switched to a new one after demo 2 that contained more common stocks and more recent earnings calls.

### Creating and Predicting with the Pretrained Model

To see how the pretrained model used in the web app was trained, refer to `combined_model.ipynb` in `models/`. This notebook uses functions in `time_series.py`. We tried a variety of hyperparameters to increase the accuracy of the model. The code that interfaces with `app.py` (the web app) is in `combined_model.py`. `combined_model.py` gets recent stock data for the provided ticker, combines it with the sentiment score of the provided earnings calls, and returns predictions over the next 5 days.

### Sentiment Analysis

To see how the sentiment analysis is run, refer to `sentiment_analysis.ipynb` in `models/`. This notebook will guide you through how to use the functions in `sentiment.py`.

### Sentiment Evaluation

In order to look at the way that we generate baselines for our sentiment analysis, refer to the code in `sentiment_evaluation.ipynb` in `models/`. This notebook will walk through how the functions are used from `sentiment.py`

### Reddit Data

