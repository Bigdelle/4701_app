from flask import Flask, render_template, request, redirect, url_for, flash
import os
from models.sentiment import finbert_handler, classify_and_explain
from models.combined_model import generate_predictions

app = Flask(__name__)
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            try:
                results_df = finbert_handler(file_path)
                results = results_df.to_dict(orient='records')
                for result in results:
                    result['explanation'] = classify_and_explain(result['sentiment_intensity'], 'models/decile_analysis.pkl')
                return render_template('sentiment.html', results=results)
            except Exception as e:
                flash(f'Error processing file: {str(e)}')
                return redirect(url_for('sentiment'))
    return render_template('sentiment.html')


@app.route('/time_series', methods=['GET', 'POST'])
def time_series():
    if request.method == 'POST':
        ticker = request.form.get('ticker')
        if not ticker:
            flash('Please provide a ticker symbol')
            return redirect(url_for('time_series'))

        sentiment_score = None
        prob_pos = None
        prob_neg = None
        prob_neu = None
        
        # Handle optional earnings call file
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            try:
                # Get sentiment score from the earnings call
                results_df = finbert_handler(file_path)
                
                # Calculate average sentiment score
                sentiment_score = results_df['sentiment_intensity'].mean()
                prob_pos = results_df['mean_pos_prob'].mean()
                prob_neu = results_df['mean_neu_prob'].mean()
                prob_neg = results_df['mean_neg_prob'].mean()
            except Exception as e:
                flash(f'Error processing earnings call file: {str(e)}')
                return redirect(url_for('time_series'))
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)

        try:
            predictions = generate_predictions(ticker, sentiment_score, prob_pos, prob_neu, prob_neg)
            print(predictions)
            return render_template('time_series.html', 
                                predictions=predictions, 
                                sentiment_score=sentiment_score,
                                ticker=ticker)
        except Exception as e:
            flash(f'Error generating predictions: {str(e)}')
            return redirect(url_for('time_series'))

    return render_template('time_series.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Get Render's PORT or default to 5000
    app.run(host="0.0.0.0", port=port, debug=False)
