# app.py

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle

# --- App Initialization ---
app = Flask(__name__)

# --- Load Model Artifacts ---
try:
    scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    pca = pickle.load(open('models/pca_model.pkl', 'rb'))
    lda = pickle.load(open('models/lda_model.pkl', 'rb'))
    classifier = pickle.load(open('models/final_classifier.pkl', 'rb'))
except FileNotFoundError:
    print("FATAL ERROR: Model files not found. Please run modeling_script.py first.")
    exit()


# --- Pre-compute all features on startup for speed ---
def prepare_features():
    """
    Loads raw data and pre-computes all features for the entire dataset.
    This uses the CORRECTED column names to match your CSV files.
    """
    # --- UPDATED: Using correct column names for your files ---
    treasury_df = pd.read_csv('treasury_rates.csv', parse_dates=['DATE']).set_index('DATE')
    indices_df = pd.read_csv('world_indices.csv', parse_dates=['date']).set_index('date')
    
    # Correctly filter for S&P 500 and use lowercase 'close'
    sp500_df = indices_df[indices_df['ticker'] == '^GSPC'].copy()

    # UPDATED: Correct yield column names
    yield_cols = ['5 YR', '10 YR', '20 YR', '30 YR']
    
    # --- ADDED: Data cleaning to match the modeling script ---
    for col in yield_cols:
        treasury_df[col] = pd.to_numeric(treasury_df[col], errors='coerce')
    treasury_df[yield_cols] = treasury_df[yield_cols].ffill().bfill()
    
    treasury_features = treasury_df[yield_cols].diff()

    # Market features
    sp500_df['log_return'] = np.log(sp500_df['close'] / sp500_df['close'].shift(1))
    sp500_df['volatility_60d'] = sp500_df['log_return'].rolling(window=60).std()
    market_features = sp500_df[['log_return', 'volatility_60d', 'close']]
    
    # Combine and drop NaNs
    all_features = pd.concat([treasury_features, market_features], axis=1).dropna()
    return all_features

# Global variable to hold all pre-computed data
ALL_FEATURES_DF = prepare_features()

def get_prediction_for_date(target_date_str):
    try:
        features_for_date = ALL_FEATURES_DF.loc[target_date_str:target_date_str]
        if features_for_date.empty: return {"error": "Data not available for this date."}
        
        # UPDATED: Use lowercase 'close'
        feature_values = features_for_date.drop(columns=['close'])
        
        # --- Make Prediction ---
        features_scaled = scaler.transform(feature_values)
        # UPDATED: Correct yield column names
        yield_indices = [feature_values.columns.get_loc(c) for c in ['5 YR', '10 YR', '20 YR', '30 YR']]
        pca_scores = pca.transform(features_scaled[:, yield_indices])[0]
        lda_features = lda.transform(features_scaled)
        prediction = classifier.predict(lda_features)[0]
        
        return {
            'predicted_regime': prediction.upper(),
            'factor_scores': {
                'Level (PC1)': f"{pca_scores[0]:.2f}", 'Slope (PC2)': f"{pca_scores[1]:.2f}", 'Curvature (PC3)': f"{pca_scores[2]:.2f}",
            },
            'regime_class': prediction.lower()
        }
    except Exception as e:
        print(f"Error during prediction: {e}") # Added for better debugging
        return {"error": f"Could not generate a prediction for {target_date_str}."}


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        selected_date = request.form['analysis_date']
    else:
        selected_date = ALL_FEATURES_DF.index.max().strftime('%Y-%m-%d')

    context = get_prediction_for_date(selected_date)
    return render_template('index.html', context=context, selected_date=selected_date)

@app.route('/analyze_range', methods=['POST'])
def analyze_range():
    data = request.get_json()
    start_date, end_date = data['start_date'], data['end_date']
    date_range_df = ALL_FEATURES_DF.loc[start_date:end_date]
    
    results = []
    for date, row in date_range_df.iterrows():
        # UPDATED: Use lowercase 'close'
        feature_values = row.drop(['close']).values.reshape(1, -1)
        features_scaled = scaler.transform(feature_values)
        lda_features = lda.transform(features_scaled)
        prediction = classifier.predict(lda_features)[0]
        # UPDATED: Use lowercase 'close'
        results.append({'date': date.strftime('%Y-%m-%d'), 'price': row['close'], 'regime': prediction})
        
    return jsonify(results)

@app.route('/simulate_scenario', methods=['POST'])
def simulate_scenario():
    data = request.get_json()
    try:
        # The order MUST match the training columns
        feature_values = np.array([[
            float(data['yr5_change']), float(data['yr10_change']), float(data['yr20_change']),
            float(data['yr30_change']), float(data['log_return']), float(data['volatility'])
        ]])

        features_scaled = scaler.transform(feature_values)
        lda_features = lda.transform(features_scaled)
        prediction = classifier.predict(lda_features)[0]

        return jsonify({'predicted_regime': prediction.upper(), 'regime_class': prediction.lower()})
    except (ValueError, KeyError):
        return jsonify({'error': 'Invalid input. Please ensure all fields are numbers.'}), 400

if __name__ == '__main__':
    app.run(debug=True)