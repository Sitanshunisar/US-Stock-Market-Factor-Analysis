# app.py

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# --- App Initialization ---
app = Flask(__name__)

# --- Load Model Artifacts ---
try:
    scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    classifier = pickle.load(open('models/final_classifier.pkl', 'rb'))
except FileNotFoundError:
    print("FATAL ERROR: Model files not found. Please run modeling_script.py first.")
    exit()


# --- Pre-compute all features on startup ---
def prepare_features():
    treasury_df = pd.read_csv('treasury_rates.csv', parse_dates=['DATE']).set_index('DATE')
    indices_df = pd.read_csv('world_indices.csv', parse_dates=['date']).set_index('date')
    
    sp500_df = indices_df[indices_df['ticker'] == '^GSPC'].copy()

    yield_cols = ['5 YR', '10 YR', '20 YR', '30 YR']
    
    for col in yield_cols:
        treasury_df[col] = pd.to_numeric(treasury_df[col], errors='coerce')
    treasury_df[yield_cols] = treasury_df[yield_cols].ffill().bfill()
    
    treasury_features = treasury_df[yield_cols].diff()

    # --- [MODIFIED] Add new features ---
    sp500_df['log_return'] = np.log(sp500_df['close'] / sp500_df['close'].shift(1))
    sp500_df['log_return'] = sp500_df['log_return'].replace([np.inf, -np.inf], np.nan)
    sp500_df['volatility_60d'] = sp500_df['log_return'].rolling(window=60).std()
    sp500_df['momentum_20d'] = sp500_df['log_return'].rolling(window=20).sum()
    market_features = sp500_df[['log_return', 'volatility_60d', 'momentum_20d', 'close']]
    
    treasury_df['slope_10_5'] = treasury_df['10 YR'] - treasury_df['5 YR']
    slope_feature = treasury_df[['slope_10_5']].dropna()

    all_features = pd.concat([treasury_features, market_features, slope_feature], axis=1).dropna()
    return all_features

ALL_FEATURES_DF = prepare_features()

def get_prediction_for_date(target_date_str):
    try:
        features_for_date = ALL_FEATURES_DF.loc[target_date_str:target_date_str]
        if features_for_date.empty: return {"error": "Data not available for this date."}
        
        # [MODIFIED] Drop 'close' to get the 8 features
        feature_values = features_for_date.drop(columns=['close'])
        
        # --- Simplified Prediction Pipeline ---
        features_scaled = scaler.transform(feature_values)
        prediction = classifier.predict(features_scaled)[0]
        
        # [MODIFIED] Show the new features
        return {
            'predicted_regime': prediction.upper(),
            'factor_scores': {
                'Log Return': f"{feature_values['log_return'].values[0]:.4f}", 
                'Volatility (60d)': f"{feature_values['volatility_60d'].values[0]:.4f}", 
                'Momentum (20d)': f"{feature_values['momentum_20d'].values[0]:.4f}",
                'Yield Slope (10Y-5Y)': f"{feature_values['slope_10_5'].values[0]:.2f}"
            },
            'regime_class': prediction.lower()
        }
    except Exception as e:
        print(f"Error during prediction for {target_date_str}: {e}")
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
        # [MODIFIED] Reshape 8 features
        feature_values = row.drop(['close']).values.reshape(1, -1)
        
        features_scaled = scaler.transform(feature_values)
        prediction = classifier.predict(features_scaled)[0]

        results.append({'date': date.strftime('%Y-%m-%d'), 'price': row['close'], 'regime': prediction})
        
    return jsonify(results)

@app.route('/simulate_scenario', methods=['POST'])
def simulate_scenario():
    data = request.get_json()
    try:
        # [MODIFIED] Create the 8-feature array
        feature_values = np.array([[
            float(data['yr5_change']), float(data['yr10_change']), float(data['yr20_change']),
            float(data['yr30_change']), float(data['log_return']), float(data['volatility']),
            float(data['momentum']), float(data['slope'])
        ]])

        features_scaled = scaler.transform(feature_values)
        prediction = classifier.predict(features_scaled)[0]

        return jsonify({'predicted_regime': prediction.upper(), 'regime_class': prediction.lower()})
    except (ValueError, KeyError):
        return jsonify({'error': 'Invalid input. Please ensure all fields are numbers.'}), 400

if __name__ == '__main__':
    app.run(debug=True)