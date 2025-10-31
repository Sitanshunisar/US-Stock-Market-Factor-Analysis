# modeling_script.py (UPDATED LOGIC)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os

print("Starting the modeling script with NEW, balanced regime logic...")

# --- 1. Data Loading ---
try:
    treasury_df = pd.read_csv('treasury_rates.csv')
    indices_df = pd.read_csv('world_indices.csv')
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure treasury_rates.csv and world_indices.csv are in the root directory.")
    exit()

# --- 2. Data Preprocessing and Feature Engineering ---

TREASURY_DATE_COL = 'DATE'
INDICES_DATE_COL = 'date'
YIELD_COLS = ['5 YR', '10 YR', '20 YR', '30 YR']
SP500_TICKER = '^GSPC'

# A. Process Treasury Data
for col in YIELD_COLS:
    treasury_df[col] = pd.to_numeric(treasury_df[col], errors='coerce')
treasury_df[YIELD_COLS] = treasury_df[YIELD_COLS].ffill().bfill()
treasury_df[TREASURY_DATE_COL] = pd.to_datetime(treasury_df[TREASURY_DATE_COL])
treasury_df = treasury_df.sort_values(TREASURY_DATE_COL).set_index(TREASURY_DATE_COL)
treasury_features = treasury_df[YIELD_COLS].diff().dropna()
print("Treasury data processed.")

# B. Process World Indices Data
indices_df[INDICES_DATE_COL] = pd.to_datetime(indices_df[INDICES_DATE_COL])
sp500_df = indices_df[indices_df['ticker'] == SP500_TICKER].copy()
sp500_df = sp500_df.sort_values(INDICES_DATE_COL).set_index(INDICES_DATE_COL)
sp500_df['log_return'] = np.log(sp500_df['close'] / sp500_df['close'].shift(1))
sp500_df['volatility_60d'] = sp500_df['log_return'].rolling(window=60).std().dropna()
market_features = sp500_df[['log_return', 'volatility_60d']].dropna()
print("Market index data processed.")


# C. Merge DataFrames
merged_df = treasury_features.join(market_features, how='inner').dropna()
print("Data merged.")

# --- D. Label Market Regimes (NEW SIMPLER LOGIC) ---
merged_df['forward_return_60d'] = merged_df['log_return'].shift(-60).rolling(window=60).sum()

# Define new thresholds: Top 33% (Bull), Bottom 33% (Bear), Middle 33% (Sideways)
return_threshold_bull = merged_df['forward_return_60d'].quantile(0.66)
return_threshold_bear = merged_df['forward_return_60d'].quantile(0.33)

def define_regime(row):
    if row['forward_return_60d'] >= return_threshold_bull:
        return 'Bull'
    elif row['forward_return_60d'] <= return_threshold_bear:
        return 'Bear'
    else:
        return 'Sideways'

merged_df['Regime'] = merged_df.apply(define_regime, axis=1)
final_df = merged_df.dropna()
print("Market regimes labeled with new logic:")
print(final_df['Regime'].value_counts()) # The classes should be much more balanced now

# --- 3. Model Training ---

features = YIELD_COLS + ['log_return', 'volatility_60d']
X = final_df[features]
y = final_df['Regime']

# A. Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# B. PCA
pca = PCA(n_components=3)
yield_indices = [X.columns.get_loc(c) for c in YIELD_COLS]
pca.fit(X_scaled[:, yield_indices])

# C. LDA
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# D. Final Classifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_lda, y)
print("Models re-trained successfully on new labels.")

# --- 4. Save Artifacts ---
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

with open(f'{models_dir}/scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
with open(f'{models_dir}/pca_model.pkl', 'wb') as f: pickle.dump(pca, f)
with open(f'{models_dir}/lda_model.pkl', 'wb') as f: pickle.dump(lda, f)
with open(f'{models_dir}/final_classifier.pkl', 'wb') as f: pickle.dump(classifier, f)

print("All new model artifacts have been saved to the 'models/' directory.")