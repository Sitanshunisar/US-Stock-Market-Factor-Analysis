# modeling_script.py (UPDATED with Model Evaluation)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("Starting the modeling script with evaluation...")

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

# D. Label Market Regimes
merged_df['forward_return_60d'] = merged_df['log_return'].shift(-60).rolling(window=60).sum()
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
print("Market regimes labeled:")
print(final_df['Regime'].value_counts())

# --- Define Features (X) and Target (y) ---
features = YIELD_COLS + ['log_return', 'volatility_60d']
X = final_df[features]
y = final_df['Regime']


# --- 3. NEW: Model Evaluation (Train/Test Split) ---
print("\n" + "="*30)
print(" MODEL EVALUATION ")
print("="*30 + "\n")

# CRITICAL: We set shuffle=False because this is time-series data.
# We must train on the past (train) and predict the future (test).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print(f"Training set size: {len(X_train)} samples (The Past)")
print(f"Test set size: {len(X_test)} samples (The Recent Future)")

# --- Train evaluation models ON THE TRAINING SET ---
scaler_eval = StandardScaler()
X_train_scaled = scaler_eval.fit_transform(X_train)

pca_eval = PCA(n_components=3)
yield_indices = [X.columns.get_loc(c) for c in YIELD_COLS]
pca_eval.fit(X_train_scaled[:, yield_indices])

lda_eval = LDA(n_components=2)
X_train_lda = lda_eval.fit_transform(X_train_scaled, y_train)

classifier_eval = KNeighborsClassifier(n_neighbors=5)
classifier_eval.fit(X_train_lda, y_train)

# --- Test the models ON THE TEST SET ---
X_test_scaled = scaler_eval.transform(X_test)
X_test_lda = lda_eval.transform(X_test_scaled)
y_pred = classifier_eval.predict(X_test_lda)

# --- Print the results ---
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Set: {accuracy * 100:.2f}%")
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred))


# --- 4. Final Model Training (Train on 100% of Data) ---
print("\n" + "="*30)
print(" FINAL PRODUCTION MODEL TRAINING ")
print("="*30 + "\n")
print("Re-training models on 100% of data for deployment...")

# A. Scaler
scaler_prod = StandardScaler()
X_scaled_prod = scaler_prod.fit_transform(X)

# B. PCA
pca_prod = PCA(n_components=3)
yield_indices = [X.columns.get_loc(c) for c in YIELD_COLS]
pca_prod.fit(X_scaled_prod[:, yield_indices])

# C. LDA
lda_prod = LDA(n_components=2)
X_lda_prod = lda_prod.fit_transform(X_scaled_prod, y)

# D. Final Classifier
classifier_prod = KNeighborsClassifier(n_neighbors=5)
classifier_prod.fit(X_lda_prod, y)
print("Production models trained successfully.")


# --- 5. Save Artifacts (Saving the 'prod' models) ---
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

with open(f'{models_dir}/scaler.pkl', 'wb') as f: pickle.dump(scaler_prod, f)
with open(f'{models_dir}/pca_model.pkl', 'wb') as f: pickle.dump(pca_prod, f)
with open(f'{models_dir}/lda_model.pkl', 'wb') as f: pickle.dump(lda_prod, f)
with open(f'{models_dir}/final_classifier.pkl', 'wb') as f: pickle.dump(classifier_prod, f)

print("All new PRODUCTION model artifacts have been saved to the 'models/' directory.")