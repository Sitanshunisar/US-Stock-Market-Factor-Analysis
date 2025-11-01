import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

print("Starting the modeling script with hyperparameter optimization...")

# --- 1. Data Loading ---
treasury_df = pd.read_csv('treasury_rates.csv')
indices_df = pd.read_csv('world_indices.csv')

TREASURY_DATE_COL = 'DATE'
INDICES_DATE_COL = 'date'
YIELD_COLS = ['5 YR', '10 YR', '20 YR', '30 YR']
SP500_TICKER = '^GSPC'

for col in YIELD_COLS:
    treasury_df[col] = pd.to_numeric(treasury_df[col], errors='coerce')
treasury_df[YIELD_COLS] = treasury_df[YIELD_COLS].ffill().bfill()
treasury_df[TREASURY_DATE_COL] = pd.to_datetime(treasury_df[TREASURY_DATE_COL])
treasury_df = treasury_df.sort_values(TREASURY_DATE_COL).set_index(TREASURY_DATE_COL)
treasury_features = treasury_df[YIELD_COLS].diff().dropna()

indices_df[INDICES_DATE_COL] = pd.to_datetime(indices_df[INDICES_DATE_COL])
sp500_df = indices_df[indices_df['ticker'] == SP500_TICKER].copy()
sp500_df = sp500_df.sort_values(INDICES_DATE_COL).set_index(INDICES_DATE_COL)
sp500_df['log_return'] = np.log(sp500_df['close'] / sp500_df['close'].shift(1))
sp500_df['volatility_60d'] = sp500_df['log_return'].rolling(window=60).std().dropna()
market_features = sp500_df[['log_return', 'volatility_60d']].dropna()

merged_df = treasury_features.join(market_features, how='inner').dropna()
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

features = YIELD_COLS + ['log_return', 'volatility_60d']
X = final_df[features]
y = final_df['Regime']

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# --- Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- PCA (on yield columns only) ---
yield_indices = [X.columns.get_loc(c) for c in YIELD_COLS]
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_scaled[:, yield_indices])
X_test_pca = pca.transform(X_test_scaled[:, yield_indices])

# --- Merge PCA back into full features ---
X_train_full = np.concatenate([X_train_pca, X_train_scaled[:, 4:]], axis=1)
X_test_full = np.concatenate([X_test_pca, X_test_scaled[:, 4:]], axis=1)

# --- LDA ---
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_full, y_train)
X_test_lda = lda.transform(X_test_full)

# --- KNN Hyperparameter Optimization ---
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}
knn_model = KNeighborsClassifier()
grid_search = GridSearchCV(knn_model, param_grid, cv=5)
grid_search.fit(X_train_lda, y_train)

print("Best Parameters for KNN:", grid_search.best_params_)
best_knn = grid_search.best_estimator_

# --- Model Evaluation ---
y_pred = best_knn.predict(X_test_lda)
test_accuracy = accuracy_score(y_test, y_pred)
print("\nTest Accuracy with optimized hyperparameters: {:.2f}%".format(test_accuracy * 100))
print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred))

# --- Production Model Training (use best params on all data) ---
scaler_prod = StandardScaler()
X_scaled_prod = scaler_prod.fit_transform(X)
X_pca_prod = pca.fit_transform(X_scaled_prod[:, yield_indices])
X_full_prod = np.concatenate([X_pca_prod, X_scaled_prod[:, 4:]], axis=1)
X_lda_prod = lda.fit_transform(X_full_prod, y)
classifier_prod = KNeighborsClassifier(**grid_search.best_params_)
classifier_prod.fit(X_lda_prod, y)

# --- Save Models ---
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
with open(f'{models_dir}/scaler.pkl', 'wb') as f: pickle.dump(scaler_prod, f)
with open(f'{models_dir}/pca_model.pkl', 'wb') as f: pickle.dump(pca, f)
with open(f'{models_dir}/lda_model.pkl', 'wb') as f: pickle.dump(lda, f)
with open(f'{models_dir}/final_classifier.pkl', 'wb') as f: pickle.dump(classifier_prod, f)
print("All optimized model artifacts have been saved to the 'models/' directory.")
