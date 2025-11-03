import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
import pickle
import os

# --- Data loading and feature engineering ---
treasury_df = pd.read_csv('treasury_rates.csv')
indices_df = pd.read_csv('world_indices.csv')
YIELD_COLS = ['5 YR', '10 YR', '20 YR', '30 YR']
SP500_TICKER = '^GSPC'
features = YIELD_COLS + ['log_return', 'volatility_60d']

for col in YIELD_COLS:
    treasury_df[col] = pd.to_numeric(treasury_df[col], errors='coerce')
treasury_df[YIELD_COLS] = treasury_df[YIELD_COLS].ffill().bfill()
treasury_df['DATE'] = pd.to_datetime(treasury_df['DATE'])
treasury_df = treasury_df.sort_values('DATE').set_index('DATE')
treasury_features = treasury_df[YIELD_COLS].diff().dropna()

indices_df['date'] = pd.to_datetime(indices_df['date'])
sp500_df = indices_df[indices_df['ticker'] == SP500_TICKER].copy()
sp500_df = sp500_df.sort_values('date').set_index('date')
sp500_df['log_return'] = np.log(sp500_df['close'] / sp500_df['close'].shift(1))
# [FIX] Replace inf/-inf with NaN so they can be dropped
sp500_df['log_return'] = sp500_df['log_return'].replace([np.inf, -np.inf], np.nan)
sp500_df['volatility_60d'] = sp500_df['log_return'].rolling(window=60).std()
market_features = sp500_df[['log_return', 'volatility_60d']].dropna()

merged_df = treasury_features.join(market_features, how='inner').dropna()
merged_df['forward_return_60d'] = merged_df['log_return'].shift(-60).rolling(window=60).sum()
merged_df = merged_df.dropna(subset=YIELD_COLS + ['log_return', 'volatility_60d', 'forward_return_60d'])

# --- Clustering-based regime labeling (KMeans) ---
cluster_features = merged_df[['forward_return_60d', 'volatility_60d']].dropna()
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(cluster_features)
centers = pd.DataFrame(kmeans.cluster_centers_, columns=['Mean_Return', 'Mean_Volatility'])
sorted_idxs = centers['Mean_Return'].argsort()
regime_mapping = {int(sorted_idxs[0]): 'Bear', int(sorted_idxs[1]): 'Sideways', int(sorted_idxs[2]): 'Bull'}
merged_df['Regime'] = pd.Series(clusters, index=cluster_features.index).map(regime_mapping)

# [NEW] Rename centers index with labels for saving
centers_to_save = centers.rename(index=regime_mapping)

final_df = merged_df.dropna(subset=YIELD_COLS + ['log_return', 'volatility_60d', 'Regime'])
X = final_df[features].dropna()
y = final_df['Regime'].loc[X.index].dropna()

# --- Chronological train/test split (indices saved for use in analysis) ---
n_test = int(0.2 * len(X))
train_indices = X.index[:-n_test].tolist()
test_indices = X.index[-n_test:].tolist()
X_train, X_test = X.loc[train_indices], X.loc[test_indices]
y_train, y_test = y.loc[train_indices], y.loc[test_indices]

# --- Feature Transform ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
yield_indices = [X.columns.get_loc(c) for c in YIELD_COLS]
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_scaled[:, yield_indices])
X_test_pca = pca.transform(X_test_scaled[:, yield_indices])
X_train_full = np.concatenate([X_train_pca, X_train_scaled[:, 4:]], axis=1)
X_test_full = np.concatenate([X_test_pca, X_test_scaled[:, 4:]], axis=1)
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_full, y_train)
X_test_lda = lda.transform(X_test_full)

# --- GridSearchCV for KNN optimization ---
param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan'],
    'leaf_size': [10, 20, 30, 40],
    'p': [1, 2]
}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_lda, y_train)

# [NEW] Store grid search results in a dictionary for saving
grid_search_results = {
    'best_score': grid_search.best_score_,
    'best_params': grid_search.best_params_
}

best_knn = grid_search.best_estimator_

# --- Train on ALL DATA (deployed model) ---
scaler_prod = StandardScaler()
X_scaled_prod = scaler_prod.fit_transform(X)

# [FIX] You must create NEW production models trained on 100% of the data
pca_prod = PCA(n_components=3)
X_pca_prod = pca_prod.fit_transform(X_scaled_prod[:, yield_indices])

X_full_prod = np.concatenate([X_pca_prod, X_scaled_prod[:, 4:]], axis=1)

lda_prod = LDA(n_components=2)
X_lda_prod = lda_prod.fit_transform(X_full_prod, y)
# [END FIX]

best_knn.fit(X_lda_prod, y)

# --- Save models, splits, and regimes for analysis.py ---
models_dir = 'models'
if not os.path.exists(models_dir): os.makedirs(models_dir)
with open(f'{models_dir}/scaler.pkl', 'wb') as f: pickle.dump(scaler_prod, f)

# [FIX] Save the correct production models
with open(f'{models_dir}/pca_model.pkl', 'wb') as f: pickle.dump(pca_prod, f)
with open(f'{models_dir}/lda_model.pkl', 'wb') as f: pickle.dump(lda_prod, f)
# [END FIX]

with open(f'{models_dir}/final_classifier.pkl', 'wb') as f: pickle.dump(best_knn, f)
with open(f'{models_dir}/train_indices.pkl', 'wb') as f: pickle.dump(train_indices, f)
with open(f'{models_dir}/test_indices.pkl', 'wb') as f: pickle.dump(test_indices, f)
final_df[['Regime']].to_csv(f'{models_dir}/assigned_regimes.csv')

# --- SAVE FINAL X and y DATA ---
X.to_csv(f'{models_dir}/final_X_data.csv')
y.to_csv(f'{models_dir}/final_y_data.csv', header=False) # Save y as a simple series

# --- [NEW] Save the new analysis files ---
with open(f'{models_dir}/kmeans_centers.pkl', 'wb') as f: pickle.dump(centers_to_save, f)
with open(f'{models_dir}/grid_search_results.pkl', 'wb') as f: pickle.dump(grid_search_results, f)

print("All models and data files saved successfully.")