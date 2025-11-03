import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA 
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
import pickle
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Data loading and feature engineering ---
treasury_df = pd.read_csv('treasury_rates.csv')
indices_df = pd.read_csv('world_indices.csv')
YIELD_COLS = ['5 YR', '10 YR', '20 YR', '30 YR']
SP500_TICKER = '^GSPC'
# [MODIFIED] We will add two new features to this list
features = YIELD_COLS + ['log_return', 'volatility_60d', 'momentum_20d', 'slope_10_5']

for col in YIELD_COLS:
    treasury_df[col] = pd.to_numeric(treasury_df[col], errors='coerce')
treasury_df[YIELD_COLS] = treasury_df[YIELD_COLS].ffill().bfill()
treasury_df['DATE'] = pd.to_datetime( treasury_df['DATE'], dayfirst=True) # Use dayfirst parser
treasury_df = treasury_df.sort_values('DATE').set_index('DATE')
treasury_features = treasury_df[YIELD_COLS].diff().dropna()

indices_df['date'] = pd.to_datetime(indices_df['date'])
sp500_df = indices_df[indices_df['ticker'] == SP500_TICKER].copy()
sp500_df = sp500_df.sort_values('date').set_index('date')
sp500_df['log_return'] = np.log(sp500_df['close'] / sp500_df['close'].shift(1))
sp500_df['log_return'] = sp500_df['log_return'].replace([np.inf, -np.inf], np.nan)
sp500_df['volatility_60d'] = sp500_df['log_return'].rolling(window=60).std()

# --- [NEW] Add a 'Momentum' feature ---
sp500_df['momentum_20d'] = sp500_df['log_return'].rolling(window=20).sum()
market_features = sp500_df[['log_return', 'volatility_60d', 'momentum_20d']].dropna()

# --- [NEW] Add 'Yield Curve Slope' feature ---
# We must use the original (non-diff'd) treasury data for this.
treasury_df['slope_10_5'] = treasury_df['10 YR'] - treasury_df['5 YR']
slope_feature = treasury_df[['slope_10_5']].dropna()

# --- [MODIFIED] Join all features together ---
merged_df = treasury_features.join(market_features, how='inner').dropna()
merged_df = merged_df.join(slope_feature, how='inner').dropna() # Join the new slope feature

merged_df['forward_return_60d'] = merged_df['log_return'].shift(-60).rolling(window=60).sum()
merged_df = merged_df.dropna(subset=features + ['forward_return_60d'])

# --- Clustering (UNSCALED) ---
print("Running K-Means clustering on UNCALED data...")
cluster_features = merged_df[['forward_return_60d', 'volatility_60d']].dropna()
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(cluster_features) 
centers = pd.DataFrame(kmeans.cluster_centers_, columns=['Mean_Return', 'Mean_Volatility'])
sorted_idxs = centers['Mean_Return'].argsort()
regime_mapping = {int(sorted_idxs[0]): 'Bear', int(sorted_idxs[1]): 'Sideways', int(sorted_idxs[2]): 'Bull'}
merged_df['Regime'] = pd.Series(clusters, index=cluster_features.index).map(regime_mapping)
centers_to_save = centers.rename(index=regime_mapping)

final_df = merged_df.dropna(subset=features + ['Regime'])
# [MODIFIED] X now has 8 features
X = final_df[features].dropna()
y = final_df['Regime'].loc[X.index].dropna()

# --- Chronological train/test split ---
n_test = int(0.2 * len(X))
train_indices = X.index[:-n_test].tolist()
test_indices = X.index[-n_test:].tolist()
X_train, X_test = X.loc[train_indices], X.loc[test_indices]
y_train, y_test = y.loc[train_indices], y.loc[test_indices]

# --- Feature Scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Train LDA model *only for analysis* ---
lda_analysis_model = LDA()
lda_analysis_model.fit(X_train_scaled, y_train)

# --- [MODIFIED] Deeper GridSearchCV ---
print("Running GridSearchCV for Random Forest...")
param_grid = {
    'n_estimators': [200, 300, 400],           # Try more trees
    'max_depth': [10, 15, 20, 25],           # Try deeper trees
    'class_weight': ['balanced'],
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': [1, 3, 5]            # Try different leaf sizes
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

grid_search_results = {
    'best_score': grid_search.best_score_,
    'best_params': grid_search.best_params_
}
best_rf = grid_search.best_estimator_

# --- Train on ALL DATA (deployed model) ---
scaler_prod = StandardScaler()
X_scaled_prod = scaler_prod.fit_transform(X)
best_rf.fit(X_scaled_prod, y)

lda_analysis_model_prod = LDA()
lda_analysis_model_prod.fit(X_scaled_prod, y)

# --- Save models ---
models_dir = 'models'
if not os.path.exists(models_dir): os.makedirs(models_dir)

with open(f'{models_dir}/scaler.pkl', 'wb') as f: pickle.dump(scaler_prod, f)
with open(f'{models_dir}/final_classifier.pkl', 'wb') as f: pickle.dump(best_rf, f)
with open(f'{models_dir}/lda_analysis_model.pkl', 'wb') as f: pickle.dump(lda_analysis_model_prod, f)
with open(f'{models_dir}/train_indices.pkl', 'wb') as f: pickle.dump(train_indices, f)
with open(f'{models_dir}/test_indices.pkl', 'wb') as f: pickle.dump(test_indices, f)
final_df[['Regime']].to_csv(f'{models_dir}/assigned_regimes.csv')
with open(f'{models_dir}/kmeans_model.pkl', 'wb') as f: pickle.dump(kmeans, f)
cluster_features.to_csv(f'{models_dir}/cluster_features_for_silhouette.csv')
with open(f'{models_dir}/kmeans_centers.pkl', 'wb') as f: pickle.dump(centers_to_save, f)
with open(f'{models_dir}/grid_search_results.pkl', 'wb') as f: pickle.dump(grid_search_results, f)
X.to_csv(f'{models_dir}/final_X_data.csv')
y.to_csv(f'{models_dir}/final_y_data.csv', header=False)

print("All models and data files saved successfully.")