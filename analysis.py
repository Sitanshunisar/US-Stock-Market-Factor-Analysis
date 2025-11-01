import pickle
import numpy as np
import pandas as pd

# === Define features and load models ===
yield_cols = ['5 YR', '10 YR', '20 YR', '30 YR']
other_features = ['log_return', 'volatility_60d']
all_features = yield_cols + other_features

scaler = pickle.load(open('models/scaler.pkl', 'rb'))
pca = pickle.load(open('models/pca_model.pkl', 'rb'))
lda = pickle.load(open('models/lda_model.pkl', 'rb'))
classifier = pickle.load(open('models/final_classifier.pkl', 'rb'))

# === REBUILD X and y from raw data ===
treasury = pd.read_csv('treasury_rates.csv')
indices = pd.read_csv('world_indices.csv')
SP500_TICKER = '^GSPC'

for col in yield_cols:
    treasury[col] = pd.to_numeric(treasury[col], errors='coerce')
treasury[yield_cols] = treasury[yield_cols].ffill().bfill()
treasury['DATE'] = pd.to_datetime(treasury['DATE'])
treasury = treasury.sort_values('DATE').set_index('DATE')
treasury_features = treasury[yield_cols].diff().dropna()

indices['date'] = pd.to_datetime(indices['date'])
sp500_df = indices[indices['ticker'] == SP500_TICKER].copy()
sp500_df = sp500_df.sort_values('date').set_index('date')
sp500_df['log_return'] = np.log(sp500_df['close'] / sp500_df['close'].shift(1))
sp500_df['volatility_60d'] = sp500_df['log_return'].rolling(window=60).std()
market_features = sp500_df[['log_return', 'volatility_60d']].dropna()

# Merge & drop NaNs
merged = treasury_features.join(market_features, how='inner').dropna()
merged['forward_return_60d'] = merged['log_return'].shift(-60).rolling(window=60).sum()
return_threshold_bull = merged['forward_return_60d'].quantile(0.66)
return_threshold_bear = merged['forward_return_60d'].quantile(0.33)

def define_regime(row):
    if row['forward_return_60d'] >= return_threshold_bull:
        return 'Bull'
    elif row['forward_return_60d'] <= return_threshold_bear:
        return 'Bear'
    else:
        return 'Sideways'

merged['Regime'] = merged.apply(define_regime, axis=1)
final_df = merged.dropna()
X = final_df[all_features]
y = final_df['Regime']

# === Transform features and calculate accuracy ===
features_scaled = scaler.transform(X)
yield_indices = [X.columns.get_loc(c) for c in yield_cols]
pca_scores = pca.transform(features_scaled[:, yield_indices])
full_features = np.concatenate([pca_scores, features_scaled[:, 4:]], axis=1)
lda_features = lda.transform(full_features)
y_pred = classifier.predict(lda_features)
accuracy = np.mean(y_pred == y)

print("\n====== PCA LOADINGS TABLE ======")
pca_df = pd.DataFrame(np.round(pca.components_, 3), columns=yield_cols, index=['PC1(Level)', 'PC2(Slope)', 'PC3(Curvature)'])
print(pca_df)

print("\n====== PCA EXPLAINED VARIANCE ======")
pca_var = np.round(pca.explained_variance_ratio_, 4)
for i, v in enumerate(pca_var, 1):
    print(f"PC{i}: {v * 100:.2f}% of variance")

print("\n====== LDA COEFFICIENTS TABLE ======")
lda_coeffs = np.round(lda.coef_, 3)
lda_classes = lda.classes_ if hasattr(lda, 'classes_') else [f'Class {i}' for i in range(lda_coeffs.shape[0])]
lda_df = pd.DataFrame(lda_coeffs, columns=['PC1(Level)','PC2(Slope)','PC3(Curvature)','log_return','volatility_60d'], index=[f"{c} vs rest" for c in lda_classes])
print(lda_df)

print("\n====== LDA INTERCEPTS ======")
print(np.round(lda.intercept_, 3))

print("\n====== FULL DATASET ACCURACY ======")
print(f"Accuracy: {accuracy * 100:.2f}%")
