import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

yield_cols = ['5 YR', '10 YR', '20 YR', '30 YR']
other_features = ['log_return', 'volatility_60d']
all_features = yield_cols + other_features

# --- Load models and split info ---
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
pca = pickle.load(open('models/pca_model.pkl', 'rb'))
lda = pickle.load(open('models/lda_model.pkl', 'rb'))
classifier = pickle.load(open('models/final_classifier.pkl', 'rb'))
train_indices = pickle.load(open('models/train_indices.pkl', 'rb'))
test_indices = pickle.load(open('models/test_indices.pkl', 'rb'))
# regimes = pd.read_csv('models/assigned_regimes.csv', index_col=0) # No longer needed, we load y directly

# --- [DELETED] All data re-building code (lines 19-39) removed ---

# --- [NEW] Load the exact X and y data saved by the modeling script ---
# This fixes the "ValueError: Found array with 0 sample(s)"
X = pd.read_csv('models/final_X_data.csv', index_col=0, parse_dates=True)
# Load y, set index, and squeeze into a Series (since we saved with header=False)
y = pd.read_csv('models/final_y_data.csv', index_col=0, parse_dates=True, header=None).squeeze("columns")
y.name = "Regime" # Restore the name for clarity

# --- The rest of the script continues, transforming and analyzing the loaded data ---

features_scaled = scaler.transform(X)
yield_indices = [X.columns.get_loc(c) for c in yield_cols]
pca_scores = pca.transform(features_scaled[:, yield_indices])
full_features = np.concatenate([pca_scores, features_scaled[:, 4:]], axis=1)
lda_features = lda.transform(full_features)

train_mask = X.index.isin(train_indices)
test_mask = X.index.isin(test_indices)

X_train_lda = lda_features[train_mask]
X_test_lda = lda_features[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

y_pred_test = classifier.predict(X_test_lda)
test_accuracy = accuracy_score(y_test, y_pred_test)
y_pred_full = classifier.predict(lda_features)
full_accuracy = accuracy_score(y, y_pred_full)

print("\n====== PCA LOADINGS TABLE ======")
pca_df = pd.DataFrame(np.round(pca.components_, 3), columns=yield_cols, index=['PC1', 'PC2', 'PC3'])
print(pca_df)
print("\n====== PCA EXPLAINED VARIANCE ======")
pca_var = np.round(pca.explained_variance_ratio_, 4)
for i, v in enumerate(pca_var, 1):
    print(f"PC{i}: {v * 100:.2f}% of variance")
print("\n====== LDA COEFFICIENTS TABLE ======")
lda_coeffs = np.round(lda.coef_, 3)
lda_classes = lda.classes_ if hasattr(lda, 'classes_') else [f'Class {i}' for i in range(lda_coeffs.shape[0])]
lda_df = pd.DataFrame(lda_coeffs, columns=['PC1','PC2','PC3','log_return','volatility_60d'], index=[f"{c} vs rest" for c in lda_classes])
print(lda_df)
print("\n====== LDA INTERCEPTS ======")
print(np.round(lda.intercept_, 3))
print("\n====== KNN ACCURACY (CLUSTERING LABELS) ======")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Full Dataset Accuracy: {full_accuracy * 100:.2f}%")