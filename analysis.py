import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, silhouette_score

yield_cols = ['5 YR', '10 YR', '20 YR', '30 YR']
other_features = ['log_return', 'volatility_60d', 'momentum_20d', 'slope_10_5'] # <-- [MODIFIED]
all_features = yield_cols + other_features

# --- Load models and split info ---
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
classifier = pickle.load(open('models/final_classifier.pkl', 'rb')) 
lda_analyzer = pickle.load(open('models/lda_analysis_model.pkl', 'rb')) 
train_indices = pickle.load(open('models/train_indices.pkl', 'rb'))
test_indices = pickle.load(open('models/test_indices.pkl', 'rb'))

kmeans_centers = pickle.load(open('models/kmeans_centers.pkl', 'rb'))
grid_search_results = pickle.load(open('models/grid_search_results.pkl', 'rb'))
kmeans = pickle.load(open('models/kmeans_model.pkl', 'rb'))
cluster_features = pd.read_csv('models/cluster_features_for_silhouette.csv', index_col=0, parse_dates=True)

# --- Load the exact X and y data ---
X = pd.read_csv('models/final_X_data.csv', index_col=0, parse_dates=True)
y = pd.read_csv('models/final_y_data.csv', index_col=0, parse_dates=True, header=None).squeeze("columns")
y.name = "Regime"

# --- K-MEANS CLUSTERING ANALYSIS ---
print("\n====== K-MEANS CLUSTER ANALYSIS ======")
print("Cluster Centers (Regime Definitions, Unscaled):")
print(kmeans_centers)

score = silhouette_score(cluster_features, kmeans.labels_)
print(f"\nSilhouette Score: {score:.4f}")
print("(A score closer to +1 is better. 0 means clusters overlap.)")

print("\nRegime Data Point Counts (Full Dataset):")
print(y.value_counts())

# --- Re-create the 8D Scaled Data ---
features_scaled = scaler.transform(X)

# --- Split the transformed data back into Train/Test ---
train_mask = X.index.isin(train_indices)
test_mask = X.index.isin(test_indices)

X_train_scaled = features_scaled[train_mask]
X_test_scaled = features_scaled[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

# --- Calculate Accuracy (using the main RF model) ---
y_pred_test = classifier.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred_test)
y_pred_full = classifier.predict(features_scaled)
full_accuracy = accuracy_score(y, y_pred_full)

# --- RANDOM FOREST ANALYSIS ---
print("\n====== RANDOM FOREST ANALYSIS ======")
print("GridSearchCV Results (from training):")
print(f"Best CV Accuracy: {grid_search_results['best_score']:.4f}")
print("Best Parameters Found:")
print(grid_search_results['best_params'])

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_test, zero_division=0))

print("\nFeature Importances (from Random Forest):")
importances = classifier.feature_importances_
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importances
}).sort_values('importance', ascending=False)
print(importance_df)

# --- LDA ANALYSIS (For Viva) ---
print("\n====== LDA COEFFICIENTS TABLE (For Analysis Only) ======")
lda_coeffs = np.round(lda_analyzer.coef_, 3)
lda_classes = lda_analyzer.classes_
# [MODIFIED] LDA table now has 8 feature columns
lda_df = pd.DataFrame(lda_coeffs, columns=all_features, index=[f"{c} vs rest" for c in lda_classes])
print(lda_df)

# --- Print Final Accuracy Scores ---
print("\n====== FINAL MODEL ACCURACY (Random Forest) ======")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Full Dataset Accuracy: {full_accuracy * 100:.2f}%")