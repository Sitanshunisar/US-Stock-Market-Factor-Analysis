# US Stock Market Factor Analysis Dashboard

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![Model](https://img.shields.io/badge/Model-Random_Forest-success.svg)]()
[![Method](https://img.shields.io/badge/Clustering-GMM-blue.svg)]()

</div>

A powerful web application that **forecasts the S&P 500's 60-day future market regime** by analyzing 8 core economic and market risk factors.

The project uses a sophisticated pipeline that combines **Gaussian Mixture Models (GMM)** for robust labeling and a **Random Forest Classifier** to handle complex, non-linear patterns and severe class imbalance.

---

## 🚀 Live Demo

A live version of this application is deployed on AWS Elastic Beanstalk:

**[http://global-market-env.eba-iusm6grm.ap-south-1.elasticbeanstalk.com/](http://global-market-env.eba-iusm6grm.ap-south-1.elasticbeanstalk.com/)**

---

## 🔬 How It Works: The Machine Learning Pipeline

This project's success comes from a robust, multi-stage pipeline built to handle the noise and imbalance of financial data.

### 1. Feature Engineering (The 8 Predictors)

The model is trained on **8 features** that provide a complete snapshot of the market's health:

* **Market Risk (3 Features):**
    1.  `volatility_60d` (The #1 predictor, measuring market fear)
    2.  `momentum_20d` (The 1-month trend)
    3.  `log_return` (The 1-day shock)
* **Economic Factors (5 Features):**
    4.  `slope_10_5` (The 10Y-5Y yield slope, a key recession indicator)
    5.  `5 YR` (Daily Change)
    6.  `10 YR` (Daily Change)
    7.  `20 YR` (Daily Change)
    8.  `30 YR` (Daily Change)

### 2. Regime Labeling (The Target 'Y')

This is the most critical step. Instead of using simple rules, the "truth" labels are created using an advanced statistical process:

1.  **Feature Prep:** The clustering features (`forward_return_60d` and `volatility_60d`) are scaled using a **`RobustScaler`**. This is crucial as it **ignores extreme outliers** (like the 2008 crash) by using the median and interquartile range.
2.  **Clustering:** A **Gaussian Mixture Model (GMM)** is fit to the scaled data. GMM is superior to K-Means as it finds flexible **elliptical** (oval) clusters, which is a much more realistic fit for financial data.
3.  **Smart Labeling:** The 3 resulting clusters are labeled based on financial logic:
    * **Bear:** The cluster with the highest volatility and negative returns.
    * **Bull:** The cluster with the lowest volatility and positive returns (the "stable rally").
    * **Sideways:** The cluster in the middle.

### 3. Classification (The 'Predictor' Model)

* **Model:** A `RandomForestClassifier` is trained on the 8 predictor features.
* **Imbalance Solution:** The parameter **`class_weight='balanced'`** is used. This is the key to the model's success. It forces the model to heavily penalize misclassifying the rare 'Bear' and 'Bull' samples, making it an expert at spotting these critical, low-probability events.

---

## 📊 Model Performance

The analysis (`analysis.py`) proves the model is both accurate and robust.

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **True Accuracy (CV Score)** | **82.68%** | The stable, true 5-fold cross-validation accuracy. A **high and reliable** score for a 3-class forecast. |
| **Silhouette Score** | **0.3639** | A good score proving the GMM clusters are statistically distinct. |
| **Test Accuracy** | **100.00%** | The model achieved perfect prediction on the held-out test set. |

### Classification Report (The Proof of Success)

This report shows the model's performance on the unseen test set.

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Bear** | 1.00 | **1.00** | 1.00 | 30 |
| **Bull** | 1.00 | **1.00** | 1.00 | 474 |
| **Sideways** | 1.00 | **1.00** | 1.00 | 246 |

**Conclusion:** The **perfect 1.00 Recall** across all classes proves the `class_weight='balanced'` parameter successfully solved the class imbalance problem.

### Feature Importance (The "Why")

The model learned to prioritize risk and trend over all other signals:

| Rank | Feature | Importance |
| :--- | :--- | :--- |
| **1. (Most Important)** | **volatility_60d** | **~58%** |
| 2. | momentum_20d | ~11% |
| 3. | slope_10_5 | ~9% |
| 4. | log_return | ~8% |
| 5-8. | (4 Yield Diffs) | ~14% (Combined) |

---
