# Global Market Regime Analysis Dashboard

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

A sophisticated web dashboard that uses a dual **PCA-LDA** machine learning pipeline to analyze macroeconomic data, classify the stock market's regime, and simulate hypothetical economic scenarios.

---

## 🚀 Live Demo

A live version of this application is deployed on AWS Elastic Beanstalk:

**[http://global-market-env.eba-iusm6grm.ap-south-1.elasticbeanstalk.com/](http://global-market-env.eba-iusm6grm.ap-south-1.elasticbeanstalk.com/)**

---

## About The Project

In finance, understanding the current state or "regime" of the market (e.g., **Bull**, **Bear**, **Sideways**) is critical for strategic decision-making. This project moves beyond simple price prediction to analyze the underlying drivers of the market. It uses daily Treasury yield curve data and S&P 500 index data to build a robust classification model, which is then served through an interactive Flask web application.

The dashboard allows users not only to see the model's current prediction but also to explore historical events and test their own economic hypotheses, making it a powerful tool for financial analysis and research.

---

## ✨ Key Features

This dashboard provides three distinct modes of analysis:

* 📅 **Point-in-Time Analysis**: Select any historical date to get a snapshot of the market regime and a breakdown of the key macroeconomic factor exposures on that day.
* 📈 **Historical Range Visualization**: Generate an interactive chart showing the evolution of market regimes over a user-defined period, plotting the S&P 500's performance against a color-coded regime background.
* 🔬 **"What-If" Scenario Simulator**: Input hypothetical market conditions—such as changes in interest rates, stock returns, and volatility—to instantly see how the model would classify the resulting economic environment.

---

## 🛠️ Technical Stack

* **Backend**: Flask, Gunicorn
* **Machine Learning**: Scikit-learn (KMeans, PCA, LDA, KNeighborsClassifier, GridSearchCV)
* **Data Science**: Pandas, NumPy
* **Frontend**: HTML, CSS, JavaScript (Fetch API)
* **Charting**: Chart.js

---

## 🔬 How It Works: The Machine Learning Pipeline

The project's core is a sophisticated workflow that transforms raw, noisy financial data into a single, actionable prediction.

<div align="center">
  
**Raw Data → K-Means Labeling → Feature Engineering → PCA (on Yields) → LDA (on Hybrid Features) → KNN → Prediction**
  
</div>

1.  **Regime Labeling (K-Means)**: The target variable (market regime) is created first. The script runs `KMeans(n_clusters=3)` on **60-day forward returns** and **60-day volatility** of the S&P 500. The resulting clusters are labeled `Bull`, `Bear`, or `Sideways` based on their average return.

2.  **Feature Engineering**: The 6 predictor features (the $X$ variables) are engineered from the *current* day's data:
    * `Daily Change in 5, 10, 20, 30-Year Treasury Yields`
    * `S&P 500 Daily Log Return`
    * `S&P 500 60-Day Volatility`

3.  **Unsupervised Factor Extraction (PCA)**: `PCA(n_components=3)` is applied *only* to the 4 correlated Treasury features. This distills them into 3 uncorrelated "Principal Components" (Level, Slope, and Curvature) that describe the yield curve's behavior.

4.  **Hybrid Feature Creation & Separation (LDA)**:
    * A **5-dimensional** hybrid feature set is created by concatenating the **3 PCA components** (from yields) and the **2 market features** (log return, volatility).
    * `LDA(n_components=2)` is then fit on this 5D data. It finds the optimal 2D projection that **maximizes the separation** between the 'Bull', 'Bear', and 'Sideways' classes.

5.  **Final Prediction (KNN)**: A `KNeighborsClassifier` is trained on the final, 2D LDA-transformed data. `GridSearchCV` is used to find the optimal hyperparameters (e.g., `n_neighbors`, `metric`) for the highest possible accuracy.

