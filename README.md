# Global Market Regime Analysis Dashboard

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

A sophisticated web dashboard that uses a dual PCA-LDA machine learning pipeline to analyze macroeconomic data, classify the stock market's regime, and simulate hypothetical economic scenarios.

---

## About The Project

In finance, understanding the current state or "regime" of the market (e.g., Bull, Bear) is critical for strategic decision-making. This project moves beyond simple price prediction to analyze the underlying drivers of the market. It uses daily Treasury yield curve data and stock market indices to build a robust classification model, which is then served through an interactive Flask web application.

The dashboard allows users not only to see the model's current prediction but also to explore historical events and test their own economic hypotheses, making it a powerful tool for financial analysis and research.

---

## ✨ Key Features

This dashboard provides three distinct modes of analysis:

* 📅 **Point-in-Time Analysis**: Select any historical date to get a snapshot of the market regime and a breakdown of the key macroeconomic factor exposures on that day.
* 📈 **Historical Range Visualization**: Generate an interactive chart showing the evolution of market regimes over a user-defined period, plotting the S&P 500's performance against a color-coded regime background.
* 🔬 **"What-If" Scenario Simulator**: Input hypothetical market conditions—such as changes in interest rates, stock returns, and volatility—to instantly see how the model would classify the resulting economic environment.

---

## 🛠️ Technical Stack

This project is built with a modern, robust set of tools:

* **Backend**: Flask, Gunicorn
* **Machine Learning**: Scikit-learn (PCA, LDA, KNeighborsClassifier)
* **Data Science**: Pandas, NumPy
* **Frontend**: HTML, CSS, JavaScript (Fetch API)
* **Charting**: Chart.js

---

## 🔬 How It Works: The Machine Learning Pipeline

The project's core is a sophisticated workflow that transforms raw data into actionable insights.

<div align="center">
  
**Raw Data → Feature Engineering → PCA → LDA → KNN → Prediction**
  
</div>

1.  **Feature Engineering**: The target variable (market regime) is created by labeling historical data into `Bull`, `Bear`, or `Sideways` categories based on the forward-looking returns and rolling volatility of the S&P 500.

2.  **Unsupervised Factor Extraction (PCA)**: Principal Component Analysis is applied to daily Treasury yield changes. It distills the correlated interest rate data into three primary, uncorrelated macroeconomic factors that explain the shape of the yield curve.

3.  **Supervised Class Separation (LDA)**: Linear Discriminant Analysis takes all available features and finds the optimal mathematical combination that maximizes the separation between the labeled market regimes, creating an ideal, low-dimensional space for classification.

4.  **Final Prediction (KNN)**: A K-Nearest Neighbors classifier is trained on the LDA-transformed features to make the final, robust prediction for any given set of market data.

---

## 🚀 Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

* Python 3.10+
* Git

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/global-market-analyzer.git](https://github.com/your-username/global-market-analyzer.git)
    cd global-market-analyzer
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    
    # For Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the data:**
    * Place your `treasury_rates.csv` and `world_indices.csv` files in the root project directory.

5.  **Train the Models (One-Time Step):**
    * Run the modeling script to generate the `.pkl` artifacts in the `models/` folder.
    ```bash
    python modeling_script.py
    ```

6.  **Run the Flask Application:**
    ```bash
    python app.py
    ```
    * Open your browser and navigate to `http://127.0.0.1:5000`.

