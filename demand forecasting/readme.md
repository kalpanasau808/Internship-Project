# Demand Forecasting for Retail

## Project Overview
This project focuses on predicting future product demand using historical sales data, promotional trends, and seasonal events (such as holidays). Accurate demand forecasting helps retail businesses optimize inventory management, prevent stockouts, reduce holding costs, and enhance overall operational efficiency.

We implement and compare statistical, modern machine learning, and deep learning architectures to capture underlying trends, seasonality, and exogenous variations in retail transaction datasets.

---

## Dataset
The project utilizes publicly available retail transaction data from Kaggle:

* **Dataset Source:** [Store Sales - Time Series Forecasting (Kaggle)](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)
* **Alternative Dataset:** [Retail Data Analytics (Kaggle)](https://www.kaggle.com/datasets/manjeetsingh/retail-data-analytics)


---

## Methodology & Models
We evaluate three distinct time-series forecasting approaches:

1. **ARIMA (AutoRegressive Integrated Moving Average)**
   * Classical statistical modeling baseline for stationary time-series data.
   * Captures linear relationships and autoregressive trends.

2. **Prophet**
   * Additive forecasting model developed by Meta.
   * Handles non-linear trends with yearly, weekly, and daily seasonality, alongside holiday effects.

3. **LSTM (Long Short-Term Memory Network)**
   * Recurrent Neural Network (RNN) architecture capable of learning complex temporal dependencies and sequential patterns.
   * Trained on lagged sequences and continuous features.

---

## Evaluation Metrics
Model performance is evaluated using standard quantitative forecasting metrics:

* **Root Mean Squared Error (RMSE):** Measures overall magnitude of forecast errors, heavily penalizing large deviations.
* **Mean Absolute Percentage Error (MAPE):** Provides relative, percentage-based error evaluation across varying scales of product demand.

---

## Project Structure
├── data/                  # Processed and raw dataset files
├── notebooks/             # Exploratory Data Analysis & Modeling Notebooks
├── requirements.txt       # Dependencies and required libraries
├── README.md              # Project documentation


