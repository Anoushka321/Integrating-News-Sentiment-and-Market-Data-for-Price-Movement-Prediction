# Integrating News Sentiment and Market Data for Price Movement Prediction


## Overview
This project focuses on predicting the **directional price movement of scrap metals** using advanced machine learning techniques. The approach integrates multi-source financial data, sentiment analysis, and feature engineering to build a robust predictive model optimized for price movement forecasting.

## Key Objectives
- **Sentiment Analysis**: Leveraging **FinBERT** and **GPT-based transformer models** to quantify market sentiment from financial news articles.
- **Feature Engineering**: Implementing dimensionality reduction via **PCA, LASSO regularization**, and **Information Gain techniques**.
- **Ensemble Learning Models**: Developing and fine-tuning models using **XGBoost, CatBoost, LightGBM, and advanced boosting strategies**.
- **Prediction & Evaluation**: Conducting comprehensive performance analysis using **ROC-AUC, Precision-Recall Curves, F1-score, and MCC**.


## Getting Started
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/scrap-metal-prediction.git
cd scrap-metal-prediction
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the Main Script
```bash
python main.py
```

## Data Sources & Processing Pipeline
The dataset is a composite of **macro-economic indicators, financial news sentiment scores, and historical price trends**:
- **Benzinga & Yahoo Finance**: Time-series financial news sentiment extraction
- **AlphaVantage API**: Real-time market sentiment indicators & historical stock data
- **FRED (Federal Reserve Economic Data)**: Macro-economic indicators affecting commodity markets
- **Weather Data**: External environmental influences on metal prices

The **ETL pipeline** processes raw data by:
- **Cleaning and handling missing values**
- **Standardizing & normalizing numerical features**
- **Applying feature selection techniques**
- **Merging multi-source data into structured input matrices**

## Feature Engineering & Model Training
This project implements multiple feature engineering strategies to maximize predictive power:
- **Dimensionality Reduction**: PCA & LASSO regression for sparse feature selection
- **Sentiment Quantification**: NLP-based financial sentiment scoring using **FinBERT & GPT**
- **Time-Series Transformations**: Rolling averages, differencing, and trend decomposition
- **Feature Aggregation**: Clustering-based grouping and cross-feature interactions

**Optimized Machine Learning Models:**
- **XGBoost with hyperparameter tuning**
- **LightGBM with GPU acceleration**
- **CatBoost with Bayesian Optimization**
- **Random Forest for feature importance analysis**
- **Stacked Ensemble Models for boosting generalization**

## Performance Metrics & Model Evaluation
Comprehensive evaluation ensures robust performance benchmarking:
- **ROC-AUC & Precision-Recall Curves** for classification assessment
- **F1 Score & Matthews Correlation Coefficient (MCC)** for imbalanced datasets
- **SHAP (SHapley Additive exPlanations) Analysis** for feature interpretability
- **Cross-validation with time-series split for robust model validation**

## Technical Skills Applied
- **Advanced Machine Learning & AI**: XGBoost, CatBoost, LightGBM, Random Forest
- **Natural Language Processing (NLP)**: Transformer-based sentiment analysis (FinBERT, GPT)
- **Time-Series Forecasting & Financial Analysis**
- **Big Data Processing & Scalable ETL Pipelines**
- **Feature Selection & Dimensionality Reduction**
- **Hyperparameter Optimization (Bayesian, Grid, Random Search)**
- **Statistical Modeling & Advanced Visualization**

## Future Enhancements
- **Integration of Real-Time Market Data** for dynamic model updates
- **Deep Reinforcement Learning (PPO, DDPG)** for adaptive trading strategies
- **Automated Data Pipelines** for seamless model retraining
- **Explainable AI (XAI) Techniques** using LIME & SHAP for transparency

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

