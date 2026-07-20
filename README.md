# 🏠 Airbnb Price Prediction Analysis

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)
![Matplotlib](https://img.shields.io/badge/Visualization-Matplotlib-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

## Project Overview

This project predicts Airbnb listing prices in New York City using Machine Learning techniques.

The objective is to identify the key factors that influence listing prices and evaluate multiple regression models for predicting Airbnb prices based on listing characteristics.

The project follows an end-to-end Machine Learning workflow including:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Visualization
- Model Building
- Model Evaluation
- Performance Comparison

---

## Dataset

The dataset contains Airbnb listings in New York City including:

- Listing Name
- Host Information
- Neighbourhood
- Room Type
- Price
- Number of Reviews
- Reviews per Month
- Availability
- Minimum Nights

Dataset File:

```
data/Air.csv
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Jupyter Notebook

---

## Project Structure

```
Airbnb-prediction-analysis/

│
├── data/
│   └── Air.csv
│
├── notebooks/
│   └── Airbnb_Price_Prediction.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── visualization.py
│   ├── models.py
│   ├── evaluation.py
│   └── utils.py
│
├── outputs/
│   ├── figures/
│   └── models/
│
├── main.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Workflow

```
Load Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Engineering
      │
      ▼
Train/Test Split
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Performance Comparison
```

---

## Machine Learning Models

The following regression models are implemented:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net Regression
- Random Forest Regressor

---

## Evaluation Metrics

The models are evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Exploratory Analysis

The project includes:

- Price Distribution
- Correlation Heatmap
- Room Type Analysis
- Neighborhood Analysis
- Residual Plots
- Feature Importance
- Prediction vs Actual Values

---

## Key Insights

- Entire homes generally command the highest prices.
- Manhattan listings have significantly higher average prices.
- Room type is a strong predictor of listing price.
- Availability and review metrics contribute to model performance.

---

## Future Improvements

- Hyperparameter tuning using Optuna
- XGBoost implementation
- LightGBM comparison
- Model deployment using Streamlit
- Docker containerization
- MLflow experiment tracking

---

## Installation

```bash
git clone https://github.com/yash-master/Airbnb-prediction-analysis.git

cd Airbnb-prediction-analysis

pip install -r requirements.txt

python main.py
```
