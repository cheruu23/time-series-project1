# Restaurant Sales Forecasting

A time series machine learning pipeline that predicts restaurant sales across three forecasting horizons — **1-day**, **7-day**, and **14-day** — using Linear Regression and Random Forest models.

## Overview

This project takes raw restaurant transaction data and builds a full forecasting pipeline: cleaning, feature engineering, target creation, model training, and evaluation. It produces three separate models, each tuned to a different business need:

| Horizon | Predicts | Used For |
|---|---|---|
| `target_1d` | Next day's total sales | Daily staffing, inventory ordering, cash flow |
| `target_7d` | Sum of next 7 days | Weekly revenue targets, supplier orders, scheduling |
| `target_14d` | Sum of next 14 days | Short-term strategic planning, budget checkpoints |

> **Note:** the original spec targeted a 30-day horizon. With only 60 days of raw transaction data available, a 28-day lag + 30-day target would require 58+ days of buffer per row — leaving almost no usable data. The horizon was shortened to 14 days to leave a workable ~32 rows for training/testing. See [Key Decisions](#key-decisions) below.

## Pipeline

1. **Data Loading & Inspection** — load Excel data, validate date parsing, check structure and sales statistics
2. **Data Cleaning** — remove test/dummy transactions, aggregate to one row per day, check for date gaps
3. **Feature Engineering** — time-based features (cyclical encoding), lag features, rolling statistics, derived ratios
4. **Target Creation** — build `target_1d`, `target_7d`, `target_14d` via forward shifting
5. **Feature Selection** — select horizon-specific feature sets (short-term lags for 1-day, longer trends for 14-day)
6. **Train/Test Split & Scaling** — chronological (non-shuffled) 80/20 split, `StandardScaler` fit on train only
7. **Model Training** — Linear Regression (baseline) + Random Forest (tuned via `GridSearchCV` with `TimeSeriesSplit`)
8. **Evaluation** — MAE, RMSE, R², MAPE, and visualizations (actual vs. predicted, feature importance, scatter plots)

## Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

### Usage

1. Place your transaction data as `Restaurant__transactions.xlsx` in the project directory (must include `Date`, `Item`, and `TotalSales` columns)
2. Open and run `time-series-full.ipynb` top to bottom in Jupyter
3. Outputs include trained models, evaluation metrics, and diagnostic charts

## Key Decisions

| Where | Decision | Why |
|---|---|---|
| Data loading | Coerce dates + explicitly check for bad values | Fails visibly instead of silently dropping rows |
| Aggregation | Always aggregate fresh from the raw table | Prevents a bug where re-running a cell progressively shrinks the data |
| Feature engineering | Cyclical sine/cosine encoding for day-of-week and month | Preserves adjacency (Dec→Jan, Sun→Mon) that a raw integer would break |
| Target creation | 30-day horizon shortened to 14-day | 60 days of raw data can't support a 58-day lag+target buffer |
| Train/test split | Chronological (not random) 80/20 split | Prevents the model from training on future data |
| Scaling | `StandardScaler` fit on training data only | Prevents test-set statistics from leaking into training |
| Model tuning | `TimeSeriesSplit` instead of standard K-Fold | Same data-leakage risk as the split, one level deeper |

## Models

- **Linear Regression** — interpretable baseline, default parameters
- **Random Forest Regressor** — captures non-linear patterns, tuned via grid search (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`)

## Evaluation Metrics

- **MAE** — average absolute error in sales units
- **RMSE** — penalizes large errors more heavily than MAE
- **R²** — proportion of sales variance explained by the model
- **MAPE** — average percentage error, for business-facing interpretation
- **Threshold accuracy** — % of predictions within 10% / 20% / 30% of actual sales

## Limitations

- Uses only historical sales patterns — no weather, holidays, local events, or promotions data
- 60 days of raw data is thin for a 14-day-ahead target; performance may be volatile
- Random Forest can overfit on small datasets even after tuning
- Longer horizons are inherently harder to predict accurately than short ones

## Next Steps

- Collect more historical data — restoring the original 30-day horizon requires 90+ days
- Add external features (holidays, weather, promotions calendar)
- Revisit hyperparameter tuning once more data is available
- Evaluate dedicated time series models (e.g. Prophet, ARIMA) as additional baselines

## License

MIT (or your preferred license)
