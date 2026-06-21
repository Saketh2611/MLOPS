# MLOPS Project - Complete Explanation

## Overview

This project is a **Stock Price Movement Prediction** system that:
1. Pulls daily stock data (RELIANCE, TCS, INFY from BSE)
2. Cleans and transforms it (ETL pipeline)
3. Stores it in Supabase (cloud PostgreSQL)
4. Trains ML models to predict whether tomorrow's closing price will go UP or DOWN
5. Tracks experiments with MLflow and saves the best model to Supabase Storage

---

## ETL Pipeline - How It Works

The ETL (Extract, Transform, Load) pipeline lives in the `ETL/` folder and runs in 4 steps:

### Step 1: Extract (pulldata.py)

- Uses the **yfinance** library to pull historical stock data
- Fetches 5 years of daily OHLCV (Open, High, Low, Close, Volume) data
- Supports BSE stocks (e.g., `RELIANCE.BO`), NSE stocks (e.g., `RELIANCE.NS`), and US stocks (e.g., `AAPL`)
- Returns raw data as a pandas DataFrame

### Step 2: Transform (transform.py)

The transformer cleans the raw data through these checks:

| Step | What It Does |
|------|-------------|
| 1. Drop duplicates | Removes rows with same symbol + date |
| 2. Drop nulls | Removes rows missing any OHLCV value |
| 3. Drop invalid prices | Removes rows where price is zero or negative |
| 4. Drop zero volume | Removes days with no trading volume |
| 5. OHLC sanity check | Removes rows where High < Low, or High < Open/Close, etc. |
| 6. Outlier detection | Flags (warns about) days with >20% price change |
| 7. Derived columns | Adds `daily_return`, `price_range`, and `vwap_approx` |

**Derived columns explained:**
- `daily_return` = percentage change from previous day's close
- `price_range` = High - Low (how much the price moved in a day)
- `vwap_approx` = (High + Low + Close) / 3 (approximate volume-weighted average price)

### Step 3: Load (load.py)

- Connects to **Supabase** (cloud database)
- Checks if data already exists for a given symbol + date (avoids duplicates)
- Uses **upsert** (insert or update) to load records into the `stock_prices` table
- Only inserts rows that don't already exist in the database

### Step 4: Orchestration (pipeline.py)

The `ETLPipeline` class ties everything together:
1. Checks if yesterday's data is already loaded (skips if yes)
2. Pulls full history from yfinance
3. Transforms and cleans the data
4. Filters out rows already in the database
5. Loads only new rows into Supabase

### Scheduling (Airflow DAG)

The file `dags/StockEtlDag.py` schedules the ETL pipeline using **Apache Airflow**:
- Runs at **3:00 AM every weekday** (Monday-Friday)
- Processes 3 stocks: RELIANCE.BSE, TCS.BSE, INFY.BSE
- Each stock gets its own Airflow task (runs independently)
- Has 2 retries with 5-minute delay on failure

---

## ML Training Pipeline - How It Works

The ML pipeline lives in the `ML/` folder and is triggered by running `runml.py`.

### Step 1: Data Fetch (dataFetch.py)

- Pulls the cleaned stock data from Supabase (the same data the ETL pipeline loaded)
- Fetches all rows for a given symbol, sorted by date

### Step 2: Feature Engineering (features.py)

This is where raw price data becomes useful inputs for the ML model.

#### Features Used (22 total):

| Category | Features | Explanation |
|----------|----------|-------------|
| **Raw OHLCV** | open, high, low, close, volume | Basic daily price and volume data |
| **From ETL** | daily_return, price_range, vwap_approx | Percentage change, day's range, approx VWAP |
| **Lag Features** | prev_close, prev_return, prev_volume, prev_range | Yesterday's values (gives model memory of recent past) |
| **Moving Averages** | ma_5, ma_10, ma_20 | Average closing price over last 5, 10, 20 days |
| **MA Signals** | close_vs_ma5, close_vs_ma10, close_vs_ma20 | How far today's close is from moving averages (in %) |
| **Volatility** | volatility_5, volatility_10 | Standard deviation of returns over 5 and 10 days |
| **Volume** | volume_ma_5, volume_ratio | 5-day average volume and today's volume relative to it |

#### Target Variable:

```
target = 1 if tomorrow's close > today's close (price goes UP)
target = 0 if tomorrow's close <= today's close (price goes DOWN)
```

This makes it a **binary classification** problem: predict UP (1) or DOWN (0).

### Step 3: EDA (eda.py)

Before training, the system runs Exploratory Data Analysis:
- **Class balance check** - Are UP and DOWN days roughly equal? Warns if imbalanced
- **Outlier detection** - Uses IQR method to find extreme values in each feature
- **Correlation matrix** - Shows which features are related to each other and to the target

Saves plots to `ML/eda_output/` (correlation heatmap, outlier boxplots).

### Step 4: Training (train.py)

#### Models Trained:

Three models are trained and compared:

| Model | Type | Key Characteristics |
|-------|------|-------------------|
| **Logistic Regression** | Linear model | Simple, fast, interpretable. Uses L1/L2 regularization |
| **Random Forest** | Ensemble of decision trees | Handles non-linear patterns, resistant to overfitting |
| **XGBoost** | Gradient boosted trees | Often best performer, handles complex patterns |

#### How Training Works:

1. **Time-based train/test split** - First 80% of data for training, last 20% for testing. No random shuffling (important for time series to avoid data leakage - you can't use future data to predict the past)

2. **Sklearn Pipeline** - Each model is wrapped in a pipeline:
   - `StandardScaler` - Normalizes features to mean=0, std=1 (important for Logistic Regression)
   - `Classifier` - The actual model

3. **Hyperparameter tuning with RandomizedSearchCV** - Tests 30 random combinations of hyperparameters for each model

4. **Best model selection** - Picks the model with highest ROC-AUC on the test set

5. **Model saving** - Saves best model as `models/best_model.pkl` and uploads to Supabase Storage

---

## Cross-Validation Method: TimeSeriesSplit

This project uses **TimeSeriesSplit** with 5 splits. This is specifically designed for time series data.

### Why Not Regular K-Fold?

Regular K-Fold randomly shuffles data into folds. For stock data, this would mean:
- Training on March 2024 data and testing on January 2024 data
- This is "looking into the future" - the model would learn patterns it shouldn't have access to
- Results would be overly optimistic and not reflect real-world performance

### How TimeSeriesSplit Works:

With 5 splits, the data is divided like this:

```
Split 1: Train [----]           Test [--]
Split 2: Train [--------]       Test [--]
Split 3: Train [------------]   Test [--]
Split 4: Train [----------------] Test [--]
Split 5: Train [--------------------] Test [--]
```

Each split:
- Training data always comes BEFORE the test data (respects time order)
- Training set grows with each fold (more historical data)
- Test set is always the next time period after training

This mimics real-world usage: you train on historical data and predict the future.

### Cross-Validation Configuration:

```
- Method: TimeSeriesSplit (5 folds)
- Search: RandomizedSearchCV (30 iterations)
- Scoring metric: ROC-AUC
- Parallel jobs: -1 (uses all CPU cores)
```

---

## Metrics Used to Evaluate Models

| Metric | What It Measures |
|--------|-----------------|
| **Accuracy** | % of correct predictions overall |
| **Precision** | Of all predicted UP days, how many actually went UP |
| **Recall** | Of all actual UP days, how many did the model catch |
| **F1 Score** | Balance between Precision and Recall |
| **ROC-AUC** | Model's ability to distinguish UP from DOWN (used to pick best model) |

---

## Model Versioning and Storage

After training:
1. Best model is saved locally as `models/best_model.pkl`
2. A versioned copy is uploaded to Supabase Storage (e.g., `xgboost_20260503_113000.pkl`)
3. The `best_model.pkl` in Supabase is overwritten with the latest best model
4. Model metadata (name, metrics, timestamp) is logged to a `model_versions` table in Supabase
5. All experiments are tracked in **MLflow** (parameters, metrics, artifacts)

---

## How to Run

| Command | What It Does |
|---------|-------------|
| `python main.py` | Runs the ETL pipeline for all 3 stocks |
| `python runml.py` | Runs the full ML training pipeline |
| `airflow scheduler` | Starts the nightly ETL schedule |

---

## Project Architecture Summary

```
main.py (runs ETL manually)
   |
   v
ETL/
   pulldata.py   --> yfinance API
   transform.py  --> clean + derive columns
   load.py       --> push to Supabase
   pipeline.py   --> orchestrates above 3

runml.py (runs ML training)
   |
   v
ML/
   dataFetch.py  --> pull from Supabase
   features.py   --> create 22 features + target
   eda.py        --> plots + analysis
   train.py      --> train 3 models, pick best, upload

dags/
   StockEtlDag.py --> Airflow DAG (schedules ETL at 3AM weekdays)
```
