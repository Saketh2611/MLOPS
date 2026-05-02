import asyncio
import logging
import os
from dotenv import load_dotenv

from ml.dataFetch import DataFetcher
from ml.features   import FeatureEngineer
from ml.eda        import EDA
from ml.train      import ModelTrainer

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

SYMBOL     = "RELIANCE.BSE"
TABLE_NAME = "stock_prices"

def main():
    # Step 1: Pull from DB
    fetcher = DataFetcher()
    df = fetcher.fetch(SYMBOL, TABLE_NAME)

    # Step 2: Feature engineering
    engineer = FeatureEngineer()
    df = engineer.build(df)

    # Step 3: EDA
    eda = EDA()
    eda.run(df, engineer.feature_cols)

    # Step 4: Train all models + select best via MLflow
    trainer = ModelTrainer(feature_cols=engineer.feature_cols)
    results = trainer.train_all(df)

    print(f"\nBest model: {results['best']['name']}")
    print(f"ROC-AUC:    {results['best']['metrics']['roc_auc']}")
    print(f"\nFull comparison:\n{results['summary']}")

if __name__ == "__main__":
    main()