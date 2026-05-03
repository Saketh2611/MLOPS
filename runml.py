import logging
import os
from dotenv import load_dotenv

from ML.dataFetch import DataFetcher
from ML.features   import FeatureEngineer
from ML.eda        import EDA
from ML.train      import ModelTrainer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

SYMBOL     = "RELIANCE.BO"
TABLE_NAME = "stock_prices"

def main():
    # Step 1: Pull from Supabase
    print("Fetching data from Supabase...")
    fetcher = DataFetcher()
    df = fetcher.fetch(SYMBOL, TABLE_NAME)
    print(f"Fetched {len(df)} rows")

    # Step 2: Feature engineering
    print("Building features...")
    engineer = FeatureEngineer()
    df = engineer.build(df)
    print(f"Features built: {len(df)} rows, {len(engineer.feature_cols)} features")

    # Step 3: EDA
    print("Running EDA...")
    eda = EDA()
    eda.run(df, engineer.feature_cols)
    print("EDA plots saved to ML/eda_output/")

    # Step 4: Train
    print("Training models...")
    trainer = ModelTrainer(feature_cols=engineer.feature_cols)
    results = trainer.train_all(df)

    print(f"\n✅ Best model : {results['best']['name']}")
    print(f"   ROC-AUC    : {results['best']['metrics']['roc_auc']}")
    print(f"   Accuracy   : {results['best']['metrics']['accuracy']}")
    print(f"   F1 Score   : {results['best']['metrics']['f1']}")
    print(f"\nFull comparison:\n{results['summary']}")

if __name__ == "__main__":
    main()