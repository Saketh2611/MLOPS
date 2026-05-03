import asyncio
import os
import logging
from dotenv import load_dotenv
from ETL.pipeline import ETLPipeline

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    force=True,       # ← forces logging to reconfigure even if already set
)
logger = logging.getLogger(__name__)

SYMBOLS = [
    "RELIANCE.BO",   # Reliance BSE
    "TCS.BO",        # TCS BSE
    "INFY.BO",       # Infosys BSE
]
TABLE_NAME = "stock_prices"

async def main():
    print("main() started")      # ← raw print, not logging

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    db_url  = os.getenv("SUPABASE_URL")
    db_key  = os.getenv("SUPABASE_KEY")

    print(f"api_key loaded: {bool(api_key)}")
    print(f"db_url  loaded: {bool(db_url)}")
    print(f"db_key  loaded: {bool(db_key)}")

    pipeline = ETLPipeline(api_key=api_key, db_url=db_url, db_key=db_key)
    print("ETLPipeline created")

    for symbol in SYMBOLS:
        print(f"Running for {symbol}...")
        try:
            result = await pipeline.run(symbol=symbol, table_name=TABLE_NAME)
            print(f"{symbol} → {result}")
        except Exception as e:
            print(f"{symbol} → FAILED: {e}")

    print("main() done")

if __name__ == "__main__":
    asyncio.run(main())