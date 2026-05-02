import asyncio
import os
from dotenv import load_dotenv
from ETL.pipeline import ETLPipeline

load_dotenv()

async def main():
    pipeline = ETLPipeline(
        api_key=os.getenv("ALPHAVANTAGE_API_KEY"),
        db_url=os.getenv("SUPABASE_URL"),
        db_key=os.getenv("SUPABASE_KEY"),
    )
    await pipeline.run(symbol="RELIANCE.BSE", table_name="stock_prices")

if __name__ == "__main__":
    asyncio.run(main())