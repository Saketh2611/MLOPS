import logging
from datetime import date, timedelta
from .pulldata import DataIngestion
from .transform import DataTransformer
from .load import DataLoader

logger = logging.getLogger(__name__)

class ETLPipeline:
    def __init__(self, api_key: str, db_url: str, db_key: str):
        self.ingestion   = DataIngestion(api_key)
        self.transformer = DataTransformer()
        self.loader      = DataLoader(db_url, db_key)

    async def run(self, symbol: str, table_name: str) -> dict:
        # Markets are closed on weekends — skip Saturday(5) and Sunday(6)
        today = date.today()
        # if today.weekday() in (5, 6):
        #     logger.info(f"[{symbol}] Weekend — skipping.")
        #     return {"status": "skipped", "reason": "weekend"}

        # Most recent trading day (yesterday's close is what Alpha Vantage serves at 3am)
        target_date = today - timedelta(days=1)

        # Step 0: Check if already loaded
        if self.loader.already_loaded(table_name, symbol, target_date):
            logger.info(f"[{symbol}] Data for {target_date} already exists — skipping.")
            return {"status": "skipped", "reason": "already_loaded", "date": str(target_date)}

        # Step 1: Pull
        raw_data = await self.ingestion.pulldata(symbol)

        # Step 2: Transform
        df = self.transformer.transform_data(raw_data,symbol)

        # Step 3: Filter to only new rows not already in DB
        existing = (
            self.loader.client.table(table_name)
            .select("date")
            .eq("symbol", symbol)
            .execute()
            .data
        )
        existing_dates = {row["date"] for row in existing}

        df["symbol"] = symbol
        df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")
        new_df = df[~df["date_str"].isin(existing_dates)].drop(columns=["date_str"])

        if new_df.empty:
            logger.info(f"[{symbol}] No new rows to insert.")
            return {"status": "skipped", "reason": "no_new_data"}

        # Step 4: Load
        new_df = new_df.fillna(0)          # ← replace NaN with 0
        # OR
        new_df = new_df.dropna()           # ← drop rows with any NaN
        records = new_df.to_dict(orient="records")
        # Convert Timestamp to string for Supabase
        for r in records:
            r["date"] = r["date"].strftime("%Y-%m-%d")

        inserted = self.loader.load_data(table_name, records)
        logger.info(f"[{symbol}] Inserted {len(inserted)} new rows into '{table_name}'")
        return {"status": "inserted", "rows": len(inserted), "date": str(target_date)}