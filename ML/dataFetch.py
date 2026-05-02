import pandas as pd
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self):
        self.client: Client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )

    def fetch(self, symbol: str, table_name: str = "stock_prices") -> pd.DataFrame:
        response = (
            self.client.table(table_name)
            .select("*")
            .eq("symbol", symbol)
            .order("date", desc=False)
            .execute()
        )

        if not response.data:
            raise ValueError(f"No data found for symbol: {symbol}")

        df = pd.DataFrame(response.data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        logger.info(f"[{symbol}] Fetched {len(df)} rows from DB")
        return df