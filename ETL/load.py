from supabase import create_client, Client
from datetime import date

class DataLoader:
    def __init__(self, url: str, key: str):
        self.client: Client = create_client(url, key)

    def already_loaded(self, table_name: str, symbol: str, check_date: date) -> bool:
        """Returns True if data for this symbol+date already exists."""
        response = (
            self.client.table(table_name)
            .select("id")
            .eq("symbol", symbol)
            .eq("date", check_date.isoformat())
            .limit(1)
            .execute()
        )
        return len(response.data) > 0

    def load_data(self, table_name: str, records: list[dict]) -> list:
        response = (
            self.client.table(table_name)
            .upsert(records, on_conflict="symbol,date")
            .execute()
        )
        return response.data