import aiohttp

class DataIngestion:
    def __init__(self, api_key):
        self.api_key = api_key

    async def pulldata(self, symbol: str) -> dict:
        url = (
            f"https://www.alphavantage.co/query"
            f"?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={self.api_key}"
        )

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.json()

        if "Note" in data:
            raise Exception("Rate limit hit. Try again later.")
        if "Information" in data:
            raise Exception(f"API issue: {data['Information']}")
        if "Error Message" in data:
            raise Exception(f"Invalid symbol: {symbol}")
        if "Time Series (Daily)" not in data:
            raise Exception(f"Unexpected response format: {data}")

        return data