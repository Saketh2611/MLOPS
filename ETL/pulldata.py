import yfinance as yf
import logging

logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self, api_key=None):
        pass  # no API key needed

    async def pulldata(self, symbol: str, period: str = "5y") -> dict:
        """
        symbol: yfinance format
                BSE stocks → "RELIANCE.BO"  (not .BSE)
                NSE stocks → "RELIANCE.NS"
                US stocks  → "AAPL"
        period: 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max
        """
        logger.info(f"[{symbol}] Fetching from yfinance ({period})...")

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period)

        if df.empty:
            raise Exception(f"No data returned for symbol: {symbol}")

        logger.info(f"[{symbol}] Got {len(df)} rows from yfinance")

        # Return in same format rest of pipeline expects
        return {"_dataframe": df, "symbol": symbol}