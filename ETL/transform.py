import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataTransformer:
    def transform_data(self, raw_data: dict, symbol: str) -> pd.DataFrame:

        # yfinance returns a DataFrame directly
        if "_dataframe" in raw_data:
            df = raw_data["_dataframe"].copy()
            df = df.reset_index()
            df = df.rename(columns={
                "Date":   "date",
                "Open":   "open",
                "High":   "high",
                "Low":    "low",
                "Close":  "close",
                "Volume": "volume",
            })
            df = df[["date", "open", "high", "low", "close", "volume"]]
            df["symbol"] = symbol
            df["date"]   = pd.to_datetime(df["date"]).dt.tz_localize(None)

        else:
            # fallback: Alpha Vantage format
            time_series = raw_data.get("Time Series (Daily)", {})
            records = [
                {
                    "symbol": symbol,
                    "date":   date,
                    "open":   float(metrics["1. open"]),
                    "high":   float(metrics["2. high"]),
                    "low":    float(metrics["3. low"]),
                    "close":  float(metrics["4. close"]),
                    "volume": int(metrics["5. volume"]),
                }
                for date, metrics in time_series.items()
            ]
            df = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])

        df = df.sort_values("date").reset_index(drop=True)
        original_len = len(df)

        # ── 1. Drop duplicates ────────────────────────────────────────
        df = df.drop_duplicates(subset=["symbol", "date"])
        dupes = original_len - len(df)
        if dupes:
            logger.warning(f"[{symbol}] Dropped {dupes} duplicate rows")

        # ── 2. Drop nulls ─────────────────────────────────────────────
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])

        # ── 3. Drop zero/negative prices ─────────────────────────────
        price_cols = ["open", "high", "low", "close"]
        invalid_price = (df[price_cols] <= 0).any(axis=1)
        if invalid_price.sum():
            logger.warning(f"[{symbol}] Dropping {invalid_price.sum()} zero/negative price rows")
            df = df[~invalid_price]

        # ── 4. Drop zero volume ───────────────────────────────────────
        zero_vol = df["volume"] == 0
        if zero_vol.sum():
            logger.warning(f"[{symbol}] Dropping {zero_vol.sum()} zero-volume rows")
            df = df[~zero_vol]

        # ── 5. OHLC sanity check ──────────────────────────────────────
        bad_candles = (
            (df["high"] < df["low"]) |
            (df["high"] < df["open"]) |
            (df["high"] < df["close"]) |
            (df["low"]  > df["open"]) |
            (df["low"]  > df["close"])
        )
        if bad_candles.sum():
            logger.warning(f"[{symbol}] Dropping {bad_candles.sum()} bad OHLC rows")
            df = df[~bad_candles]

        # ── 6. Outlier detection ──────────────────────────────────────
        df["prev_close"] = df["close"].shift(1)
        df["pct_change"] = (df["close"] - df["prev_close"]) / df["prev_close"] * 100
        outliers = df[df["pct_change"].abs() > 20]
        if not outliers.empty:
            logger.warning(
                f"[{symbol}] Suspicious moves (>20%) on: "
                f"{outliers['date'].dt.strftime('%Y-%m-%d').tolist()}"
            )

        # ── 7. Derived columns ────────────────────────────────────────
        df["daily_return"] = df["pct_change"].round(4)
        df["price_range"]  = (df["high"] - df["low"]).round(4)
        df["vwap_approx"]  = ((df["high"] + df["low"] + df["close"]) / 3).round(4)

        # ── 8. Cleanup ────────────────────────────────────────────────
        df = df.drop(columns=["prev_close", "pct_change"])
        df = df.dropna().reset_index(drop=True)

        logger.info(f"[{symbol}] Transform complete: {len(df)} clean rows")
        return df