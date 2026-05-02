import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ── Lag features ──────────────────────────────────────────────
        df["prev_close"]   = df["close"].shift(1)
        df["prev_return"]  = df["daily_return"].shift(1)
        df["prev_volume"]  = df["volume"].shift(1)
        df["prev_range"]   = df["price_range"].shift(1)

        # ── Moving averages ───────────────────────────────────────────
        df["ma_5"]   = df["close"].rolling(window=5).mean()
        df["ma_10"]  = df["close"].rolling(window=10).mean()
        df["ma_20"]  = df["close"].rolling(window=20).mean()

        # ── Price vs MA signals ───────────────────────────────────────
        df["close_vs_ma5"]  = (df["close"] - df["ma_5"])  / df["ma_5"]  * 100
        df["close_vs_ma10"] = (df["close"] - df["ma_10"]) / df["ma_10"] * 100
        df["close_vs_ma20"] = (df["close"] - df["ma_20"]) / df["ma_20"] * 100

        # ── Volatility features ───────────────────────────────────────
        df["volatility_5"]  = df["daily_return"].rolling(window=5).std()
        df["volatility_10"] = df["daily_return"].rolling(window=10).std()

        # ── Volume momentum ───────────────────────────────────────────
        df["volume_ma_5"]   = df["volume"].rolling(window=5).mean()
        df["volume_ratio"]  = df["volume"] / df["volume_ma_5"]  # >1 = high volume day

        # ── Target: 1 if next day close > today close ─────────────────
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

        # Drop rows with NaN from rolling/shift operations
        df = df.dropna().reset_index(drop=True)

        logger.info(f"Feature engineering done: {len(df)} rows, {len(df.columns)} columns")
        return df

    @property
    def feature_cols(self) -> list[str]:
        return [
            # Raw OHLCV
            "open", "high", "low", "close", "volume",
            # Precomputed in transform
            "daily_return", "price_range", "vwap_approx",
            # Lag features
            "prev_close", "prev_return", "prev_volume", "prev_range",
            # Moving averages
            "ma_5", "ma_10", "ma_20",
            # MA signals
            "close_vs_ma5", "close_vs_ma10", "close_vs_ma20",
            # Volatility
            "volatility_5", "volatility_10",
            # Volume
            "volume_ma_5", "volume_ratio",
        ]