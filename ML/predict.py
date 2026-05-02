import pickle
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class Predictor:
    def __init__(self, model_path: str = "models/best_model.pkl"):
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        self.model        = bundle["model"]
        self.feature_cols = bundle["feature_cols"]
        self.model_name   = bundle["model_name"]
        self.metrics      = bundle["metrics"]
        logger.info(f"Loaded model: {self.model_name} | ROC-AUC: {self.metrics['roc_auc']}")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.feature_cols]
        df = df.copy()
        df["prediction"]  = self.model.predict(X)
        df["probability"] = self.model.predict_proba(X)[:, 1]
        df["signal"]      = df["prediction"].map({1: "UP ↑", 0: "DOWN ↓"})
        return df[["date", "close", "signal", "probability"]]