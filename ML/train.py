import os
import pickle
import logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# ── Model definitions + hyperparameter grids ─────────────────────────────────
MODELS = {
    "logistic_regression": {
        "model": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "params": {
            "classifier__C":        [0.01, 0.1, 1, 10, 100],
            "classifier__penalty":  ["l1", "l2"],
            "classifier__solver":   ["liblinear", "saga"],
        }
    },
    "random_forest": {
        "model": RandomForestClassifier(class_weight="balanced", random_state=42),
        "params": {
            "classifier__n_estimators":      [100, 200, 300],
            "classifier__max_depth":         [5, 10, 20, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf":  [1, 2, 4],
            "classifier__max_features":      ["sqrt", "log2"],
        }
    },
    "xgboost": {
        "model": XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=1,  # adjust if class imbalance
        ),
        "params": {
            "classifier__n_estimators":  [100, 200, 300],
            "classifier__max_depth":     [3, 5, 7],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__subsample":     [0.7, 0.8, 1.0],
            "classifier__colsample_bytree": [0.7, 0.8, 1.0],
            "classifier__reg_alpha":     [0, 0.1, 0.5],
            "classifier__reg_lambda":    [1, 1.5, 2],
        }
    }
}


class ModelTrainer:
    def __init__(
        self,
        feature_cols: list[str],
        model_dir:    str = "models",
        experiment:   str = "stock_movement_prediction",
        n_iter:       int = 30,
        cv_splits:    int = 5,
    ):
        self.feature_cols = feature_cols
        self.model_dir    = model_dir
        self.experiment   = experiment
        self.n_iter       = n_iter
        self.cv_splits    = cv_splits
        os.makedirs(model_dir, exist_ok=True)
        mlflow.set_experiment(experiment)

    def _time_split(self, df: pd.DataFrame):
        """Temporal split — never shuffle time series data."""
        split_idx = int(len(df) * 0.8)
        train = df.iloc[:split_idx]
        test  = df.iloc[split_idx:]
        logger.info(f"Train: {len(train)} rows | Test: {len(test)} rows")
        logger.info(f"Train period: {train['date'].min().date()} → {train['date'].max().date()}")
        logger.info(f"Test period:  {test['date'].min().date()}  → {test['date'].max().date()}")
        return train, test

    def _metrics(self, y_true, y_pred, y_prob) -> dict:
        return {
            "accuracy":  round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
            "roc_auc":   round(roc_auc_score(y_true, y_prob), 4),
        }

    def _train_one(self, name: str, config: dict, X_train, y_train, X_test, y_test):
        logger.info(f"\n{'='*50}\nTraining: {name}\n{'='*50}")

        # Build sklearn Pipeline: scaler → classifier
        pipe = Pipeline([
            ("scaler",     StandardScaler()),
            ("classifier", config["model"]),
        ])

        # TimeSeriesSplit — respects temporal order, no data leakage
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)

        search = RandomizedSearchCV(
            estimator  = pipe,
            param_distributions = config["params"],
            n_iter     = self.n_iter,
            cv         = tscv,
            scoring    = "roc_auc",
            n_jobs     = -1,
            random_state = 42,
            verbose    = 1,
        )

        with mlflow.start_run(run_name=name):
            # Fit
            search.fit(X_train, y_train)
            best_model = search.best_estimator_

            # Predict
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]

            # Metrics
            metrics = self._metrics(y_test, y_pred, y_prob)
            logger.info(f"[{name}] Metrics: {metrics}")
            logger.info(f"\n{classification_report(y_test, y_pred)}")

            # Log to MLflow
            mlflow.log_params(search.best_params_)
            mlflow.log_metrics(metrics)
            mlflow.log_metric("cv_best_roc_auc", round(search.best_score_, 4))
            mlflow.sklearn.log_model(best_model, artifact_path=name)

            # Feature importance (RF and XGBoost only)
            clf = best_model.named_steps["classifier"]
            if hasattr(clf, "feature_importances_"):
                importances = pd.Series(clf.feature_importances_, index=self.feature_cols)
                top = importances.sort_values(ascending=False).head(10)
                logger.info(f"Top 10 features:\n{top}")
                mlflow.log_dict(top.to_dict(), "feature_importance.json")

            run_id = mlflow.active_run().info.run_id

        return {
            "name":    name,
            "model":   best_model,
            "metrics": metrics,
            "run_id":  run_id,
        }

    def train_all(self, df: pd.DataFrame) -> dict:
        train_df, test_df = self._time_split(df)

        X_train = train_df[self.feature_cols]
        y_train = train_df["target"]
        X_test  = test_df[self.feature_cols]
        y_test  = test_df["target"]

        results = []
        for name, config in MODELS.items():
            result = self._train_one(name, config, X_train, y_train, X_test, y_test)
            results.append(result)

        # ── Select best model by ROC-AUC ─────────────────────────────
        best = max(results, key=lambda r: r["metrics"]["roc_auc"])
        logger.info(f"\n{'='*50}")
        logger.info(f"BEST MODEL: {best['name']}")
        logger.info(f"ROC-AUC:    {best['metrics']['roc_auc']}")
        logger.info(f"MLflow Run: {best['run_id']}")
        logger.info(f"{'='*50}")

        # ── Save best model as pkl ────────────────────────────────────
        pkl_path = os.path.join(self.model_dir, "best_model.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump({
                "model":        best["model"],
                "feature_cols": self.feature_cols,
                "model_name":   best["name"],
                "metrics":      best["metrics"],
                "run_id":       best["run_id"],
            }, f)
        logger.info(f"Best model saved → {pkl_path}")

        # ── Summary table ─────────────────────────────────────────────
        summary = pd.DataFrame([r["metrics"] | {"model": r["name"]} for r in results])
        summary = summary.set_index("model").sort_values("roc_auc", ascending=False)
        logger.info(f"\nModel Comparison:\n{summary.to_string()}")

        return {"best": best, "all_results": results, "summary": summary}