import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

logger = logging.getLogger(__name__)

class EDA:
    def __init__(self, output_dir: str = "ml/eda_output"):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def run(self, df: pd.DataFrame, feature_cols: list[str]):
        self._class_balance(df)
        self._outliers(df, feature_cols)
        self._correlation_matrix(df, feature_cols)

    def _class_balance(self, df: pd.DataFrame):
        counts = df["target"].value_counts()
        pct    = df["target"].value_counts(normalize=True) * 100
        logger.info(f"Class balance:\n  UP (1):   {counts[1]} ({pct[1]:.1f}%)\n  DOWN (0): {counts[0]} ({pct[0]:.1f}%)")

        if abs(pct[1] - pct[0]) > 10:
            logger.warning("Class imbalance detected (>10% skew) — consider class_weight='balanced'")

    def _outliers(self, df: pd.DataFrame, feature_cols: list[str]):
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        axes = axes.flatten()

        for i, col in enumerate(feature_cols[:16]):
            axes[i].boxplot(df[col].dropna())
            axes[i].set_title(col, fontsize=9)
            axes[i].tick_params(axis='x', labelbottom=False)

            # IQR outlier count
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            n_outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
            if n_outliers > 0:
                logger.info(f"  {col}: {n_outliers} outliers (IQR method)")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle("Feature Outlier Analysis", fontsize=14)
        plt.tight_layout()
        path = f"{self.output_dir}/outliers.png"
        plt.savefig(path)
        plt.close()
        logger.info(f"Outlier plot saved → {path}")

    def _correlation_matrix(self, df: pd.DataFrame, feature_cols: list[str]):
        corr_cols = feature_cols + ["target"]
        corr = df[corr_cols].corr()

        plt.figure(figsize=(18, 14))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, linewidths=0.5,
            annot_kws={"size": 7}
        )
        plt.title("Feature Correlation Matrix", fontsize=14)
        plt.tight_layout()
        path = f"{self.output_dir}/correlation.png"
        plt.savefig(path)
        plt.close()
        logger.info(f"Correlation matrix saved → {path}")

        # Log top correlations with target
        target_corr = corr["target"].drop("target").abs().sort_values(ascending=False)
        logger.info(f"Top 5 features correlated with target:\n{target_corr.head()}")