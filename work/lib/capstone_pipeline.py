"""Reusable helpers for the FlyRank Search Intelligence capstone.

Public-safe by design: model features use only pre-decision aggregates and never raw
client names, domains, URLs, titles, queries, or credentials.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import duckdb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

FEATURES = [
    "impressions_feature15",
    "clicks_feature15",
    "ctr_feature15",
    "avg_position_feature15",
    "active_impression_days_feature15",
]
LABEL = "is_declining_proxy"
CONTEXT = ["client_hash_id", "content_hash_id"]

MARCH_PATH = (
    "hf://datasets/FlyRank/internship-warehouse/"
    "fact_content_daily_performance/month=2026-03/*.parquet"
)


def connect_warehouse(hf_token: str) -> duckdb.DuckDBPyConnection:
    """Create an authenticated in-memory DuckDB connection."""
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required. Store it in Colab Secrets; never paste it into a notebook.")
    con = duckdb.connect()
    safe_token = hf_token.replace("'", "''")
    con.execute(
        f"""
        CREATE OR REPLACE SECRET hf_token (
            TYPE huggingface,
            TOKEN '{safe_token}'
        )
        """
    )
    return con


def build_analysis_frame(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Build the Week-3/4 March frame with a strict past->future timeline.

    Feature window: March 1-15, 2026.
    Outcome window: March 16-30, 2026.
    One row: one pseudonymized client-content item with >=100 feature-window impressions.
    """
    march = f"read_parquet('{MARCH_PATH}')"
    return con.sql(
        f"""
        WITH page_windows AS (
            SELECT
                client_hash_id,
                content_hash_id,
                SUM(CASE WHEN report_date BETWEEN DATE '2026-03-01' AND DATE '2026-03-15'
                         THEN COALESCE(gsc_impressions, 0) ELSE 0 END) AS impressions_feature15,
                SUM(CASE WHEN report_date BETWEEN DATE '2026-03-01' AND DATE '2026-03-15'
                         THEN COALESCE(gsc_clicks, 0) ELSE 0 END) AS clicks_feature15,
                SUM(CASE WHEN report_date BETWEEN DATE '2026-03-01' AND DATE '2026-03-15'
                              AND gsc_avg_position > 0
                         THEN gsc_avg_position * COALESCE(gsc_impressions, 0) ELSE 0 END)
                / NULLIF(
                    SUM(CASE WHEN report_date BETWEEN DATE '2026-03-01' AND DATE '2026-03-15'
                                  AND gsc_avg_position > 0
                             THEN COALESCE(gsc_impressions, 0) ELSE 0 END),
                    0
                ) AS avg_position_feature15,
                COUNT(DISTINCT CASE
                    WHEN report_date BETWEEN DATE '2026-03-01' AND DATE '2026-03-15'
                         AND COALESCE(gsc_impressions, 0) > 0
                    THEN report_date END) AS active_impression_days_feature15,
                SUM(CASE WHEN report_date BETWEEN DATE '2026-03-16' AND DATE '2026-03-30'
                         THEN COALESCE(gsc_impressions, 0) ELSE 0 END) AS impressions_outcome15
            FROM {march}
            GROUP BY client_hash_id, content_hash_id
        )
        SELECT
            *,
            100.0 * clicks_feature15 / NULLIF(impressions_feature15, 0) AS ctr_feature15,
            CASE WHEN impressions_outcome15 < 0.80 * impressions_feature15
                 THEN 1 ELSE 0 END AS is_declining_proxy
        FROM page_windows
        WHERE impressions_feature15 >= 100
        """
    ).df()


def add_position_tier(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["position_tier"] = pd.cut(
        out["avg_position_feature15"],
        bins=[0, 3, 10, 20, np.inf],
        labels=["position_1_3", "position_4_10", "position_11_20", "position_21_plus"],
        include_lowest=True,
    )
    return out


def apply_baseline(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the frozen Week-4 rule using train-only tier medians."""
    train = add_position_tier(train_df)
    out = add_position_tier(eval_df)
    medians = (
        train[train["avg_position_feature15"].between(1, 20)]
        .groupby("position_tier", observed=True)["ctr_feature15"]
        .median()
        .to_dict()
    )
    out["expected_ctr_feature15"] = out["position_tier"].map(medians).astype(float)
    out["ctr_gap_severity"] = (
        1 - out["ctr_feature15"] / out["expected_ctr_feature15"].replace(0, np.nan)
    ).clip(lower=0)
    out["rule_flag"] = (
        (out["impressions_feature15"] >= 500)
        & out["avg_position_feature15"].between(1, 20)
        & (out["ctr_gap_severity"] >= 0.20)
    )
    out["baseline_action_score"] = np.where(
        out["rule_flag"],
        out["impressions_feature15"] * out["ctr_gap_severity"],
        0.0,
    )
    out["reason_code"] = np.where(out["rule_flag"], "visible_low_ctr_for_position", "")
    out["action_label"] = np.where(out["rule_flag"], "review_title_meta_and_intent", "monitor")
    return out


def grouped_split(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, df[LABEL], groups=df["client_hash_id"]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def random_split(
    df: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        random_state=random_state,
        stratify=df[LABEL],
    )
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def model_library(random_state: int = 42) -> Dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeClassifier(max_depth=4, min_samples_leaf=100, random_state=random_state)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=250,
                max_depth=8,
                min_samples_leaf=25,
                n_jobs=-1,
                random_state=random_state,
            )),
        ]),
    }


def precision_at_k(y_true: Iterable[int], scores: Iterable[float], k: int) -> float:
    tmp = pd.DataFrame({"target": np.asarray(y_true), "score": np.asarray(scores)})
    if len(tmp) == 0:
        return float("nan")
    top = tmp.nlargest(min(k, len(tmp)), "score")
    return float(top["target"].mean())


def score_row(name: str, y_true: pd.Series, scores: np.ndarray) -> dict:
    return {
        "method": name,
        "precision_at_10": precision_at_k(y_true, scores, 10),
        "precision_at_20": precision_at_k(y_true, scores, 20),
        "precision_at_50": precision_at_k(y_true, scores, 50),
        "average_precision": float(average_precision_score(y_true, scores)),
    }


def train_compare(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int = 42,
):
    """Train readable models and compare them with the frozen baseline on one test frame."""
    X_train = train_df[FEATURES]
    y_train = train_df[LABEL]
    X_test = test_df[FEATURES]
    y_test = test_df[LABEL]

    baseline_test = apply_baseline(train_df, test_df)
    rows = [score_row("Week-4 baseline", y_test, baseline_test["baseline_action_score"].to_numpy())]
    fitted = {}
    scores = {"Week-4 baseline": baseline_test["baseline_action_score"].to_numpy()}

    for name, pipeline in model_library(random_state).items():
        pipeline.fit(X_train, y_train)
        prob = pipeline.predict_proba(X_test)[:, 1]
        fitted[name] = pipeline
        scores[name] = prob
        rows.append(score_row(name, y_test, prob))

    comparison = pd.DataFrame(rows).sort_values(
        ["precision_at_50", "precision_at_20", "average_precision"],
        ascending=False,
    ).reset_index(drop=True)
    return comparison, fitted, scores, baseline_test


def best_model_name(comparison: pd.DataFrame) -> str:
    learned = comparison[comparison["method"] != "Week-4 baseline"].copy()
    return str(learned.iloc[0]["method"])


def feature_importance(pipeline: Pipeline, feature_names=FEATURES) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        raise TypeError("Model does not expose coefficients or feature importances.")
    return (
        pd.DataFrame({"feature": list(feature_names), "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def ensure_output_dir(repo_root: str | Path = "/content/Internship") -> Path:
    path = Path(repo_root) / "work" / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path
