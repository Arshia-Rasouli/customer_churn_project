"""
export_model.py
================
Reproduces the preprocessing + training pipeline of learn.ipynb exactly and exports:

  1. churn_model.joblib      -> trained RandomForest + everything needed for inference
  2. test_stream_data.csv    -> 200 raw test rows (CustomerID + raw features + true Churn label)

Usage:
    python export_model.py                                   # reads E-Commerce-Dataset-2.csv
    python export_model.py --data "E Commerce Dataset.xlsx"  # .xlsx also supported (sheet "E Comm")

Dependencies: pandas, numpy, scikit-learn, joblib (an .xlsx input additionally needs openpyxl).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------- #
# Constants (taken 1:1 from the notebook)
# --------------------------------------------------------------------------- #
TARGET_COL = "Churn"
ID_COL = "CustomerID"
EXCEL_SHEET = "E Comm"

CATEGORICAL_COLS: List[str] = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
]

# Duplicate-label merges (notebook section "Cleaning Categorical Features").
CATEGORY_MAPS: Dict[str, Dict[str, str]] = {
    "PreferedOrderCat": {"Mobile": "Mobile Phone"},
    "PreferredPaymentMode": {"CC": "Credit Card", "COD": "Cash on Delivery"},
}

# Columns capped with the IQR method (notebook section "Outlier Detection & Capping").
CONTINUOUS_COLS: List[str] = [
    "Tenure",
    "WarehouseToHome",
    "HourSpendOnApp",
    "NumberOfAddress",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
]

ENGINEERED_COL = "avg_cashbk_per_order"

TEST_SIZE = 0.20
RANDOM_STATE = 42
STREAM_ROWS = 200

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "churn_model.joblib"
STREAM_PATH = DATA_DIR / "test_stream_data.csv"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_raw_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=EXCEL_SHEET)
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# Preprocessing (fit on the training pipeline, exactly like the notebook)
# --------------------------------------------------------------------------- #
def fit_preprocessing(raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """
    Replays the notebook steps in the SAME order and returns (X, y, params).

    Order in the notebook:
      1. drop CustomerID
      2. median imputation for every column with missing values
      3. merge duplicate categories
      4. IQR capping (Q1 - 1.5*IQR, Q3 + 1.5*IQR) on CONTINUOUS_COLS
      5. avg_cashbk_per_order = CashbackAmount / OrderCount
      6. One-Hot Encoding (drop_first=True, dtype=int)
    """
    required = {ID_COL, TARGET_COL, *CATEGORICAL_COLS, *CONTINUOUS_COLS}
    missing_cols = sorted(required - set(raw.columns))
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    data = raw.drop(columns=[ID_COL]).copy()
    feature_cols_raw = [c for c in data.columns if c != TARGET_COL]

    # 2) median imputation (medians stored so inference can reuse them)
    numeric_raw_cols = [c for c in feature_cols_raw if c not in CATEGORICAL_COLS]
    medians: Dict[str, float] = {}
    for col in numeric_raw_cols:
        medians[col] = float(data[col].median())
        if data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(medians[col])

    # 3) merge duplicate categories
    for col, mapping in CATEGORY_MAPS.items():
        data[col] = data[col].replace(mapping)

    # 4) IQR capping
    iqr_bounds: Dict[str, List[float]] = {}
    for col in CONTINUOUS_COLS:
        q1, q3 = np.percentile(data[col], [25, 75])
        iqr = q3 - q1
        lower, upper = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
        iqr_bounds[col] = [lower, upper]
        data[col] = np.where(data[col] > upper, upper, data[col])
        data[col] = np.where(data[col] < lower, lower, data[col])

    # 5) feature engineering
    data[ENGINEERED_COL] = data["CashbackAmount"] / data["OrderCount"]

    # 6) One-Hot Encoding
    categorical_levels = {
        col: sorted(data[col].dropna().unique().tolist()) for col in CATEGORICAL_COLS
    }
    data = pd.get_dummies(data, columns=CATEGORICAL_COLS, drop_first=True, dtype=int)

    X = data.drop(columns=[TARGET_COL])
    y = data[TARGET_COL]

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values found after preprocessing.")

    dummy_columns = {}
    for col in CATEGORICAL_COLS:
        for level in categorical_levels[col]:
            name = f"{col}_{level}"
            if name in X.columns:
                dummy_columns[name] = [col, level]

    params = {
        "raw_feature_columns": feature_cols_raw,
        "numeric_raw_columns": numeric_raw_cols,
        "categorical_cols": list(CATEGORICAL_COLS),
        "continuous_cols": list(CONTINUOUS_COLS),
        "medians": medians,
        "iqr_bounds": iqr_bounds,
        "category_maps": CATEGORY_MAPS,
        "categorical_levels": categorical_levels,
        "dummy_columns": dummy_columns,
        "engineered_col": ENGINEERED_COL,
    }
    return X, y, params


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Train churn model and export artifacts.")
    parser.add_argument(
    "--data",
    default=str(DATA_DIR / "E-Commerce-Dataset.csv"),
    help="CSV or XLSX dataset path",
)
    args = parser.parse_args()


    raw = load_raw_data(Path(args.data))
    print(f"Loaded dataset: {raw.shape[0]} rows x {raw.shape[1]} columns")

    X, y, params = fit_preprocessing(raw)
    print(f"Feature matrix: {X.shape}  |  churn rate: {y.mean() * 100:.2f}%")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    model = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print("\nTest-set evaluation")
    print(classification_report(y_test, y_pred, digits=3))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    importances = (
        pd.Series(model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(5)
    )
    print("Top-5 Gini importances:")
    print(importances.round(4).to_string())

    # ------------------------------------------------------------------ #
    # Save model bundle
    # ------------------------------------------------------------------ #
    bundle = {
        "model": model,
        "feature_columns": list(X_train.columns),
        "preprocessing": params,
        "metadata": {
            "sklearn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "random_state": RANDOM_STATE,
            "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        },
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / MODEL_PATH.name
    joblib.dump(bundle, model_path, compress=3)
    print(f"\nSaved model bundle -> {model_path}")

    # ------------------------------------------------------------------ #
    # Save real-time simulation stream (RAW rows, so that they can be fed
    # directly into predict_single_customer)
    # ------------------------------------------------------------------ #
    stream_idx = X_test.sample(n=min(STREAM_ROWS, len(X_test)), random_state=RANDOM_STATE).index
    stream_columns = [ID_COL] + params["raw_feature_columns"] + [TARGET_COL]
    stream_df = raw.loc[stream_idx, stream_columns].reset_index(drop=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    stream_path = DATA_DIR / STREAM_PATH.name
    stream_df.to_csv(stream_path, index=False)
    print(f"Saved {len(stream_df)} stream rows -> {stream_path}")

    # ------------------------------------------------------------------ #
    # Optional parity check against churn_inference.py (if it sits next to this file)
    # ------------------------------------------------------------------ #
    try:
        import churn_inference  # noqa: WPS433
    except ImportError:
        print("churn_inference.py not found next to export_model.py -> parity check skipped.")
        return

    probs_direct = model.predict_proba(X_test.loc[stream_idx])[:, 1] * 100.0
    probs_helper = np.array(
        [
            churn_inference.predict_single_customer(
                row.drop(labels=[ID_COL, TARGET_COL]).to_dict(), model_path=model_path
            )["churn_probability_pct"]
            for _, row in stream_df.iterrows()
        ]
    )
    max_diff = float(np.max(np.abs(probs_direct - probs_helper)))
    print(f"Parity check (helper vs. model on X_test): max |diff| = {max_diff:.4f} percentage points")
    if max_diff > 0.01:
        raise AssertionError("Inference preprocessing does not match the training pipeline!")
    print("Parity check passed.")


if __name__ == "__main__":
    main()
