"""
churn_inference.py
==================
Inference helper for the e-commerce churn model exported by export_model.py.

    from churn_inference import predict_single_customer
    result = predict_single_customer(customer_dict)

Returns a dict:
    {
      "churn_probability_pct": 63.33,              # 0..100
      "top_risk_drivers": [
          {"feature": "Complain", "value": 1, "impact_pct_points": 9.41},
          ...
      ]
    }

How the risk drivers are computed
---------------------------------
For every tree of the RandomForest the prediction path is walked from root to leaf. Each split
moves the churn probability of the current node to the probability of the child node; that
difference is credited to the feature used in the split (Saabas / "tree interpreter" method).
Averaged over all trees, the contributions plus the bias (base rate) add up exactly to the
predicted churn probability. Positive contributions push the customer TOWARDS churn, so the
three largest positive ones are the Top-3 Risk Drivers.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DEFAULT_MODEL_PATH = MODELS_DIR / "churn_model.joblib"
TOP_K = 3


# --------------------------------------------------------------------------- #
# Model loading (cached; tree arrays are pre-extracted once for fast inference)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=4)
def _load_bundle(model_path: str) -> dict:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path.resolve()} (run export_model.py first)")

    bundle = joblib.load(path)
    model = bundle["model"]
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("The model does not contain the positive class 1 (churn).")
    positive_index = classes.index(1)

    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        values = tree.value[:, 0, :].astype(float)
        probabilities = values / values.sum(axis=1, keepdims=True)
        trees.append(
            {
                "left": tree.children_left,
                "right": tree.children_right,
                "feature": tree.feature,
                "threshold": tree.threshold,
                "p_churn": probabilities[:, positive_index],
            }
        )

    bundle["_trees"] = trees
    bundle["_feature_index"] = {name: i for i, name in enumerate(bundle["feature_columns"])}
    return bundle


# --------------------------------------------------------------------------- #
# Input cleaning helpers
# --------------------------------------------------------------------------- #
def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _to_float(name: str, value: Any, median: float) -> float:
    if _is_missing(value):
        return float(median)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Field '{name}' must be numeric, got {value!r}.") from exc
    if not np.isfinite(number):
        return float(median)
    return number


def _clean_category(name: str, value: Any, prep: dict) -> str:
    if _is_missing(value):
        raise ValueError(f"Categorical field '{name}' is required.")
    label = str(value).strip()
    label = prep["category_maps"].get(name, {}).get(label, label)
    allowed = prep["categorical_levels"][name]
    if label not in allowed:
        raise ValueError(f"Unknown value {value!r} for '{name}'. Allowed values: {allowed}")
    return label


# --------------------------------------------------------------------------- #
# Preprocessing of ONE raw customer (mirrors export_model.fit_preprocessing)
# --------------------------------------------------------------------------- #
def _build_feature_vector(customer: Mapping[str, Any], bundle: dict) -> np.ndarray:
    prep = bundle["preprocessing"]
    feature_index: Dict[str, int] = bundle["_feature_index"]
    vector = np.zeros(len(feature_index), dtype=np.float64)

    # 1) numeric columns: median imputation -> IQR capping
    numeric: Dict[str, float] = {}
    for col in prep["numeric_raw_columns"]:
        numeric[col] = _to_float(col, customer.get(col), prep["medians"][col])
    for col in prep["continuous_cols"]:
        lower, upper = prep["iqr_bounds"][col]
        numeric[col] = min(max(numeric[col], lower), upper)

    # 2) engineered feature
    if numeric["OrderCount"] <= 0:
        raise ValueError("OrderCount must be greater than 0 (avg_cashbk_per_order divides by it).")
    numeric[prep["engineered_col"]] = numeric["CashbackAmount"] / numeric["OrderCount"]

    for col, value in numeric.items():
        if col in feature_index:
            vector[feature_index[col]] = value

    # 3) one-hot (drop_first=True -> the dropped baseline level is simply all zeros)
    for col in prep["categorical_cols"]:
        level = _clean_category(col, customer.get(col), prep)
        dummy_name = f"{col}_{level}"
        if dummy_name in feature_index:
            vector[feature_index[dummy_name]] = 1

    return vector


# --------------------------------------------------------------------------- #
# Per-customer explanation (Saabas contributions for a RandomForest)
# --------------------------------------------------------------------------- #
def _explain(vector: np.ndarray, bundle: dict) -> Tuple[float, float, np.ndarray]:
    """Returns (bias, probability, contributions) - all in probability units (0..1)."""
    x32 = vector.astype(np.float32)  # scikit-learn trees compare float32 values
    trees = bundle["_trees"]
    contributions = np.zeros(vector.shape[0], dtype=np.float64)
    bias = 0.0
    probability = 0.0

    for tree in trees:
        left, right = tree["left"], tree["right"]
        feature, threshold, p_churn = tree["feature"], tree["threshold"], tree["p_churn"]

        node = 0
        bias += p_churn[0]
        while left[node] != -1:
            split_feature = feature[node]
            child = left[node] if x32[split_feature] <= threshold[node] else right[node]
            contributions[split_feature] += p_churn[child] - p_churn[node]
            node = child
        probability += p_churn[node]

    n_trees = len(trees)
    return bias / n_trees, probability / n_trees, contributions / n_trees


def _describe(feature_name: str, value: float, bundle: dict) -> Tuple[str, Any]:
    dummy = bundle["preprocessing"]["dummy_columns"].get(feature_name)
    if dummy is not None:
        column, level = dummy
        label = f"{column} = {level}" if value >= 0.5 else f"{column} != {level}"
        return label, int(round(value))
    rounded = round(float(value), 4)
    return feature_name, int(rounded) if float(rounded).is_integer() else rounded


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def predict_single_customer(
    customer_dict: Mapping[str, Any],
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> Dict[str, Any]:
    """
    Predict churn probability (0-100 %) and the Top-3 risk drivers for one RAW customer record.

    Parameters
    ----------
    customer_dict : mapping with the raw dataset columns (CustomerID / Churn are ignored).
        Missing or NaN numeric values are imputed with the training medians.
        Categorical fields are required and may use the raw labels (e.g. "CC", "COD", "Mobile").
    model_path : path to churn_model.joblib (defaults to the file next to this module).
    """
    if not isinstance(customer_dict, Mapping):
        raise TypeError("customer_dict must be a dict-like object with raw customer fields.")

    bundle = _load_bundle(str(model_path))
    vector = _build_feature_vector(customer_dict, bundle)

    _, probability, contributions = _explain(vector, bundle)

    # Cross-check with scikit-learn itself (guards against silent drift between both code paths).
    frame = pd.DataFrame([vector], columns=bundle["feature_columns"])
    sk_probability = float(bundle["model"].predict_proba(frame)[0, list(bundle["model"].classes_).index(1)])
    if abs(sk_probability - probability) > 1e-6:
        raise RuntimeError("Explanation and model probability disagree; check model/inference versions.")

    order = np.argsort(contributions)[::-1]
    drivers: List[Dict[str, Any]] = []
    for idx in order:
        if len(drivers) == TOP_K or contributions[idx] <= 0:
            break
        name = bundle["feature_columns"][idx]
        label, shown_value = _describe(name, vector[idx], bundle)
        drivers.append(
            {
                "feature": label,
                "value": shown_value,
                "impact_pct_points": round(float(contributions[idx]) * 100.0, 2),
            }
        )

    return {
        "churn_probability_pct": round(sk_probability * 100.0, 2),
        "top_risk_drivers": drivers,
    }


if __name__ == "__main__":
    import json
    import sys

    stream_file = DATA_DIR / "test_stream_data.csv"
    if not stream_file.exists():
        sys.exit("test_stream_data.csv not found - run export_model.py first.")
    sample = pd.read_csv(stream_file).iloc[0].to_dict()
    print("Actual churn label:", sample.get("Churn"))
    print(json.dumps(predict_single_customer(sample), indent=2, ensure_ascii=False))
