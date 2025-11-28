from __future__ import annotations

from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


model_feature_columns = [
    "side_is_buy",
    "orderqty",
    "limitprice",
    "bid_price",
    "ask_price",
    "bid_size",
    "ask_size",
]

def _load_exchange_models(
    path: str = "exchange_price_improvement_models.joblib",
) -> Dict[str, Pipeline]:
    """
    Load the per-exchange price improvement models from disk.

    The file is expected to contain a dictionary mapping exchange identifiers
    (e.g. 'XNAS', 'XNYS') to trained sklearn Pipelines.
    """
    models: Dict[str, Pipeline] = joblib.load(path)
    return models
