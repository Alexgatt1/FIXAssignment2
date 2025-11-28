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

exchange_models: Dict[str, Pipeline] = _load_exchange_models()


def _build_feature_row(
    side: str,
    quantity: int,
    limit_price: float,
    bid_price: float,
    ask_price: float,
    bid_size: int,
    ask_size: int,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame of model features from raw order + NBBO data.

    Args:
        side: 'B' for buy, 'S' for sell.
        quantity: Order quantity.
        limit_price: Order limit price.
        bid_price: Current best bid price.
        ask_price: Current best ask price.
        bid_size: Current best bid size.
        ask_size: Current best ask size.

    Returns:
        A pandas DataFrame with one row and columns matching model_feature_columns.
    """
    side_flag = 1 if side.upper() == "B" else 0

    feature_row = pd.DataFrame(
        {
            "side_is_buy": [side_flag],
            "orderqty": [quantity],
            "limitprice": [limit_price],
            "bid_price": [bid_price],
            "ask_price": [ask_price],
            "bid_size": [bid_size],
            "ask_size": [ask_size],
        }
    )

    feature_row = feature_row[model_feature_columns]

    return feature_row

def best_price_improvement(
    symbol: str,
    side: str,
    quantity: int,
    limit_price: float,
    bid_price: float,
    ask_price: float,
    bid_size: int,
    ask_size: int,
) -> Tuple[str, float]:
    """
    Predict the exchange with the best expected price improvement for a new order.

    This function evaluates the trained price improvement model for each exchange
    and returns the exchange with the highest predicted price improvement.

    Raises RuntimeError: If no exchange models are loaded or no prediction can be made.
    """
    if not exchange_models:
        raise RuntimeError("No exchange models are loaded. Train models before routing.")

    feature_row = _build_feature_row(
        side=side,
        quantity=quantity,
        limit_price=limit_price,
        bid_price=bid_price,
        ask_price=ask_price,
        bid_size=bid_size,
        ask_size=ask_size,
    )

    best_exchange: str | None = None
    best_predicted_improvement: float | None = None

    for exchange, model in exchange_models.items():
        predicted_array = model.predict(feature_row)
        predicted_improvement = float(predicted_array[0])

        if (best_predicted_improvement is None) or (
            predicted_improvement > best_predicted_improvement
        ):
            best_predicted_improvement = predicted_improvement
            best_exchange = exchange

    if best_exchange is None or best_predicted_improvement is None:
        raise RuntimeError("Failed to compute best price improvement across exchanges.")

    return best_exchange, best_predicted_improvement