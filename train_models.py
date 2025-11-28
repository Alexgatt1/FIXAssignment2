from __future__ import annotations

from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

feature_columns: List[str] = [
    "side_is_buy",
    "orderqty",
    "limitprice",
    "bid_price",
    "ask_price",
    "bid_size",
    "ask_size",
]

target_column = "price_improvement"
exchange_column = "lastmkt"
min_rows_per_exchange = 200

def load_training_dataframe(path: str = "executions_with_nbbo.parquet") -> pd.DataFrame:
    """
    Load the engineered dataset and prepare it for model training.

    Converts Side ('1' = buy) into a numeric side_is_buy feature and drops
    rows missing any required fields.
    """
    dataframe = pd.read_parquet(path)


def load_training_dataframe(path: str = "executions_with_nbbo.parquet") -> pd.DataFrame:
    """
    Load the engineered dataset and prepare it for model training.

    Converts Side ('1' = buy) into a numeric side_is_buy feature and drops
    rows missing any required fields.
    """
    dataframe = pd.read_parquet(path)
    dataframe.columns = [col.lower() for col in dataframe.columns]
    dataframe["side"] = dataframe["side"].astype(str)
    dataframe["side_is_buy"] = (dataframe["side"] == "1").astype(int)
    dataframe = dataframe.dropna(
        subset=feature_columns + [target_column, exchange_column]
    )

    return dataframe
print("This is working")
