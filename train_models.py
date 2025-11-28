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
    dataframe.columns = [col.lower() for col in dataframe.columns]
    dataframe["side"] = dataframe["side"].astype(str)
    dataframe["side_is_buy"] = (dataframe["side"] == "1").astype(int)
    dataframe = dataframe.dropna(
        subset=feature_columns + [target_column, exchange_column]
    )

    return dataframe

def build_grid_search_model() -> GridSearchCV:
    """
    Create a RandomForest regression pipeline wrapped in GridSearchCV.

    The grid search explores a small parameter space and uses n_jobs=-1
    for full parallelism.
    """
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
        ]
    )

    param_grid = {
        "model__n_estimators": [50, 100],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_leaf": [1, 5],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    return grid_search

def train_models_for_all_exchanges(dataframe: pd.DataFrame) -> Dict[str, Pipeline]:
    """
    Train individual models for each exchange (lastmkt).

    Exchanges with fewer than min_rows_per_exchange rows are skipped.
    """
    models: Dict[str, Pipeline] = {}

    for exchange, group in dataframe.groupby(exchange_column):
        if len(group) < min_rows_per_exchange:
            print(f"Skipping exchange {exchange}: only {len(group)} rows.")
            continue

        features = group[feature_columns]
        target = group[target_column]

        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )

        print(f"\nTraining model for exchange {exchange} on {len(group)} rows...")
        grid_search = build_grid_search_model()
        grid_search.fit(x_train, y_train)

        best_model: Pipeline = grid_search.best_estimator_

        predictions = best_model.predict(x_test)
        r_squared = r2_score(y_test, predictions)
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))

        print(f"Exchange {exchange}: R^2={r_squared:.4f}, RMSE={rmse:.6f}")
        print(f"Best parameters: {grid_search.best_params_}")

        models[exchange] = best_model
    
    return models

def main() -> None:
    """Train per-exchange regression models and save them to disk."""
    dataframe = load_training_dataframe()
    print(f"Loaded {len(dataframe)} rows for training.")

    models = train_models_for_all_exchanges(dataframe)

    if not models:
        print("No models were trained — check data availability.")
        return

    joblib.dump(models, "exchange_price_improvement_models.joblib")
    print(f"Saved {len(models)} models.")


if __name__ == "__main__":
    main()