from datetime import time

import numpy as np
import pandas as pd


def main() -> None:
    """Create a feature-engineered executions_with_nbbo.parquet file."""

    executions_path = "/opt/assignment3/executions.csv"
    quotes_path = "/opt/assignment4/quotes_2025-09-10_small.csv.gz"

    executions_df = pd.read_csv(
        executions_path,
        dtype={"Side": "category"},
        nrows=100_000,
    )

    quotes_df = pd.read_csv(
        quotes_path,
        dtype={"ticker": "category"},
        nrows=100_000,
    )

    executions_df["order_time"] = pd.to_datetime(
        executions_df["OrderTransactTime"],
        format="%Y%m%d-%H:%M:%S.%f",
    )
    executions_df["execution_time"] = pd.to_datetime(
        executions_df["ExecutionTransactTime"],
        format="%Y%m%d-%H:%M:%S.%f",
    )

    quotes_df["quote_time"] = pd.to_datetime(
        quotes_df["sip_timestamp"],
        unit="ns",
    )
    quotes_df = quotes_df.rename(columns={"ticker": "Symbol"})

    executions_df["Symbol"] = executions_df["Symbol"].astype(str)
    quotes_df["Symbol"] = quotes_df["Symbol"].astype(str)

    market_open = time(9, 30)
    market_close = time(16, 0)

    executions_market_mask = (
        (executions_df["order_time"].dt.time >= market_open)
        & (executions_df["order_time"].dt.time <= market_close)
    )
    executions_df = executions_df[executions_market_mask]

    executions_df = executions_df.sort_values(
        ["Symbol", "order_time"],
    ).reset_index(drop=True)

    quotes_df = quotes_df.sort_values(
        ["Symbol", "quote_time"],
    ).reset_index(drop=True)

    quote_columns = [
        "Symbol",
        "quote_time",
        "ask_price",
        "bid_price",
        "bid_size",
        "ask_size",
    ]

    merged_df = pd.merge_asof(
        executions_df,
        quotes_df[quote_columns],
        by="Symbol",
        left_on="order_time",
        right_on="quote_time",
        direction="backward",
    )

    buy_mask = merged_df["Side"] == "1"
    merged_df["price_improvement"] = np.where(
        buy_mask,
        merged_df["ask_price"] - merged_df["AvgPx"],
        merged_df["AvgPx"] - merged_df["bid_price"],
    )

    print("Rows in merged_df before dropna:", len(merged_df))

    merged_df = merged_df.dropna(
        subset=[
            "bid_price",
            "ask_price",
            "bid_size",
            "ask_size",
            "price_improvement",
            "LastMkt",
        ]
    )

    print("Rows in merged_df after dropna:", len(merged_df))
    print(merged_df.head())

    merged_df.to_parquet("executions_with_nbbo.parquet")


if __name__ == "__main__":
    main()
