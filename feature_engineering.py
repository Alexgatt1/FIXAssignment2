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
        nrows=500_000,
    )
    quotes_df = pd.read_csv(
        quotes_path,
        dtype={"ticker": "category"},
        nrows=500_000,
    )

    print("Initial executions rows:", len(executions_df))
    print("Initial quotes rows:", len(quotes_df))

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

    print("Unique execution symbols (sample):", executions_df["Symbol"].nunique())
    print("Unique quote symbols (sample):", quotes_df["Symbol"].nunique())

    exec_syms = set(executions_df["Symbol"].unique())
    quote_syms = set(quotes_df["Symbol"].unique())
    overlap = exec_syms & quote_syms
    print("Overlapping symbols count:", len(overlap))
    print("Example overlapping symbols:", list(overlap)[:10])

    market_open = time(9, 30)
    market_close = time(16, 0)

    executions_market_mask = (
        (executions_df["order_time"].dt.time >= market_open)
        & (executions_df["order_time"].dt.time <= market_close)
    )
    executions_df = executions_df[executions_market_mask]
    print("Executions rows after market-hours filter:", len(executions_df))

    if overlap:
        executions_df = executions_df[executions_df["Symbol"].isin(overlap)]
    print("Executions rows after symbol-overlap filter:", len(executions_df))

    if executions_df.empty:
        print("No executions left after filters – cannot continue.")
        return

    executions_df = executions_df.sort_values(
        ["order_time", "Symbol"],
    ).reset_index(drop=True)

    quotes_df = quotes_df.sort_values(
        ["quote_time", "Symbol"],
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

    print("Rows in merged_df before dropna:", len(merged_df))

    buy_mask = merged_df["Side"] == "1"
    merged_df["price_improvement"] = np.where(
        buy_mask,
        merged_df["ask_price"] - merged_df["AvgPx"],
        merged_df["AvgPx"] - merged_df["bid_price"],
    )

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
