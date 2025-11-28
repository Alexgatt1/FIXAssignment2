from datetime import time

import numpy as np
import pandas as pd

executions = pd.read_csv("/opt/assignment3/executions.csv", dtype= {"Side":"category",}, nrows= 100_000)
quotes = pd.read_csv("/opt/assignment4/quotes_2025-09-10_small.csv.gz", dtype = {"ticker":"category",}, nrows=100_000)

executions["order_time"] = pd.to_datetime(executions["OrderTransactTime"], format="%Y%m%d-%H:%M:%S.%f")
executions["execution_time"] = pd.to_datetime(executions["ExecutionTransactTime"], format="%Y%m%d-%H:%M:%S.%f")
quotes["quote_time"] = pd.to_datetime(quotes["sip_timestamp"], unit="ns")
quotes = quotes.rename(columns={"ticker":"Symbol"})
executions["Symbol"] = executions["Symbol"].astype(str)
quotes["Symbol"] = quotes["Symbol"].astype(str)


market_open = time(9,30)
market_close = time(16,0)
exec_mask = ((executions["order_time"].dt.time >= market_open) & (executions["order_time"].dt.time <= market_close))
executions = executions[exec_mask]
quote_mask = ((quotes["quote_time"].dt.time >= market_open) & (quotes["quote_time"].dt.time <= market_close))
quotes = quotes[quote_mask]

executions = executions.sort_values("order_time").reset_index(drop=True)
quotes = quotes.sort_values("quote_time").reset_index(drop=True)

quote_cols = ["Symbol", "quote_time", "ask_price", "bid_price", "bid_size", "ask_size"]

merged_pd = pd.merge_asof(executions, quotes[quote_cols], by="Symbol", left_on="order_time", right_on="quote_time", direction="backward")
buy_mask = merged_pd["Side"] == '1'
merged_pd["price_improvement"] = np.where(buy_mask, merged_pd["ask_price"] - merged_pd["AvgPx"], merged_pd["AvgPx"] - merged_pd["bid_price"])
merged_pd.dropna(subset=["bid_price","ask_price", "bid_size", "ask_size", "price_improvement", "LastMkt",])
print("Rows in merged_pd after dropna:", len(merged_pd))


merged_pd.to_parquet("executions_with_nbbo.parquet")



print(merged_pd.head())