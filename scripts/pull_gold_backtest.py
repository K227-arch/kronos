import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

mt5.initialize(r"C:\Program Files\MetaTrader 5\terminal64.exe")
mt5.login(111645481, password="@kFm4zZv", server="MetaQuotes-Demo")

rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15, datetime(2025, 1, 1), datetime(2026, 2, 28))
df = pd.DataFrame(rates)
df["date"] = pd.to_datetime(df["time"], unit="s")
df = df.rename(columns={"tick_volume": "volume"})[["date", "open", "high", "low", "close", "volume"]]

out = r"C:\Users\HP OMEN\Desktop\projects\trade\Kronos\data\XAUUSD_15m_backtest.csv"
df.to_csv(out, index=False)
print(f"XAUUSD backtest: {len(df)} bars")
print(f"From: {df['date'].iloc[0]}")
print(f"To:   {df['date'].iloc[-1]}")
mt5.shutdown()
