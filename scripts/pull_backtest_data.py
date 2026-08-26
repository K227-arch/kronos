import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

mt5.initialize(r"C:\Program Files\MetaTrader 5\terminal64.exe")
mt5.login(111645481, password="@kFm4zZv", server="MetaQuotes-Demo")

# Pull GBPUSD 15m from Jan 2025 to Feb 2026
start = datetime(2025, 1, 1)
end = datetime(2026, 2, 28)

rates = mt5.copy_rates_range("GBPUSD", mt5.TIMEFRAME_M15, start, end)
print(f"Bars received: {len(rates)}")

df = pd.DataFrame(rates)
df["time"] = pd.to_datetime(df["time"], unit="s")
df = df.rename(columns={"time": "date", "tick_volume": "volume"})
df = df[["date", "open", "high", "low", "close", "volume"]]

print(f"Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")
print(df.head(3))
print(df.tail(3))

# Save
output_path = r"C:\Users\HP OMEN\Desktop\projects\trade\Kronos\data\GBPUSD_15m_backtest.csv"
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} rows to {output_path}")

mt5.shutdown()
