"""
Prepare K-line CSV data for Kronos tokenizer fine-tuning.
Converts raw OHLCV CSV to the format expected by CustomKlineDataset:
  - 'timestamps' column (datetime)
  - feature columns: open, high, low, close, volume, amount
"""
import pandas as pd
import os
import argparse


def prepare(input_path: str, output_path: str):
    df = pd.read_csv(input_path)

    # Normalize the timestamp column name to 'timestamps'
    if "timestamps" in df.columns:
        pass
    elif "date" in df.columns:
        df = df.rename(columns={"date": "timestamps"})
    elif "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "timestamps"})
    else:
        raise ValueError(f"No timestamp column found. Columns: {list(df.columns)}")

    # Parse to datetime, strip timezone
    df["timestamps"] = pd.to_datetime(df["timestamps"], utc=True).dt.tz_localize(None)

    # Required price columns
    required = ["open", "high", "low", "close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Volume (fill with 0 if absent)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    # Amount = volume * typical price (if absent)
    if "amount" not in df.columns:
        df["amount"] = df["volume"] * df[["open", "high", "low", "close"]].mean(axis=1)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Keep only needed columns, drop NaNs, sort
    out = df[["timestamps", "open", "high", "low", "close", "volume", "amount"]].copy()
    out = out.dropna().sort_values("timestamps").reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Prepared {len(out)} rows -> {output_path}")
    print(f"Time range: {out['timestamps'].iloc[0]} to {out['timestamps'].iloc[-1]}")
    print(out.head(3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw OHLCV CSV path")
    parser.add_argument("--output", required=True, help="Output prepared CSV path")
    args = parser.parse_args()
    prepare(args.input, args.output)
