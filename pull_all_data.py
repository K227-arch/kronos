"""
pull_all_data.py
Fetch all project data:
  - Chinese A-share stocks via yfinance (.SS/.SZ suffixes) — works globally
  - Gold OHLCV via yfinance (GC=F, 2016 to today)
"""

import os, time
import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# symbol -> (yfinance_ticker, name)
STOCK_SYMBOLS = [
    ("002354", "002354.SZ", "天娱数科"),
    ("600580", "600580.SS", "卧龙电驱"),
    ("300207", "300207.SZ", "欣旺达"),
    ("300418", "300418.SZ", "昆仑万维"),
    ("000021", "000021.SZ", "深科技"),
    ("603288", "603288.SS", "海天味业"),
]

RESULTS = {}


# ── Pull all Chinese stocks via yfinance ─────────────────────────────────────

def pull_all_cn_stocks():
    print("\n══════════════════════════════════════════")
    print("  Chinese A-share stocks  (via yfinance)")
    print("══════════════════════════════════════════")

    for idx, (symbol, yf_ticker, name) in enumerate(STOCK_SYMBOLS):
        print(f"\n▶  [{idx+1}/{len(STOCK_SYMBOLS)}]  {symbol}  {name}  ({yf_ticker})")
        try:
            raw = yf.Ticker(yf_ticker).history(start="2016-01-01", auto_adjust=True)
            if raw is None or raw.empty:
                raise ValueError("Empty result from yfinance")

            df = raw.reset_index()
            df = df.rename(columns={
                "Date":   "timestamps",
                "Open":   "open",
                "High":   "high",
                "Low":    "low",
                "Close":  "close",
                "Volume": "volume",
            })
            # Strip timezone from timestamps
            df["timestamps"] = pd.to_datetime(df["timestamps"]).dt.tz_localize(None).dt.normalize()
            df["stock_code"] = symbol

            keep = [c for c in ["timestamps","open","high","low","close","volume","stock_code"] if c in df.columns]
            df = df[keep].sort_values("timestamps").reset_index(drop=True)
            df[["open","high","low","close"]] = df[["open","high","low","close"]].round(4)

            out = os.path.join(DATA_DIR, f"{symbol}_daily.csv")
            df.to_csv(out, index=False, encoding="utf-8-sig")

            start = df["timestamps"].min().strftime("%Y-%m-%d")
            end   = df["timestamps"].max().strftime("%Y-%m-%d")
            print(f"  ✓  {len(df):,} rows  {start} → {end}  →  {out}")
            RESULTS[symbol] = {"status": "ok", "rows": len(df), "start": start, "end": end, "file": out}

        except Exception as e:
            print(f"  ✗  {symbol} failed: {e}")
            RESULTS[symbol] = {"status": "failed", "rows": 0, "error": str(e)}

        time.sleep(1)   # light pause between tickers


# ── Gold OHLCV ───────────────────────────────────────────────────────────────

def pull_gold():
    print("\n══════════════════════════════════════════")
    print("  Gold daily OHLCV  (GC=F via yfinance)")
    print("══════════════════════════════════════════")
    try:
        df = yf.Ticker("GC=F").history(start="2016-01-01", auto_adjust=True)
        if df is None or df.empty:
            raise ValueError("yfinance returned empty dataframe")
        df = df.reset_index()
        df = df.rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume"
        })
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        df = df[["date","open","high","low","close","volume"]].sort_values("date").reset_index(drop=True)
        df[["open","high","low","close"]] = df[["open","high","low","close"]].round(2)
        out = os.path.join(DATA_DIR, "gold_daily_ohlcv.csv")
        df.to_csv(out, index=False)
        start = df["date"].min().strftime("%Y-%m-%d")
        end   = df["date"].max().strftime("%Y-%m-%d")
        print(f"  ✓  {len(df):,} rows  {start} → {end}")
        print(f"     → {out}")
        RESULTS["gold"] = {"status": "ok", "rows": len(df), "start": start, "end": end, "file": out}
    except Exception as e:
        print(f"  ✗  Gold fetch failed: {e}")
        RESULTS["gold"] = {"status": "failed", "error": str(e)}


# ── Summary ──────────────────────────────────────────────────────────────────

def print_summary():
    print("\n══════════════════════════════════════════")
    print("  SUMMARY")
    print("══════════════════════════════════════════")
    ok   = [(k, v) for k, v in RESULTS.items() if v["status"] == "ok"]
    fail = [(k, v) for k, v in RESULTS.items() if v["status"] != "ok"]
    for k, v in ok:
        print(f"  ✓  {k:<12}  {v['rows']:>6,} rows   {v['start']} → {v['end']}")
    for k, v in fail:
        print(f"  ✗  {k:<12}  FAILED")
    print(f"\n  {len(ok)}/{len(RESULTS)} succeeded")
    print(f"  Files saved to: {DATA_DIR}")


if __name__ == "__main__":
    pull_all_cn_stocks()
    pull_gold()
    print_summary()
