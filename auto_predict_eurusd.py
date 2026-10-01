"""
auto_predict_eurusd.py
─────────────────────────────────────────────────────────────────────────────
Automatically runs Kronos prediction on EURUSD_15min.csv using optimal
settings for this dataset:

  Model    : Kronos-mini  (2048-bar context — best for 15-min intraday data)
  Lookback : 2000 bars    (≈ 3 weeks of 15-min candles)
  Pred len : 96 bars      (= 24 hours ahead)
  Temp     : 0.8          (conservative; forex is mean-reverting)
  top_p    : 0.85
  Samples  : 3            (averaged for smoother output)

Output
  • PNG chart saved to  webui/prediction_results/
  • JSON results saved to webui/prediction_results/
─────────────────────────────────────────────────────────────────────────────
"""

import os, sys, json, warnings, datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Config ────────────────────────────────────────────────────────────────────

DATA_FILE   = os.path.join(os.path.dirname(__file__), "data", "EURUSD_15min.csv")
OUT_DIR     = os.path.join(os.path.dirname(__file__), "webui", "prediction_results")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_ID      = "NeoQuasar/Kronos-mini"
TOKENIZER_ID  = "NeoQuasar/Kronos-Tokenizer-2k"
MAX_CONTEXT   = 2048

LOOKBACK      = 2000   # bars of history fed to the model
PRED_LEN      = 96     # bars to predict (96 × 15min = 24 h)
TEMPERATURE   = 0.8    # lower = more conservative / mean-reverting
TOP_P         = 0.85
SAMPLE_COUNT  = 3      # average N independent draws

# ── Load data ─────────────────────────────────────────────────────────────────

print("=" * 60)
print("  Kronos EURUSD 15-min Auto-Prediction")
print("=" * 60)
print(f"\n[1/5] Loading data: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)
df = df.rename(columns={"date": "timestamps"})
df["timestamps"] = pd.to_datetime(df["timestamps"])
df = df.sort_values("timestamps").reset_index(drop=True)

for col in ["open", "high", "low", "close", "volume"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=["open", "high", "low", "close"])

print(f"    {len(df):,} bars loaded  |  "
      f"{df['timestamps'].min().strftime('%Y-%m-%d')} → "
      f"{df['timestamps'].max().strftime('%Y-%m-%d')}")
print(f"    15-min bars  |  Price range: "
      f"{df['close'].min():.5f} – {df['close'].max():.5f}")

if len(df) < LOOKBACK + PRED_LEN:
    sys.exit(f"Not enough data: need {LOOKBACK + PRED_LEN}, got {len(df)}")

# Use the MOST RECENT lookback window as input
x_df    = df.iloc[-(LOOKBACK + PRED_LEN):-PRED_LEN].copy()
# Ground-truth for the prediction window (for comparison chart)
actual_df = df.iloc[-PRED_LEN:].copy()

x_timestamp = x_df["timestamps"].reset_index(drop=True)
y_timestamp = actual_df["timestamps"].reset_index(drop=True)

print(f"    Lookback window : {x_timestamp.iloc[0]} → {x_timestamp.iloc[-1]}")
print(f"    Predict window  : {y_timestamp.iloc[0]} → {y_timestamp.iloc[-1]}")

# ── Load model ────────────────────────────────────────────────────────────────

print(f"\n[2/5] Loading model: {MODEL_ID}")

from model import Kronos, KronosTokenizer, KronosPredictor
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"    Device: {device}")

tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_ID)
model     = Kronos.from_pretrained(MODEL_ID)
predictor = KronosPredictor(model, tokenizer, device=device, max_context=MAX_CONTEXT)

print(f"    Model loaded  ({sum(p.numel() for p in model.parameters()):,} params)")

# ── Predict ───────────────────────────────────────────────────────────────────

print(f"\n[3/5] Running prediction ...")
print(f"    lookback={LOOKBACK}  pred_len={PRED_LEN}  "
      f"T={TEMPERATURE}  top_p={TOP_P}  samples={SAMPLE_COUNT}")

feat_df = x_df[["open", "high", "low", "close", "volume"]].reset_index(drop=True)

pred_df = predictor.predict(
    df           = feat_df,
    x_timestamp  = x_timestamp,
    y_timestamp  = y_timestamp,
    pred_len     = PRED_LEN,
    T            = TEMPERATURE,
    top_p        = TOP_P,
    sample_count = SAMPLE_COUNT,
    verbose      = True,
)

print(f"\n    Prediction complete — {len(pred_df)} bars")

# ── Chart ─────────────────────────────────────────────────────────────────────

print(f"\n[4/5] Generating chart ...")

# Show last 200 bars of history + predictions + actuals
hist_plot = x_df.tail(200).copy()

fig, axes = plt.subplots(2, 1, figsize=(18, 10),
                         gridspec_kw={"height_ratios": [3, 1]},
                         sharex=False)
fig.patch.set_facecolor("#0d1117")
for ax in axes:
    ax.set_facecolor("#161b22")

ax1, ax2 = axes

# ── Price panel ──────────────────────────────────────────────────────────────
ax1.plot(hist_plot["timestamps"], hist_plot["close"],
         color="#58a6ff", lw=1.2, label="Historical (close)")

ax1.plot(pred_df.index, pred_df["close"],
         color="#f78166", lw=1.8, linestyle="--", label=f"Predicted close (next {PRED_LEN} bars)")

ax1.plot(actual_df["timestamps"], actual_df["close"],
         color="#3fb950", lw=1.2, alpha=0.7, label="Actual close (ground truth)")

# Prediction band: use high/low
ax1.fill_between(pred_df.index, pred_df["low"], pred_df["high"],
                 color="#f78166", alpha=0.12, label="Pred high/low band")

# Divider
div_x = x_timestamp.iloc[-1]
ax1.axvline(div_x, color="#e3b341", lw=1.2, linestyle=":", alpha=0.8)
ax1.annotate("↑ prediction start", xy=(div_x, hist_plot["close"].iloc[-1]),
             xytext=(8, 8), textcoords="offset points",
             color="#e3b341", fontsize=8)

ax1.set_ylabel("Price (EUR/USD)", color="#c9d1d9", fontsize=11)
ax1.tick_params(colors="#c9d1d9")
ax1.spines[:].set_color("#30363d")
ax1.legend(loc="upper left", facecolor="#161b22", labelcolor="#c9d1d9",
           edgecolor="#30363d", fontsize=9)
ax1.set_title(
    f"EURUSD 15-min  |  Kronos-mini  |  Lookback {LOOKBACK} bars → Predict {PRED_LEN} bars (24 h)\n"
    f"T={TEMPERATURE}  top_p={TOP_P}  samples={SAMPLE_COUNT}  |  "
    f"Pred start: {div_x.strftime('%Y-%m-%d %H:%M')}",
    color="#c9d1d9", fontsize=11, pad=10
)
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, color="#c9d1d9")

# ── Volume panel ─────────────────────────────────────────────────────────────
ax2.bar(hist_plot["timestamps"], hist_plot["volume"],
        color="#58a6ff", alpha=0.5, width=0.009, label="Hist volume")
ax2.bar(pred_df.index, pred_df["volume"],
        color="#f78166", alpha=0.5, width=0.009, label="Pred volume")
ax2.axvline(div_x, color="#e3b341", lw=1.0, linestyle=":", alpha=0.7)
ax2.set_ylabel("Volume", color="#c9d1d9", fontsize=10)
ax2.tick_params(colors="#c9d1d9")
ax2.spines[:].set_color("#30363d")
ax2.legend(loc="upper left", facecolor="#161b22", labelcolor="#c9d1d9",
           edgecolor="#30363d", fontsize=8)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, color="#c9d1d9")

plt.tight_layout(pad=1.5)

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
chart_path = os.path.join(OUT_DIR, f"EURUSD_15min_prediction_{ts}.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"    Chart saved → {chart_path}")

# ── Save JSON ─────────────────────────────────────────────────────────────────

print(f"\n[5/5] Saving JSON results ...")

# Error metrics vs actual
mae_close  = float(np.mean(np.abs(pred_df["close"].values - actual_df["close"].values)))
rmse_close = float(np.sqrt(np.mean((pred_df["close"].values - actual_df["close"].values)**2)))
last_actual_close = float(actual_df["close"].iloc[-1])
last_pred_close   = float(pred_df["close"].iloc[-1])
direction_actual  = 1 if last_actual_close > float(actual_df["close"].iloc[0]) else -1
direction_pred    = 1 if last_pred_close   > float(pred_df["close"].iloc[0])   else -1

print(f"\n    ── Accuracy metrics ──────────────────")
print(f"    MAE  (close): {mae_close:.6f}")
print(f"    RMSE (close): {rmse_close:.6f}")
print(f"    Direction match: {'✓ YES' if direction_actual == direction_pred else '✗ NO'}")
print(f"    Actual end : {last_actual_close:.5f}")
print(f"    Pred end   : {last_pred_close:.5f}")

prediction_records = []
for i, ts_val in enumerate(pred_df.index):
    prediction_records.append({
        "timestamp":    str(ts_val),
        "open":         round(float(pred_df["open"].iloc[i]),  5),
        "high":         round(float(pred_df["high"].iloc[i]),  5),
        "low":          round(float(pred_df["low"].iloc[i]),   5),
        "close":        round(float(pred_df["close"].iloc[i]), 5),
        "volume":       round(float(pred_df["volume"].iloc[i]), 2),
        "actual_close": round(float(actual_df["close"].iloc[i]), 5),
    })

results = {
    "run_timestamp": datetime.datetime.now().isoformat(),
    "dataset":       "EURUSD_15min",
    "model":         MODEL_ID,
    "tokenizer":     TOKENIZER_ID,
    "params": {
        "lookback":     LOOKBACK,
        "pred_len":     PRED_LEN,
        "temperature":  TEMPERATURE,
        "top_p":        TOP_P,
        "sample_count": SAMPLE_COUNT,
        "device":       device,
    },
    "data_range": {
        "lookback_start": str(x_timestamp.iloc[0]),
        "lookback_end":   str(x_timestamp.iloc[-1]),
        "pred_start":     str(y_timestamp.iloc[0]),
        "pred_end":       str(y_timestamp.iloc[-1]),
    },
    "metrics": {
        "mae_close":       mae_close,
        "rmse_close":      rmse_close,
        "direction_match": direction_actual == direction_pred,
        "actual_end_price": last_actual_close,
        "pred_end_price":   last_pred_close,
    },
    "chart_path":   chart_path,
    "predictions":  prediction_records,
}

json_ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
json_path  = os.path.join(OUT_DIR, f"EURUSD_15min_prediction_{json_ts}.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"    JSON saved  → {json_path}")

print("\n" + "=" * 60)
print("  DONE")
print(f"  Chart : {chart_path}")
print(f"  JSON  : {json_path}")
print("=" * 60)
