"""
Kronos Backtest Engine
- Walks forward through historical data
- Uses Kronos model to predict at each step
- Executes real trades on MT5
- Reports results in real-time via a shared state dict
"""

import pandas as pd
import numpy as np
import threading
import time
from datetime import datetime
import MetaTrader5 as mt5


class BacktestEngine:
    """Walk-forward backtest that executes real trades on MT5"""

    def __init__(self, predictor, mt5_conn):
        self.predictor = predictor
        self.mt5_conn = mt5_conn
        self.running = False
        self.thread = None

        # Backtest parameters
        self.lookback = 400
        self.pred_len = 120
        self.step_size = 120  # Move forward by pred_len after each prediction
        self.temperature = 0.8
        self.top_p = 0.85
        self.sample_count = 3
        self.volume = 0.01
        self.sl_pips = 50
        self.tp_pips = 100
        self.signal_threshold = 0.03  # 0.03% to trigger trade

        # State (shared with API for real-time updates)
        self.state = {
            "status": "idle",  # idle, running, completed, stopped
            "progress": 0,
            "total_steps": 0,
            "current_step": 0,
            "current_date": "",
            "trades": [],
            "signals": [],
            "equity_curve": [],
            "stats": {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "total_profit": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "starting_balance": 0.0,
                "current_balance": 0.0,
            },
            "positions": [],
            "error": None,
        }

    def start(self, data_path, symbol="GBPUSD"):
        """Start backtest in a background thread"""
        if self.running:
            return False, "Backtest already running"

        self.running = True
        self.state["status"] = "running"
        self.state["error"] = None
        self.state["trades"] = []
        self.state["signals"] = []
        self.state["equity_curve"] = []

        self.thread = threading.Thread(
            target=self._run_backtest, args=(data_path, symbol), daemon=True
        )
        self.thread.start()
        return True, "Backtest started"

    def stop(self):
        """Stop the backtest"""
        self.running = False
        self.state["status"] = "stopped"
        return True, "Backtest stopping..."

    def _run_backtest(self, data_path, symbol):
        """Main backtest loop"""
        try:
            # Load data
            df = pd.read_csv(data_path)
            df["date"] = pd.to_datetime(df["date"])
            df["timestamps"] = df["date"]

            total_bars = len(df)
            total_steps = (total_bars - self.lookback - self.pred_len) // self.step_size
            self.state["total_steps"] = total_steps

            # Get starting balance
            account = self.mt5_conn.get_account_info()
            if account:
                self.state["stats"]["starting_balance"] = account["balance"]
                self.state["stats"]["current_balance"] = account["balance"]
                self.state["equity_curve"].append({
                    "step": 0,
                    "date": str(df["date"].iloc[self.lookback]),
                    "equity": account["balance"],
                })

            peak_equity = self.state["stats"]["starting_balance"]
            step = 0

            while self.running and step < total_steps:
                start_idx = step * self.step_size
                end_idx = start_idx + self.lookback

                # Check bounds
                if end_idx + self.pred_len > total_bars:
                    break

                # Get historical window
                hist_df = df.iloc[start_idx:end_idx]
                current_date = str(hist_df["timestamps"].iloc[-1])
                self.state["current_date"] = current_date
                self.state["current_step"] = step + 1
                self.state["progress"] = int((step + 1) / total_steps * 100)

                # Prepare prediction inputs
                required_cols = ["open", "high", "low", "close"]
                if "volume" in df.columns:
                    required_cols.append("volume")

                x_df = hist_df[required_cols].copy()
                x_timestamp = pd.Series(hist_df["timestamps"].values, name="timestamps")

                # Future timestamps
                future_df = df.iloc[end_idx:end_idx + self.pred_len]
                y_timestamp = pd.Series(future_df["timestamps"].values, name="timestamps")

                # Run prediction
                try:
                    pred_df = self.predictor.predict(
                        df=x_df,
                        x_timestamp=x_timestamp,
                        y_timestamp=y_timestamp,
                        pred_len=self.pred_len,
                        T=self.temperature,
                        top_p=self.top_p,
                        sample_count=self.sample_count,
                        verbose=False,
                    )
                except Exception as e:
                    self.state["signals"].append({
                        "step": step,
                        "date": current_date,
                        "signal": "ERROR",
                        "error": str(e),
                    })
                    step += 1
                    continue

                # Generate signal
                last_close = float(hist_df["close"].iloc[-1])
                pred_close_mean = float(pred_df["close"].mean())
                change_pct = (pred_close_mean - last_close) / last_close * 100

                if change_pct > self.signal_threshold:
                    signal = "BUY"
                elif change_pct < -self.signal_threshold:
                    signal = "SELL"
                else:
                    signal = "HOLD"

                self.state["signals"].append({
                    "step": step,
                    "date": current_date,
                    "signal": signal,
                    "change_pct": round(float(change_pct), 4),
                    "last_close": round(float(last_close), 5),
                    "pred_close": round(float(pred_close_mean), 5),
                })

                # Close existing positions before opening new ones
                positions = self.mt5_conn.get_positions()
                for pos in positions:
                    if pos["symbol"] == symbol:
                        self.mt5_conn.close_position(pos["ticket"])
                        # Record closed trade
                        self.state["trades"].append({
                            "step": step,
                            "date": current_date,
                            "action": "CLOSE",
                            "symbol": symbol,
                            "ticket": pos["ticket"],
                            "profit": pos["profit"],
                        })
                        if pos["profit"] >= 0:
                            self.state["stats"]["winning_trades"] += 1
                        else:
                            self.state["stats"]["losing_trades"] += 1
                        self.state["stats"]["total_profit"] += pos["profit"]

                # Execute new trade
                if signal != "HOLD":
                    success, message = self.mt5_conn.execute_trade(
                        symbol, signal, self.volume, self.sl_pips, self.tp_pips
                    )
                    self.state["trades"].append({
                        "step": step,
                        "date": current_date,
                        "action": signal,
                        "symbol": symbol,
                        "success": success,
                        "message": message,
                    })
                    if success:
                        self.state["stats"]["total_trades"] += 1

                # Update equity curve
                account = self.mt5_conn.get_account_info()
                if account:
                    current_equity = account["equity"]
                    self.state["stats"]["current_balance"] = account["balance"]
                    self.state["equity_curve"].append({
                        "step": step + 1,
                        "date": current_date,
                        "equity": current_equity,
                    })

                    # Max drawdown
                    if current_equity > peak_equity:
                        peak_equity = current_equity
                    drawdown = (peak_equity - current_equity) / peak_equity * 100
                    if drawdown > self.state["stats"]["max_drawdown"]:
                        self.state["stats"]["max_drawdown"] = round(drawdown, 2)

                # Update positions
                self.state["positions"] = self.mt5_conn.get_positions()

                # Win rate
                total_closed = (
                    self.state["stats"]["winning_trades"]
                    + self.state["stats"]["losing_trades"]
                )
                if total_closed > 0:
                    self.state["stats"]["win_rate"] = round(
                        self.state["stats"]["winning_trades"] / total_closed * 100, 1
                    )

                step += 1

                # Small delay to not overwhelm MT5
                time.sleep(0.5)

            # Close remaining positions at end
            positions = self.mt5_conn.get_positions()
            for pos in positions:
                if pos["symbol"] == symbol:
                    self.mt5_conn.close_position(pos["ticket"])
                    self.state["trades"].append({
                        "step": step,
                        "date": self.state["current_date"],
                        "action": "CLOSE_FINAL",
                        "symbol": symbol,
                        "ticket": pos["ticket"],
                        "profit": pos["profit"],
                    })
                    if pos["profit"] >= 0:
                        self.state["stats"]["winning_trades"] += 1
                    else:
                        self.state["stats"]["losing_trades"] += 1
                    self.state["stats"]["total_profit"] += pos["profit"]

            # Final stats
            total_closed = (
                self.state["stats"]["winning_trades"]
                + self.state["stats"]["losing_trades"]
            )
            if total_closed > 0:
                self.state["stats"]["win_rate"] = round(
                    self.state["stats"]["winning_trades"] / total_closed * 100, 1
                )

            account = self.mt5_conn.get_account_info()
            if account:
                self.state["stats"]["current_balance"] = account["balance"]

            self.state["stats"]["total_profit"] = round(
                self.state["stats"]["total_profit"], 2
            )

            if self.running:
                self.state["status"] = "completed"
            self.state["progress"] = 100
            self.running = False

        except Exception as e:
            self.state["status"] = "error"
            self.state["error"] = str(e)
            self.running = False


# Global backtest instance
backtest_engine = None
