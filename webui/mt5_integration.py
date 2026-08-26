"""
MetaTrader 5 Integration for Kronos
- Live OHLCV data feed
- Trade execution (buy/sell)
- Account info and position management
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading

# MT5 Configuration
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
MT5_LOGIN = 111645481
MT5_PASSWORD = "@kFm4zZv"
MT5_SERVER = "MetaQuotes-Demo"

# Timeframe mapping
TIMEFRAME_MAP = {
    '1m': mt5.TIMEFRAME_M1,
    '5m': mt5.TIMEFRAME_M5,
    '15m': mt5.TIMEFRAME_M15,
    '30m': mt5.TIMEFRAME_M30,
    '1h': mt5.TIMEFRAME_H1,
    '4h': mt5.TIMEFRAME_H4,
    '1d': mt5.TIMEFRAME_D1,
}


class MT5Connection:
    """Manages MT5 connection state"""

    def __init__(self):
        self.connected = False
        self.account_info = None
        self.auto_trade_active = False
        self.auto_trade_thread = None
        self.trade_history = []

    def connect(self):
        """Initialize MT5 and login"""
        if not mt5.initialize(MT5_PATH):
            error = mt5.last_error()
            return False, f"MT5 init failed: {error}"

        if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
            error = mt5.last_error()
            mt5.shutdown()
            return False, f"Login failed: {error}"

        self.connected = True
        self.account_info = self._get_account_info()
        return True, "Connected successfully"

    def disconnect(self):
        """Shutdown MT5 connection"""
        self.auto_trade_active = False
        mt5.shutdown()
        self.connected = False
        self.account_info = None
        return True, "Disconnected"

    def _get_account_info(self):
        """Get account details"""
        info = mt5.account_info()
        if info is None:
            return None
        return {
            'name': info.name,
            'login': info.login,
            'server': info.server,
            'balance': info.balance,
            'equity': info.equity,
            'profit': info.profit,
            'margin': info.margin,
            'margin_free': info.margin_free,
            'leverage': info.leverage,
            'currency': info.currency
        }

    def get_account_info(self):
        """Public method to refresh and return account info"""
        if not self.connected:
            return None
        self.account_info = self._get_account_info()
        return self.account_info

    def get_symbols(self):
        """Get available trading symbols"""
        if not self.connected:
            return []
        symbols = mt5.symbols_get()
        if symbols is None:
            return []
        # Return popular forex pairs first
        forex_pairs = []
        for s in symbols:
            if s.visible and s.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL:
                forex_pairs.append({
                    'name': s.name,
                    'description': s.description,
                    'bid': s.bid,
                    'ask': s.ask,
                    'spread': s.spread,
                    'digits': s.digits,
                    'volume_min': s.volume_min,
                    'volume_max': s.volume_max,
                    'volume_step': s.volume_step
                })
        # Sort: major forex pairs first
        majors = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD', 'XAUUSD']
        forex_pairs.sort(key=lambda x: (x['name'] not in majors, x['name']))
        return forex_pairs[:50]  # Limit to 50

    def get_live_data(self, symbol, timeframe='15m', bars=520):
        """Pull live OHLCV data from MT5"""
        if not self.connected:
            return None, "Not connected to MT5"

        tf = TIMEFRAME_MAP.get(timeframe)
        if tf is None:
            return None, f"Invalid timeframe: {timeframe}"

        # Ensure symbol is available
        if not mt5.symbol_select(symbol, True):
            return None, f"Symbol {symbol} not available"

        rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
        if rates is None or len(rates) == 0:
            return None, f"No data returned for {symbol}"

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.rename(columns={
            'time': 'date',
            'tick_volume': 'volume'
        })
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
        return df, None

    def get_positions(self):
        """Get open positions"""
        if not self.connected:
            return []
        positions = mt5.positions_get()
        if positions is None:
            return []
        result = []
        for p in positions:
            result.append({
                'ticket': p.ticket,
                'symbol': p.symbol,
                'type': 'BUY' if p.type == mt5.ORDER_TYPE_BUY else 'SELL',
                'volume': p.volume,
                'open_price': p.price_open,
                'current_price': p.price_current,
                'profit': p.profit,
                'sl': p.sl,
                'tp': p.tp,
                'time': datetime.fromtimestamp(p.time).isoformat()
            })
        return result

    def execute_trade(self, symbol, action, volume=0.01, sl_pips=50, tp_pips=100):
        """Execute a trade on MT5"""
        if not self.connected:
            return False, "Not connected to MT5"

        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return False, f"Symbol {symbol} not found"

        if not symbol_info.visible:
            mt5.symbol_select(symbol, True)

        point = symbol_info.point
        price = mt5.symbol_info_tick(symbol)

        if price is None:
            return False, "Failed to get current price"

        if action == 'BUY':
            order_type = mt5.ORDER_TYPE_BUY
            entry_price = price.ask
            sl = entry_price - sl_pips * point
            tp = entry_price + tp_pips * point
        elif action == 'SELL':
            order_type = mt5.ORDER_TYPE_SELL
            entry_price = price.bid
            sl = entry_price + sl_pips * point
            tp = entry_price - tp_pips * point
        else:
            return False, f"Invalid action: {action}. Use 'BUY' or 'SELL'"

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": entry_price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 123456,
            "comment": "Kronos AI Trade",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            return False, f"Order failed: {mt5.last_error()}"

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return False, f"Order failed: {result.comment} (code: {result.retcode})"

        trade_record = {
            'ticket': result.order,
            'symbol': symbol,
            'action': action,
            'volume': volume,
            'price': entry_price,
            'sl': sl,
            'tp': tp,
            'time': datetime.now().isoformat()
        }
        self.trade_history.append(trade_record)

        return True, f"{action} {volume} {symbol} @ {entry_price:.5f} | SL: {sl:.5f} | TP: {tp:.5f}"

    def close_position(self, ticket):
        """Close a specific position by ticket"""
        if not self.connected:
            return False, "Not connected to MT5"

        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            return False, f"Position {ticket} not found"

        pos = position[0]
        symbol = pos.symbol
        volume = pos.volume

        # Opposite order to close
        if pos.type == mt5.ORDER_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": "Kronos Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            return False, f"Close failed: {mt5.last_error()}"

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return False, f"Close failed: {result.comment} (code: {result.retcode})"

        return True, f"Position {ticket} closed @ {price:.5f}"

    def generate_signal(self, pred_df, historical_df):
        """
        Generate BUY/SELL/HOLD signal from Kronos prediction.
        Compares predicted close vs last historical close.
        """
        if pred_df is None or len(pred_df) == 0:
            return 'HOLD', 0.0

        last_close = historical_df['close'].iloc[-1]
        pred_close_mean = pred_df['close'].mean()
        pred_high_mean = pred_df['high'].mean()
        pred_low_mean = pred_df['low'].mean()

        # Calculate predicted movement percentage
        change_pct = (pred_close_mean - last_close) / last_close * 100

        # Signal thresholds
        threshold = 0.05  # 0.05% minimum movement to trigger

        if change_pct > threshold:
            return 'BUY', change_pct
        elif change_pct < -threshold:
            return 'SELL', change_pct
        else:
            return 'HOLD', change_pct


# Global MT5 connection instance
mt5_conn = MT5Connection()
