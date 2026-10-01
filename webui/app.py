import os
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import warnings
import datetime
warnings.filterwarnings('ignore')

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: Kronos model cannot be imported, will use simulated data for demonstration")

try:
    from mt5_integration import mt5_conn
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("Warning: MT5 integration not available")

try:
    from textblob import TextBlob
    SENTIMENT_AVAILABLE = True
except ImportError:
    SENTIMENT_AVAILABLE = False
    print("Warning: TextBlob not available, sentiment analysis disabled")

# Global sentiment state (shared with predict endpoint)
current_sentiment = {
    'score': 0.0,      # -1.0 (very bearish) to +1.0 (very bullish)
    'signal': 'NEUTRAL',
    'bias': 0.0,       # price bias percentage applied to predictions
    'articles': 0,
    'query': ''
}

try:
    from backtest_engine import BacktestEngine
    BACKTEST_AVAILABLE = True
except ImportError:
    BACKTEST_AVAILABLE = False
    print("Warning: Backtest engine not available")

# Global backtest instance
backtest_engine = None

app = Flask(__name__)
CORS(app)

# Global variables to store models
tokenizer = None
model = None
predictor = None

# Available model configurations
AVAILABLE_MODELS = {
    'kronos-mini': {
        'name': 'Kronos-mini',
        'model_id': 'NeoQuasar/Kronos-mini',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-2k',
        'context_length': 2048,
        'params': '4.1M',
        'description': 'Lightweight model, suitable for fast prediction'
    },
    'kronos-small': {
        'name': 'Kronos-small',
        'model_id': 'NeoQuasar/Kronos-small',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '24.7M',
        'description': 'Small model, balanced performance and speed'
    },
    'kronos-base': {
        'name': 'Kronos-base',
        'model_id': 'NeoQuasar/Kronos-base',
        'tokenizer_id': 'NeoQuasar/Kronos-Tokenizer-base',
        'context_length': 512,
        'params': '102.3M',
        'description': 'Base model, provides better prediction quality'
    }
}

def load_data_files():
    """Scan data directory and return available data files"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    data_files = []
    
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(('.csv', '.feather')):
                file_path = os.path.join(data_dir, file)
                file_size = os.path.getsize(file_path)
                data_files.append({
                    'name': file,
                    'path': file_path,
                    'size': f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
                })
    
    return data_files

def load_data_file(file_path):
    """Load data file"""
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.feather'):
            df = pd.read_feather(file_path)
        else:
            return None, "Unsupported file format"
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return None, f"Missing required columns: {required_cols}"
        
        # Process timestamp column
        if 'timestamps' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamps'], utc=True).dt.tz_localize(None)
        elif 'timestamp' in df.columns:
            df['timestamps'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
        elif 'date' in df.columns:
            # If column name is 'date', rename it to 'timestamps'
            df['timestamps'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
        else:
            # If no timestamp column exists, create one
            df['timestamps'] = pd.date_range(start='2024-01-01', periods=len(df), freq='1H')
        
        # Ensure numeric columns are numeric type
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Process volume column (optional)
        if 'volume' in df.columns:
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Process amount column (optional, but not used for prediction)
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # Remove rows containing NaN values
        df = df.dropna()
        
        return df, None
        
    except Exception as e:
        return None, f"Failed to load file: {str(e)}"

def save_prediction_results(file_path, prediction_type, prediction_results, actual_data, input_data, prediction_params):
    """Save prediction results to file"""
    try:
        # Create prediction results directory
        results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'prediction_{timestamp}.json'
        filepath = os.path.join(results_dir, filename)
        
        # Prepare data for saving
        save_data = {
            'timestamp': datetime.datetime.now().isoformat(),
            'file_path': file_path,
            'prediction_type': prediction_type,
            'prediction_params': prediction_params,
            'input_data_summary': {
                'rows': len(input_data),
                'columns': list(input_data.columns),
                'price_range': {
                    'open': {'min': float(input_data['open'].min()), 'max': float(input_data['open'].max())},
                    'high': {'min': float(input_data['high'].min()), 'max': float(input_data['high'].max())},
                    'low': {'min': float(input_data['low'].min()), 'max': float(input_data['low'].max())},
                    'close': {'min': float(input_data['close'].min()), 'max': float(input_data['close'].max())}
                },
                'last_values': {
                    'open': float(input_data['open'].iloc[-1]),
                    'high': float(input_data['high'].iloc[-1]),
                    'low': float(input_data['low'].iloc[-1]),
                    'close': float(input_data['close'].iloc[-1])
                }
            },
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'analysis': {}
        }
        
        # If actual data exists, perform comparison analysis
        if actual_data and len(actual_data) > 0:
            # Calculate continuity analysis
            if len(prediction_results) > 0 and len(actual_data) > 0:
                last_pred = prediction_results[0]  # First prediction point
            first_actual = actual_data[0]      # First actual point
                
            save_data['analysis']['continuity'] = {
                    'last_prediction': {
                        'open': last_pred['open'],
                        'high': last_pred['high'],
                        'low': last_pred['low'],
                        'close': last_pred['close']
                    },
                    'first_actual': {
                        'open': first_actual['open'],
                        'high': first_actual['high'],
                        'low': first_actual['low'],
                        'close': first_actual['close']
                    },
                    'gaps': {
                        'open_gap': abs(last_pred['open'] - first_actual['open']),
                        'high_gap': abs(last_pred['high'] - first_actual['high']),
                        'low_gap': abs(last_pred['low'] - first_actual['low']),
                        'close_gap': abs(last_pred['close'] - first_actual['close'])
                    },
                    'gap_percentages': {
                        'open_gap_pct': (abs(last_pred['open'] - first_actual['open']) / first_actual['open']) * 100,
                        'high_gap_pct': (abs(last_pred['high'] - first_actual['high']) / first_actual['high']) * 100,
                        'low_gap_pct': (abs(last_pred['low'] - first_actual['low']) / first_actual['low']) * 100,
                        'close_gap_pct': (abs(last_pred['close'] - first_actual['close']) / first_actual['close']) * 100
                    }
                }
        
        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"Prediction results saved to: {filepath}")
        return filepath
        
    except Exception as e:
        print(f"Failed to save prediction results: {e}")
        return None

def create_prediction_chart(df, pred_df, lookback, pred_len, actual_df=None, historical_start_idx=0):
    """Create prediction chart"""
    # Use specified historical data start position
    if historical_start_idx + lookback + pred_len <= len(df):
        historical_df = df.iloc[historical_start_idx:historical_start_idx+lookback]
    else:
        available_lookback = min(lookback, len(df) - historical_start_idx)
        historical_df = df.iloc[historical_start_idx:historical_start_idx+available_lookback]
    
    # Build x-axis values
    has_timestamps = 'timestamps' in historical_df.columns and len(historical_df) > 0
    
    if has_timestamps:
        hist_x = historical_df['timestamps'].tolist()
        # Calculate time interval for prediction timestamps
        if len(df) > 1:
            time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
        else:
            time_diff = pd.Timedelta(minutes=1)
        last_ts = historical_df['timestamps'].iloc[-1]
        pred_x = pd.date_range(start=last_ts + time_diff, periods=len(pred_df), freq=time_diff).tolist()
    else:
        hist_x = list(range(len(historical_df)))
        pred_x = list(range(len(historical_df), len(historical_df) + len(pred_df)))

    # Create traces as plain dicts for reliability
    traces = []
    
    # Historical candlestick
    traces.append({
        'type': 'candlestick',
        'x': hist_x,
        'open': historical_df['open'].tolist(),
        'high': historical_df['high'].tolist(),
        'low': historical_df['low'].tolist(),
        'close': historical_df['close'].tolist(),
        'name': 'Historical (400 pts)',
        'increasing': {'line': {'color': '#26A69A'}},
        'decreasing': {'line': {'color': '#EF5350'}}
    })
    
    # Prediction candlestick
    if pred_df is not None and len(pred_df) > 0:
        traces.append({
            'type': 'candlestick',
            'x': pred_x,
            'open': pred_df['open'].tolist(),
            'high': pred_df['high'].tolist(),
            'low': pred_df['low'].tolist(),
            'close': pred_df['close'].tolist(),
            'name': 'Prediction (120 pts)',
            'increasing': {'line': {'color': '#66BB6A'}},
            'decreasing': {'line': {'color': '#FF7043'}}
        })
    
    # Actual data candlestick for comparison
    if actual_df is not None and len(actual_df) > 0:
        traces.append({
            'type': 'candlestick',
            'x': pred_x[:len(actual_df)],
            'open': actual_df['open'].tolist(),
            'high': actual_df['high'].tolist(),
            'low': actual_df['low'].tolist(),
            'close': actual_df['close'].tolist(),
            'name': 'Actual (120 pts)',
            'increasing': {'line': {'color': '#FF9800'}},
            'decreasing': {'line': {'color': '#F44336'}}
        })
    
    layout = {
        'title': 'Kronos Prediction Results',
        'xaxis': {
            'title': 'Time',
            'rangeslider': {'visible': False}
        },
        'yaxis': {'title': 'Price'},
        'template': 'plotly_white',
        'height': 600,
        'showlegend': True
    }
    
    chart = {'data': traces, 'layout': layout}
    return json.dumps(chart, default=str)

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/api/data-files')
def get_data_files():
    """Get available data file list"""
    data_files = load_data_files()
    return jsonify(data_files)

@app.route('/api/upload-data', methods=['POST'])
def upload_data():
    """Upload a data file from the user's browser"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Validate file extension
        allowed_extensions = ('.csv', '.feather')
        if not file.filename.lower().endswith(allowed_extensions):
            return jsonify({'error': f'Unsupported file type. Allowed: {allowed_extensions}'}), 400

        # Save to data directory
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        os.makedirs(data_dir, exist_ok=True)

        # Secure the filename
        filename = file.filename.replace('\\', '/').split('/')[-1]
        file_path = os.path.join(data_dir, filename)
        file.save(file_path)

        # Validate the file contents
        df, error = load_data_file(file_path)
        if error:
            os.remove(file_path)
            return jsonify({'error': f'Invalid file: {error}'}), 400

        file_size = os.path.getsize(file_path)
        return jsonify({
            'success': True,
            'message': f'File "{filename}" uploaded successfully',
            'file': {
                'name': filename,
                'path': file_path,
                'size': f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
            }
        })

    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/load-data', methods=['POST'])
def load_data():
    """Load data file"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400
        
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        # Detect data time frequency
        def detect_timeframe(df):
            if len(df) < 2:
                return "Unknown"
            
            time_diffs = []
            for i in range(1, min(10, len(df))):  # Check first 10 time differences
                diff = df['timestamps'].iloc[i] - df['timestamps'].iloc[i-1]
                time_diffs.append(diff)
            
            if not time_diffs:
                return "Unknown"
            
            # Calculate average time difference
            avg_diff = sum(time_diffs, pd.Timedelta(0)) / len(time_diffs)
            
            # Convert to readable format
            if avg_diff < pd.Timedelta(minutes=1):
                return f"{avg_diff.total_seconds():.0f} seconds"
            elif avg_diff < pd.Timedelta(hours=1):
                return f"{avg_diff.total_seconds() / 60:.0f} minutes"
            elif avg_diff < pd.Timedelta(days=1):
                return f"{avg_diff.total_seconds() / 3600:.0f} hours"
            else:
                return f"{avg_diff.days} days"
        
        # Return data information
        data_info = {
            'rows': len(df),
            'columns': list(df.columns),
            'start_date': df['timestamps'].min().isoformat() if 'timestamps' in df.columns else 'N/A',
            'end_date': df['timestamps'].max().isoformat() if 'timestamps' in df.columns else 'N/A',
            'price_range': {
                'min': float(df[['open', 'high', 'low', 'close']].min().min()),
                'max': float(df[['open', 'high', 'low', 'close']].max().max())
            },
            'prediction_columns': ['open', 'high', 'low', 'close'] + (['volume'] if 'volume' in df.columns else []),
            'timeframe': detect_timeframe(df)
        }
        
        return jsonify({
            'success': True,
            'data_info': data_info,
            'message': f'Successfully loaded data, total {len(df)} rows'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to load data: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Perform prediction"""
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        lookback = int(data.get('lookback', 400))
        pred_len = int(data.get('pred_len', 120))
        
        # Get prediction quality parameters
        temperature = float(data.get('temperature', 1.0))
        top_p = float(data.get('top_p', 0.9))
        sample_count = int(data.get('sample_count', 1))
        
        if not file_path:
            return jsonify({'error': 'File path cannot be empty'}), 400
        
        # Load data
        df, error = load_data_file(file_path)
        if error:
            return jsonify({'error': error}), 400
        
        if len(df) < lookback:
            return jsonify({'error': f'Insufficient data length, need at least {lookback} rows'}), 400
        
        # Perform prediction
        if MODEL_AVAILABLE and predictor is not None:
            try:
                # Use real Kronos model
                # Only use necessary columns: OHLCV, excluding amount
                required_cols = ['open', 'high', 'low', 'close']
                if 'volume' in df.columns:
                    required_cols.append('volume')
                
                # Process time period selection
                start_date = data.get('start_date')
                
                if start_date:
                    # Custom time period - fix logic: use data within selected window
                    start_dt = pd.to_datetime(start_date)
                    
                    # Find data after start time
                    mask = df['timestamps'] >= start_dt
                    time_range_df = df[mask]
                    
                    # Ensure sufficient data: lookback + pred_len
                    if len(time_range_df) < lookback + pred_len:
                        return jsonify({'error': f'Insufficient data from start time {start_dt.strftime("%Y-%m-%d %H:%M")}, need at least {lookback + pred_len} data points, currently only {len(time_range_df)} available'}), 400
                    
                    # Use first lookback data points within selected window for prediction
                    x_df = time_range_df.iloc[:lookback][required_cols]
                    x_timestamp = time_range_df.iloc[:lookback]['timestamps']
                    
                    # Use last pred_len data points within selected window as actual values
                    y_timestamp = time_range_df.iloc[lookback:lookback+pred_len]['timestamps']
                    
                    # Calculate actual time period length
                    start_timestamp = time_range_df['timestamps'].iloc[0]
                    end_timestamp = time_range_df['timestamps'].iloc[lookback+pred_len-1]
                    time_span = end_timestamp - start_timestamp
                    
                    prediction_type = f"Kronos model prediction (within selected window: first {lookback} data points for prediction, last {pred_len} data points for comparison, time span: {time_span})"
                else:
                    # Use latest data
                    x_df = df.iloc[:lookback][required_cols]
                    x_timestamp = df.iloc[:lookback]['timestamps']
                    y_timestamp = df.iloc[lookback:lookback+pred_len]['timestamps']
                    prediction_type = "Kronos model prediction (latest data)"
                
                # Ensure timestamps are Series format, not DatetimeIndex, to avoid .dt attribute error in Kronos model
                if isinstance(x_timestamp, pd.DatetimeIndex):
                    x_timestamp = pd.Series(x_timestamp, name='timestamps')
                if isinstance(y_timestamp, pd.DatetimeIndex):
                    y_timestamp = pd.Series(y_timestamp, name='timestamps')
                
                pred_df = predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamp,
                    y_timestamp=y_timestamp,
                    pred_len=pred_len,
                    T=temperature,
                    top_p=top_p,
                    sample_count=sample_count
                )

                # Apply news sentiment bias to predicted prices
                if current_sentiment['bias'] != 0.0:
                    price_cols = ['open', 'high', 'low', 'close']
                    bias_factor = 1.0 + (current_sentiment['bias'] / 100.0)
                    for col in price_cols:
                        if col in pred_df.columns:
                            pred_df[col] = pred_df[col] * bias_factor
                
            except Exception as e:
                return jsonify({'error': f'Kronos model prediction failed: {str(e)}'}), 500
        else:
            return jsonify({'error': 'Kronos model not loaded, please load model first'}), 400
        
        # Prepare actual data for comparison (if exists)
        actual_data = []
        actual_df = None
        
        if start_date:  # Custom time period
            # Fix logic: use data within selected window
            # Prediction uses first 400 data points within selected window
            # Actual data should be last 120 data points within selected window
            start_dt = pd.to_datetime(start_date)
            
            # Find data starting from start_date
            mask = df['timestamps'] >= start_dt
            time_range_df = df[mask]
            
            if len(time_range_df) >= lookback + pred_len:
                # Get last 120 data points within selected window as actual values
                actual_df = time_range_df.iloc[lookback:lookback+pred_len]
                
                for i, (_, row) in enumerate(actual_df.iterrows()):
                    actual_data.append({
                        'timestamp': row['timestamps'].isoformat(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if 'volume' in row else 0,
                        'amount': float(row['amount']) if 'amount' in row else 0
                    })
        else:  # Latest data
            # Prediction uses first 400 data points
            # Actual data should be 120 data points after first 400 data points
            if len(df) >= lookback + pred_len:
                actual_df = df.iloc[lookback:lookback+pred_len]
                for i, (_, row) in enumerate(actual_df.iterrows()):
                    actual_data.append({
                        'timestamp': row['timestamps'].isoformat(),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']) if 'volume' in row else 0,
                        'amount': float(row['amount']) if 'amount' in row else 0
                    })
        
        # Create chart - pass historical data start position
        if start_date:
            # Custom time period: find starting position of historical data in original df
            start_dt = pd.to_datetime(start_date)
            mask = df['timestamps'] >= start_dt
            historical_start_idx = df[mask].index[0] if len(df[mask]) > 0 else 0
        else:
            # Latest data: start from beginning
            historical_start_idx = 0
        
        chart_json = create_prediction_chart(df, pred_df, lookback, pred_len, actual_df, historical_start_idx)
        
        # Prepare prediction result data - fix timestamp calculation logic
        if 'timestamps' in df.columns:
            if start_date:
                # Custom time period: use selected window data to calculate timestamps
                start_dt = pd.to_datetime(start_date)
                mask = df['timestamps'] >= start_dt
                time_range_df = df[mask]
                
                if len(time_range_df) >= lookback:
                    # Calculate prediction timestamps starting from last time point of selected window
                    last_timestamp = time_range_df['timestamps'].iloc[lookback-1]
                    time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
                    future_timestamps = pd.date_range(
                        start=last_timestamp + time_diff,
                        periods=pred_len,
                        freq=time_diff
                    )
                else:
                    future_timestamps = []
            else:
                # Latest data: calculate from last time point of entire data file
                last_timestamp = df['timestamps'].iloc[-1]
                time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
                future_timestamps = pd.date_range(
                    start=last_timestamp + time_diff,
                    periods=pred_len,
                    freq=time_diff
                )
        else:
            future_timestamps = range(len(df), len(df) + pred_len)
        
        prediction_results = []
        for i, (_, row) in enumerate(pred_df.iterrows()):
            prediction_results.append({
                'timestamp': future_timestamps[i].isoformat() if i < len(future_timestamps) else f"T{i}",
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']) if 'volume' in row else 0,
                'amount': float(row['amount']) if 'amount' in row else 0
            })
        
        # Save prediction results to file
        try:
            save_prediction_results(
                file_path=file_path,
                prediction_type=prediction_type,
                prediction_results=prediction_results,
                actual_data=actual_data,
                input_data=x_df,
                prediction_params={
                    'lookback': lookback,
                    'pred_len': pred_len,
                    'temperature': temperature,
                    'top_p': top_p,
                    'sample_count': sample_count,
                    'start_date': start_date if start_date else 'latest'
                }
            )
        except Exception as e:
            print(f"Failed to save prediction results: {e}")
        
        return jsonify({
            'success': True,
            'prediction_type': prediction_type,
            'chart': chart_json,
            'prediction_results': prediction_results,
            'actual_data': actual_data,
            'has_comparison': len(actual_data) > 0,
            'sentiment': current_sentiment,
            'message': f'Prediction completed, generated {pred_len} prediction points' + (f', including {len(actual_data)} actual data points for comparison' if len(actual_data) > 0 else '') + (f' | Sentiment bias: {current_sentiment["signal"]} ({current_sentiment["bias"]:+.2f}%)' if current_sentiment["bias"] != 0.0 else '')
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/load-model', methods=['POST'])
def load_model():
    """Load Kronos model"""
    global tokenizer, model, predictor
    
    try:
        if not MODEL_AVAILABLE:
            return jsonify({'error': 'Kronos model library not available'}), 400
        
        data = request.get_json()
        model_key = data.get('model_key', 'kronos-small')
        device = data.get('device', 'cpu')
        
        if model_key not in AVAILABLE_MODELS:
            return jsonify({'error': f'Unsupported model: {model_key}'}), 400
        
        model_config = AVAILABLE_MODELS[model_key]
        
        # Load tokenizer and model
        tokenizer = KronosTokenizer.from_pretrained(model_config['tokenizer_id'])
        model = Kronos.from_pretrained(model_config['model_id'])
        
        # Create predictor
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=model_config['context_length'])
        
        return jsonify({
            'success': True,
            'message': f'Model loaded successfully: {model_config["name"]} ({model_config["params"]}) on {device}',
            'model_info': {
                'name': model_config['name'],
                'params': model_config['params'],
                'context_length': model_config['context_length'],
                'description': model_config['description']
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Model loading failed: {str(e)}'}), 500

@app.route('/api/available-models')
def get_available_models():
    """Get available model list"""
    return jsonify({
        'models': AVAILABLE_MODELS,
        'model_available': MODEL_AVAILABLE
    })

@app.route('/api/model-status')
def get_model_status():
    """Get model status"""
    if MODEL_AVAILABLE:
        if predictor is not None:
            return jsonify({
                'available': True,
                'loaded': True,
                'message': 'Kronos model loaded and available',
                'current_model': {
                    'name': predictor.model.__class__.__name__,
                    'device': str(next(predictor.model.parameters()).device)
                }
            })
        else:
            return jsonify({
                'available': True,
                'loaded': False,
                'message': 'Kronos model available but not loaded'
            })
    else:
        return jsonify({
            'available': False,
            'loaded': False,
            'message': 'Kronos model library not available, please install related dependencies'
        })

# ==================== MT5 INTEGRATION ROUTES ====================

@app.route('/api/mt5/connect', methods=['POST'])
def mt5_connect():
    """Connect to MetaTrader 5"""
    if not MT5_AVAILABLE:
        return jsonify({'error': 'MT5 integration not available'}), 400
    
    success, message = mt5_conn.connect()
    if success:
        return jsonify({
            'success': True,
            'message': message,
            'account': mt5_conn.account_info
        })
    else:
        return jsonify({'error': message}), 400

@app.route('/api/mt5/disconnect', methods=['POST'])
def mt5_disconnect():
    """Disconnect from MetaTrader 5"""
    if not MT5_AVAILABLE:
        return jsonify({'error': 'MT5 integration not available'}), 400
    
    success, message = mt5_conn.disconnect()
    return jsonify({'success': True, 'message': message})

@app.route('/api/mt5/account')
def mt5_account():
    """Get MT5 account info"""
    if not MT5_AVAILABLE:
        return jsonify({'error': 'MT5 integration not available'}), 400
    
    info = mt5_conn.get_account_info()
    if info is None:
        return jsonify({'error': 'Not connected to MT5'}), 400
    return jsonify({'success': True, 'account': info})

@app.route('/api/mt5/symbols')
def mt5_symbols():
    """Get available trading symbols"""
    if not MT5_AVAILABLE:
        return jsonify({'error': 'MT5 integration not available'}), 400
    
    symbols = mt5_conn.get_symbols()
    return jsonify({'success': True, 'symbols': symbols})

@app.route('/api/mt5/live-data', methods=['POST'])
def mt5_live_data():
    """Pull live OHLCV data from MT5"""
    if not MT5_AVAILABLE:
        return jsonify({'error': 'MT5 integration not available'}), 400
    
    data = request.get_json()
    symbol = data.get('symbol', 'EURUSD')
    timeframe = data.get('timeframe', '15m')
    bars = int(data.get('bars', 520))
    
    df, error = mt5_conn.get_live_data(symbol, timeframe, bars)
    if error:
        return jsonify({'error': error}), 400
    
    # Save to temp file for prediction pipeline
    temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'live_data_temp.csv')
    df.to_csv(temp_path, index=False)
    
    return jsonify({
        'success': True,
        'symbol': symbol,
        'timeframe': timeframe,
        'bars': len(df),
        'file_path': temp_path,
        'latest': {
            'date': str(df['date'].iloc[-1]),
            'open': float(df['open'].iloc[-1]),
            'high': float(df['high'].iloc[-1]),
            'low': float(df['low'].iloc[-1]),
            'close': float(df['close'].iloc[-1]),
            'volume': float(df['volume'].iloc[-1])
        }
    })

@app.route('/api/mt5/positions')
def mt5_positions():
    """Get open positions"""
    if not MT5_AVAILABLE:
        return jsonify({'error': 'MT5 integration not available'}), 400
    
    positions = mt5_conn.get_positions()
    return jsonify({'success': True, 'positions': positions})

@app.route('/api/mt5/trade', methods=['POST'])
def mt5_trade():
    """Execute a trade"""
    if not MT5_AVAILABLE:
        return jsonify({'error': 'MT5 integration not available'}), 400
    
    data = request.get_json()
    symbol = data.get('symbol', 'EURUSD')
    action = data.get('action', '').upper()
    volume = float(data.get('volume', 0.01))
    sl_pips = int(data.get('sl_pips', 50))
    tp_pips = int(data.get('tp_pips', 100))
    
    if action not in ('BUY', 'SELL'):
        return jsonify({'error': 'Action must be BUY or SELL'}), 400
    
    success, message = mt5_conn.execute_trade(symbol, action, volume, sl_pips, tp_pips)
    if success:
        return jsonify({'success': True, 'message': message})
    else:
        return jsonify({'error': message}), 400

@app.route('/api/mt5/close-position', methods=['POST'])
def mt5_close_position():
    """Close a position"""
    if not MT5_AVAILABLE:
        return jsonify({'error': 'MT5 integration not available'}), 400
    
    data = request.get_json()
    ticket = int(data.get('ticket', 0))
    
    success, message = mt5_conn.close_position(ticket)
    if success:
        return jsonify({'success': True, 'message': message})
    else:
        return jsonify({'error': message}), 400

@app.route('/api/integrated-predict', methods=['POST'])
def integrated_predict():
    """
    Full Gold prediction pipeline:
    1. Pull latest XAUUSD data directly from MT5
    2. Fetch Gold news sentiment (auto)
    3. Run Kronos-base prediction with sentiment bias applied
    Returns chart + prediction + sentiment + signal
    """
    global current_sentiment

    if not MODEL_AVAILABLE or predictor is None:
        return jsonify({'error': 'Kronos model not loaded. Load Kronos-base first.'}), 400

    data = request.get_json() or {}
    symbol    = data.get('symbol', 'XAUUSD')
    timeframe = data.get('timeframe', '15m')
    bars      = int(data.get('bars', 520))
    temperature  = float(data.get('temperature', 0.7))
    top_p        = float(data.get('top_p', 0.85))
    sample_count = int(data.get('sample_count', 3))
    lookback  = 400
    pred_len  = 120

    # ── Step 1: Pull live data from MT5 ──────────────────────────
    if MT5_AVAILABLE and mt5_conn.connected:
        df, error = mt5_conn.get_live_data(symbol, timeframe, bars)
        if error:
            return jsonify({'error': f'MT5 data pull failed: {error}'}), 400
        data_source = f'Live MT5 ({symbol} {timeframe})'
    else:
        # Fallback to local XAUUSD file
        xau_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'XAUUSD_15m.csv')
        if not os.path.exists(xau_path):
            return jsonify({'error': 'MT5 not connected and no local XAUUSD data found. Connect MT5 first.'}), 400
        df, err = load_data_file(xau_path)
        if err:
            return jsonify({'error': err}), 400
        data_source = 'Local XAUUSD_15m.csv'

    # Rename 'date' to 'timestamps' if needed
    if 'date' in df.columns and 'timestamps' not in df.columns:
        df['timestamps'] = pd.to_datetime(df['date'])
    df['timestamps'] = pd.to_datetime(df['timestamps'])

    if len(df) < lookback + pred_len:
        return jsonify({'error': f'Not enough data: {len(df)} bars, need {lookback + pred_len}'}), 400

    # Use the most recent lookback bars
    hist_df = df.iloc[-(lookback + pred_len):-(pred_len)].copy().reset_index(drop=True)
    actual_window = df.iloc[-(pred_len):].copy().reset_index(drop=True)

    required_cols = ['open', 'high', 'low', 'close']
    if 'volume' in df.columns:
        required_cols.append('volume')

    x_df        = hist_df[required_cols].copy()
    x_timestamp = pd.Series(hist_df['timestamps'].values, name='timestamps')
    y_timestamp = pd.Series(actual_window['timestamps'].values, name='timestamps')

    # ── Step 2: Auto-refresh news sentiment ───────────────────────
    try:
        import urllib.request, urllib.parse, xml.etree.ElementTree as ET
        from textblob import TextBlob as _TB
        gold_query = urllib.parse.quote('Gold XAUUSD price forecast')
        url = f'https://news.google.com/rss/search?q={gold_query}&hl=en&gl=US&ceid=US:en'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=6) as resp:
            root = ET.fromstring(resp.read())
        polarity_sum = 0.0; count = 0
        for item in root.iter('item'):
            title = item.findtext('title', '')
            desc  = item.findtext('description', '')
            import re as _re
            text = _re.sub(r'<[^>]+>', '', f"{title}. {desc}")
            p = _TB(text).sentiment.polarity
            macro_b = any(k in text.lower() for k in ['rate hike', 'hawkish', 'strong dollar'])
            macro_u = any(k in text.lower() for k in ['rate cut', 'dovish', 'safe haven', 'recession'])
            if macro_b: p -= 0.15
            if macro_u: p += 0.15
            polarity_sum += max(-1.0, min(1.0, p)); count += 1
            if count >= 15: break
        if count > 0:
            avg = round(polarity_sum / count, 4)
            sig = 'BULLISH' if avg > 0.12 else 'BEARISH' if avg < -0.12 else 'NEUTRAL'
            current_sentiment.update({
                'score': avg, 'signal': sig,
                'bias': round(avg * 2.5, 4),
                'articles': count, 'query': 'Gold XAUUSD'
            })
    except Exception as e:
        print(f'News auto-refresh failed (non-fatal): {e}')

    # ── Step 3: Run Kronos prediction ─────────────────────────────
    try:
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=temperature,
            top_p=top_p,
            sample_count=sample_count
        )
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

    # Apply sentiment bias
    bias_pct = current_sentiment.get('bias', 0.0)
    if bias_pct != 0.0:
        factor = 1.0 + (bias_pct / 100.0)
        for col in ['open', 'high', 'low', 'close']:
            if col in pred_df.columns:
                pred_df[col] = pred_df[col] * factor

    # ── Step 4: Build chart and signal ───────────────────────────
    chart_json = create_prediction_chart(df, pred_df, lookback, pred_len, actual_window, max(0, len(df) - lookback - pred_len))

    last_close   = float(hist_df['close'].iloc[-1])
    pred_avg     = float(pred_df['close'].mean())
    change_pct   = (pred_avg - last_close) / last_close * 100
    signal = 'BUY' if change_pct > 0.05 else 'SELL' if change_pct < -0.05 else 'HOLD'

    # Blend with sentiment signal
    if current_sentiment['signal'] == 'BULLISH' and signal == 'HOLD': signal = 'BUY'
    if current_sentiment['signal'] == 'BEARISH' and signal == 'HOLD': signal = 'SELL'

    prediction_results = []
    for i, (_, row) in enumerate(pred_df.iterrows()):
        prediction_results.append({
            'timestamp': str(y_timestamp.iloc[i]) if i < len(y_timestamp) else f'T+{i}',
            'open':   float(row['open']),
            'high':   float(row['high']),
            'low':    float(row['low']),
            'close':  float(row['close']),
            'volume': float(row['volume']) if 'volume' in row else 0,
        })

    actual_data = []
    for i, (_, row) in enumerate(actual_window.iterrows()):
        actual_data.append({
            'timestamp': str(row['timestamps']),
            'open':   float(row['open']),
            'high':   float(row['high']),
            'low':    float(row['low']),
            'close':  float(row['close']),
            'volume': float(row['volume']) if 'volume' in row else 0,
        })

    return jsonify({
        'success': True,
        'data_source': data_source,
        'symbol': symbol,
        'timeframe': timeframe,
        'signal': signal,
        'change_pct': round(change_pct, 4),
        'current_price': last_close,
        'predicted_avg_close': pred_avg,
        'chart': chart_json,
        'prediction_results': prediction_results,
        'actual_data': actual_data,
        'has_comparison': len(actual_data) > 0,
        'sentiment': current_sentiment,
        'prediction_type': f'Kronos-base | {data_source} | Sentiment: {current_sentiment["signal"]} ({bias_pct:+.2f}%)',
        'message': f'✅ {symbol} prediction: {signal} | {change_pct:+.3f}% | News: {current_sentiment["signal"]}'
    })


@app.route('/api/mt5/predict-and-trade', methods=['POST'])
def mt5_predict_and_trade():
    """Full pipeline: pull live data → predict → generate signal → optionally trade"""
    if not MT5_AVAILABLE:
        return jsonify({'error': 'MT5 integration not available'}), 400
    
    if not MODEL_AVAILABLE or predictor is None:
        return jsonify({'error': 'Kronos model not loaded'}), 400
    
    data = request.get_json()
    symbol = data.get('symbol', 'EURUSD')
    timeframe = data.get('timeframe', '15m')
    auto_execute = data.get('auto_execute', False)
    volume = float(data.get('volume', 0.01))
    sl_pips = int(data.get('sl_pips', 50))
    tp_pips = int(data.get('tp_pips', 100))
    
    # Pull live data
    df, error = mt5_conn.get_live_data(symbol, timeframe, 520)
    if error:
        return jsonify({'error': f'Data pull failed: {error}'}), 400
    
    lookback = 400
    pred_len = 120
    
    if len(df) < lookback:
        return jsonify({'error': f'Insufficient data: {len(df)} bars, need {lookback}'}), 400
    
    # Prepare data for Kronos
    df['timestamps'] = pd.to_datetime(df['date'])
    required_cols = ['open', 'high', 'low', 'close']
    if 'volume' in df.columns:
        required_cols.append('volume')
    
    x_df = df.iloc[:lookback][required_cols]
    x_timestamp = pd.Series(df.iloc[:lookback]['timestamps'].values, name='timestamps')
    y_timestamp = pd.Series(df.iloc[lookback:lookback+pred_len]['timestamps'].values, name='timestamps') if len(df) >= lookback + pred_len else None
    
    # If not enough future timestamps, generate them
    if y_timestamp is None or len(y_timestamp) < pred_len:
        last_ts = df['timestamps'].iloc[lookback-1]
        time_diff = df['timestamps'].iloc[1] - df['timestamps'].iloc[0]
        future_ts = pd.date_range(start=last_ts + time_diff, periods=pred_len, freq=time_diff)
        y_timestamp = pd.Series(future_ts, name='timestamps')
    
    # Run prediction
    try:
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=0.8,
            top_p=0.85,
            sample_count=3
        )
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
    
    # Generate signal
    signal, change_pct = mt5_conn.generate_signal(pred_df, df.iloc[:lookback])
    
    # Build chart
    chart_json = create_prediction_chart(df, pred_df, lookback, pred_len, None, 0)
    
    result = {
        'success': True,
        'symbol': symbol,
        'timeframe': timeframe,
        'signal': signal,
        'change_pct': round(change_pct, 4),
        'current_price': float(df['close'].iloc[lookback-1]),
        'predicted_avg_close': float(pred_df['close'].mean()),
        'chart': chart_json,
        'trade_executed': False
    }
    
    # Auto-execute trade if enabled
    if auto_execute and signal != 'HOLD':
        success, message = mt5_conn.execute_trade(symbol, signal, volume, sl_pips, tp_pips)
        result['trade_executed'] = success
        result['trade_message'] = message
    
    return jsonify(result)

@app.route('/api/mt5/status')
def mt5_status():
    """Get MT5 connection status"""
    return jsonify({
        'available': MT5_AVAILABLE,
        'connected': mt5_conn.connected if MT5_AVAILABLE else False,
        'account': mt5_conn.account_info if MT5_AVAILABLE else None
    })

@app.route('/api/mt5/live-chart-data', methods=['POST'])
def mt5_live_chart_data():
    """Get live OHLCV data formatted for chart rendering"""
    if not MT5_AVAILABLE:
        return jsonify({'error': 'MT5 integration not available'}), 400
    
    data = request.get_json()
    symbol = data.get('symbol', 'EURUSD')
    timeframe = data.get('timeframe', '15m')
    bars = int(data.get('bars', 200))
    
    df, error = mt5_conn.get_live_data(symbol, timeframe, bars)
    if error:
        return jsonify({'error': error}), 400
    
    return jsonify({
        'success': True,
        'symbol': symbol,
        'timeframe': timeframe,
        'dates': df['date'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
        'open': df['open'].tolist(),
        'high': df['high'].tolist(),
        'low': df['low'].tolist(),
        'close': df['close'].tolist(),
        'volume': df['volume'].tolist()
    })

# ==================== BACKTEST ROUTES ====================

@app.route('/api/backtest/start', methods=['POST'])
def backtest_start():
    """Start a backtest"""
    global backtest_engine
    
    if not BACKTEST_AVAILABLE:
        return jsonify({'error': 'Backtest engine not available'}), 400
    if not MT5_AVAILABLE or not mt5_conn.connected:
        return jsonify({'error': 'MT5 not connected'}), 400
    if not MODEL_AVAILABLE or predictor is None:
        return jsonify({'error': 'Model not loaded. Load Kronos-base first.'}), 400
    
    data = request.get_json()
    symbol = data.get('symbol', 'XAUUSD')
    volume = float(data.get('volume', 0.01))
    sl_pips = int(data.get('sl_pips', 200))   # Gold needs wider SL
    tp_pips = int(data.get('tp_pips', 400))   # Gold needs wider TP

    # Data path — prefer XAUUSD, fallback to GBPUSD
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', f'{symbol}_15m_backtest.csv')
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'GBPUSD_15m_backtest.csv')
    if not os.path.exists(data_path):
        return jsonify({'error': f'Backtest data not found. Pull XAUUSD or GBPUSD 15m data first.'}), 400
    
    # Create and configure engine
    backtest_engine = BacktestEngine(predictor, mt5_conn)
    backtest_engine.volume = volume
    backtest_engine.sl_pips = sl_pips
    backtest_engine.tp_pips = tp_pips
    
    success, message = backtest_engine.start(data_path, symbol)
    return jsonify({'success': success, 'message': message})

@app.route('/api/backtest/stop', methods=['POST'])
def backtest_stop():
    """Stop the backtest"""
    global backtest_engine
    if backtest_engine is None:
        return jsonify({'error': 'No backtest running'}), 400
    success, message = backtest_engine.stop()
    return jsonify({'success': success, 'message': message})

@app.route('/api/backtest/status')
def backtest_status():
    """Get backtest state (polled by frontend for real-time updates)"""
    global backtest_engine
    if backtest_engine is None:
        return jsonify({'status': 'idle', 'stats': {}})
    return jsonify(backtest_engine.state)

# ==================== NEWS SENTIMENT ROUTES ====================

@app.route('/api/news/fetch', methods=['POST'])
def news_fetch():
    """Fetch Gold/XAUUSD financial news and analyse sentiment"""
    global current_sentiment

    if not SENTIMENT_AVAILABLE:
        return jsonify({'error': 'TextBlob not installed'}), 400

    import urllib.request, urllib.parse, xml.etree.ElementTree as ET

    data = request.get_json()
    query = data.get('query', 'Gold XAUUSD').strip()

    articles = []

    GOLD_KEYWORDS = ['gold', 'xauusd', 'xau', 'bullion', 'precious metal',
                     'safe haven', 'inflation', 'fed', 'interest rate',
                     'dollar', 'fomc', 'cpi', 'nfp', 'rate hike']

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    # ── 1. Google News RSS (Gold focused) ─────────────────────────
    gold_searches = [
        f"Gold price XAUUSD {query}",
        "Gold XAU inflation Fed rate",
        "XAUUSD forecast bullion"
    ]
    for search in gold_searches:
        try:
            encoded = urllib.parse.quote(search)
            url = f"https://news.google.com/rss/search?q={encoded}&hl=en&gl=US&ceid=US:en"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                xml_data = resp.read()
            root = ET.fromstring(xml_data)
            for item in root.iter('item'):
                title   = item.findtext('title', '').strip()
                desc    = item.findtext('description', '').strip()
                link    = item.findtext('link', '').strip()
                pub     = item.findtext('pubDate', '').strip()
                if not title:
                    continue
                # Clean HTML tags from description
                import re as _re
                desc_clean = _re.sub(r'<[^>]+>', '', desc)[:200]
                text = f"{title}. {desc_clean}"
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                lower = text.lower()
                is_gold = any(k in lower for k in ['gold', 'xauusd', 'xau', 'bullion'])
                # Apply gold correlation for macro events
                macro_bearish = any(k in lower for k in ['rate hike', 'hawkish', 'strong dollar', 'nfp beat', 'jobs beat'])
                macro_bullish = any(k in lower for k in ['rate cut', 'dovish', 'weak dollar', 'inflation surge', 'recession', 'safe haven'])
                if macro_bearish: polarity -= 0.15
                if macro_bullish: polarity += 0.15
                polarity = max(-1.0, min(1.0, polarity))
                # Skip if duplicate title
                if any(a['title'][:50] == title[:50] for a in articles):
                    continue
                articles.append({
                    'title': title[:150],
                    'link': link,
                    'published': pub,
                    'polarity': round(float(polarity), 4),
                    'subjectivity': round(float(blob.sentiment.subjectivity), 4),
                    'snippet': desc_clean,
                    'relevance': 'gold' if is_gold else 'macro',
                    'source': 'Google News'
                })
                if len(articles) >= 20:
                    break
        except Exception as e:
            print(f"Google News error: {e}")
        if len(articles) >= 20:
            break

    # ── 2. Yahoo Finance RSS (Gold/commodity news) ───────────────
    if len(articles) < 10:
        try:
            url = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=GC=F&region=US&lang=en-US"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=8) as resp:
                xml_data = resp.read()
            root = ET.fromstring(xml_data)
            for item in root.iter('item'):
                title = item.findtext('title', '').strip()
                desc  = item.findtext('description', '').strip()
                link  = item.findtext('link', '').strip()
                pub   = item.findtext('pubDate', '').strip()
                if not title or any(a['title'][:50] == title[:50] for a in articles):
                    continue
                text = f"{title}. {desc}"
                blob = TextBlob(text)
                articles.append({
                    'title': title[:150],
                    'link': link,
                    'published': pub,
                    'polarity': round(float(blob.sentiment.polarity), 4),
                    'subjectivity': round(float(blob.sentiment.subjectivity), 4),
                    'snippet': desc[:200],
                    'relevance': 'gold',
                    'source': 'Yahoo Finance'
                })
                if len(articles) >= 25:
                    break
        except Exception as e:
            print(f"Yahoo Finance error: {e}")

    if not articles:
        return jsonify({'error': 'Could not fetch news. Please check your internet connection and try again.'}), 404

    # Sort: gold-relevant first, then strongest sentiment
    articles.sort(key=lambda a: (0 if a.get('relevance') == 'gold' else 1, -abs(a['polarity'])))
    articles = articles[:20]

    # Weighted sentiment — gold articles count 2x
    weighted_sum = sum(a['polarity'] * (2.0 if a.get('relevance') == 'gold' else 1.0) for a in articles)
    weight_total = sum(2.0 if a.get('relevance') == 'gold' else 1.0 for a in articles)
    avg_polarity = round(weighted_sum / weight_total, 4) if weight_total > 0 else 0.0

    signal = 'BULLISH' if avg_polarity > 0.12 else 'BEARISH' if avg_polarity < -0.12 else 'NEUTRAL'
    bias   = round(avg_polarity * 2.5, 4)  # max ±2.5% for Gold

    current_sentiment.update({
        'score': avg_polarity, 'signal': signal,
        'bias': bias, 'articles': len(articles), 'query': query
    })

    return jsonify({'success': True, 'query': query, 'articles': articles, 'sentiment': current_sentiment})
@app.route('/api/news/sentiment')
def news_sentiment():
    """Return current cached sentiment"""
    return jsonify(current_sentiment)


@app.route('/api/news/clear', methods=['POST'])
def news_clear():
    """Clear sentiment bias"""
    global current_sentiment
    current_sentiment = {'score': 0.0, 'signal': 'NEUTRAL', 'bias': 0.0, 'articles': 0, 'query': ''}
    return jsonify({'success': True, 'message': 'Sentiment cleared'})


if __name__ == '__main__':
    print("Starting Kronos Web UI...")
    print(f"Model availability: {MODEL_AVAILABLE}")
    if MODEL_AVAILABLE:
        print("Tip: You can load Kronos model through /api/load-model endpoint")
    else:
        print("Tip: Will use simulated data for demonstration")
    
    app.run(debug=True, host='0.0.0.0', port=7070)
