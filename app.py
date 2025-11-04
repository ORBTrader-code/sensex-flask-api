from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import re
import os

app = Flask(__name__)
CORS(app)

# === CONFIG ===
CSV_FILE = os.path.join(os.path.dirname(__file__), "nifty_data.csv")
# Global TF_MAP used for pandas rule lookup
TF_MAP = {
    "1m": "1T", 
    "3m": "3T", 
    "5m": "5T", 
    "10m": "10T", 
    "15m": "15T", 
    "30m": "30T", 
    "1h": "1H"
}

# === LOAD DATA ===
try:
    df = pd.read_csv(CSV_FILE)
    # Ensure consistent names and dtypes
    df.columns = [c.strip() for c in df.columns]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Create normalized tradingsymbol and extracted strike/type columns
    df['tradingsymbol'] = df['tradingsymbol'].astype(str)
except Exception as e:
    print("❌ Error loading CSV:", e)
    df = pd.DataFrame(columns=["date","tradingsymbol","expiry","timestamp","open","high","low","close","volume"])

# helper: extract strike + type (like "24700PE") and strike number "24700"
def extract_strike_and_type(sym):
    m = re.search(r'(\d{5})(CE|PE)', sym)
    if m:
        return m.group(1), m.group(2)
    return None, None

df['_strike_num'] = df['tradingsymbol'].apply(lambda s: extract_strike_and_type(s)[0])
df['_opt_type'] = df['tradingsymbol'].apply(lambda s: extract_strike_and_type(s)[1])
# normalize expiry if present as string (keep as-is)
df['expiry'] = df['expiry'].astype(str)

# === RESAMPLE UTILITY (CORRECTED) ===
def resample_ohlcv(df_slice, timeframe):
    """
    Resamples 1-minute OHLCV data to a higher timeframe, respecting
    the Indian market session (09:15:00 to 15:30:00).
    The logic ensures:
    - Open = First candle's open in the group.
    - Close = Last candle's close in the group.
    - High/Low = Max/Min of all candles in the group.
    """
    if df_slice.empty:
        return df_slice

    rule = TF_MAP.get(timeframe, "1T")

    df_slice = df_slice.copy()
    df_slice["timestamp"] = pd.to_datetime(df_slice["timestamp"])
    df_slice = df_slice.set_index("timestamp").sort_index()

    # Filter to the trading session (09:15:00 to 15:30:00 inclusive)
    df_slice = df_slice.between_time("09:15", "15:30")
    
    # If 1m, no resampling needed, just format and return the already filtered data
    if timeframe == '1m':
        res = df_slice.reset_index().rename(columns={"timestamp": "timestamp"})
        res["timestamp"] = res["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return res[["timestamp", "open", "high", "low", "close", "volume"]]

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }

    resampled_days = []
    
    # Group by date to handle daily sessions separately and prevent data leakage across days
    for day, day_df in df_slice.groupby(df_slice.index.date):
        
        if day_df.empty:
            continue

        # 1. Resample the daily data. 
        # Use origin='9:15:00' to align all timeframes to the exact market open time.
        res_day = (
            day_df.resample(
                rule, 
                label="left", 
                closed="left",
                origin='9:15:00'
            )
            .apply(agg)
            # Drop any bars that had no 1-minute data in their period
            .dropna(subset=['open', 'high', 'low', 'close']) 
        )
        
        # 2. FIX: The strict boundary check that dropped the final partial bar 
        # (e.g., 3:15 PM candle) has been removed.
        # Since the input data is already strictly filtered to 09:15-15:30, 
        # the resampled data will naturally stop at the final bar containing data,
        # regardless of its length.
        
        resampled_days.append(res_day)

    if not resampled_days:
        return pd.DataFrame() # Return empty DataFrame if no data found

    res = pd.concat(resampled_days).sort_index().reset_index()
    res.rename(columns={"index": "timestamp"}, inplace=True)
    res["timestamp"] = res["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return res[["timestamp", "open", "high", "low", "close", "volume"]]


# === ROUTES ===
@app.route('/')
def home():
    return "✅ Nifty Options API (resample-enabled) is running."

@app.route('/preview')
def preview():
    return jsonify(df.head(5).to_dict(orient='records'))

@app.route('/get_chart')
def get_chart():
    """
    Query params:
      - symbol=FULL_TRADINGSYMBOL   (optional if using strike+type)
      - strike=24700                (optional)
      - type=CE|PE                  (optional)
      - expiry=YYYY-MM-DD           (optional)
      - timeframe=1m|3m|5m|15m|30m|1h  (default: 1m)
    """
    timeframe = request.args.get('timeframe', '1m')
    symbol = request.args.get('symbol')
    strike = request.args.get('strike')
    opt_type = request.args.get('type')  # CE or PE
    expiry = request.args.get('expiry')

    # 1) If symbol provided, filter exact match (case-insensitive)
    if symbol:
        mask = df['tradingsymbol'].str.upper() == symbol.upper()
        df_slice = df[mask].copy()
    else:
        # 2) Use strike + type (both required ideally)
        if not strike:
            return jsonify({"error":"Provide symbol or strike parameter"}), 400
        strike = str(int(strike))  # normalize like '24700'
        if opt_type:
            opt_type = opt_type.upper()
            mask = (df['_strike_num'] == strike) & (df['_opt_type'] == opt_type)
        else:
            # if type not provided, accept both CE/PE for given strike
            mask = (df['_strike_num'] == strike)
        df_slice = df[mask].copy()

        # 3) Optional expiry filter
        if expiry:
            df_slice = df_slice[df_slice['expiry'] == expiry]

    if df_slice.empty:
        return jsonify({"error":"No data found for the requested parameters"}), 404

    # Resample to requested timeframe
    try:
        resampled = resample_ohlcv(df_slice, timeframe)
    except Exception as e:
        # Log the error internally for better debugging
        print(f"Resample failed: {e}") 
        return jsonify({"error": f"Resample error: {e}"}), 500

    # Return JSON array
    return jsonify(resampled.to_dict(orient='records'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
