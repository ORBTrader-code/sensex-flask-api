# app.py
from flask import Flask, request, jsonify
import pandas as pd
import re
import os

app = Flask(__name__)

# === CONFIG ===
CSV_FILE = os.path.join(os.path.dirname(__file__), "nifty_data.csv")
TF_MAP = {"1m":"1T","3m":"3T","5m":"5T","15m":"15T","30m":"30T","1h":"1H"}

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

# === RESAMPLE UTILITY ===
def resample_ohlcv(df_slice, timeframe):
    if df_slice.empty:
        return df_slice
    rule = TF_MAP.get(timeframe, "1T")
    temp = df_slice.set_index('timestamp').sort_index()
    agg = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    res = temp.resample(rule).apply(agg).dropna().reset_index()
    # Convert timestamp to string for JSON
    res['timestamp'] = res['timestamp'].dt.strftime("%Y-%m-%d %H:%M:%S")
    # Keep only expected columns
    return res[['timestamp','open','high','low','close','volume']]

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
      - symbol=FULL_TRADINGSYMBOL  (optional if using strike+type)
      - strike=24700               (optional)
      - type=CE|PE                 (optional)
      - expiry=YYYY-MM-DD         (optional)
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
        return jsonify({"error": f"Resample error: {e}"}), 500

    # Return JSON array
    return jsonify(resampled.to_dict(orient='records'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
