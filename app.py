from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import re
import os

app = Flask(__name__)
CORS(app)

# === CONFIG ===
TF_MAP = {
    "1m": "1T", 
    "3m": "3T", 
    "5m": "5T", 
    "10m": "10T", 
    "15m": "15T", 
    "30m": "30T", 
    "1h": "1H",
    "1D": "D"
}

def extract_strike_and_type(sym):
    m = re.search(r'(\d{5})(CE|PE)', str(sym))
    if m:
        return m.group(1), m.group(2)
    return None, None

def resample_ohlcv(df_slice, timeframe):
    if df_slice.empty:
        return df_slice

    rule = TF_MAP.get(timeframe, "1T")
    df_slice = df_slice.copy()
    df_slice["timestamp"] = pd.to_datetime(df_slice["timestamp"])
    df_slice = df_slice.set_index("timestamp").sort_index()
    df_slice = df_slice.between_time("09:15", "15:30")

    if timeframe == '1m':
        res = df_slice.reset_index()
        res["timestamp"] = res["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return res[["timestamp", "open", "high", "low", "close", "volume"]]

    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}

    if timeframe == '1D':
        res = df_slice.groupby(df_slice.index.date).agg(agg)
        res.index.name = "timestamp"
        res = res.reset_index()
        res["timestamp"] = pd.to_datetime(res["timestamp"]).dt.strftime("%Y-%m-%d 00:00:00")
        return res[["timestamp", "open", "high", "low", "close", "volume"]]

    parts = []
    for _, day_df in df_slice.groupby(df_slice.index.date):
        if day_df.empty:
            continue
        res_day = (
            day_df.resample(rule, label="left", closed="left", origin='9:15:00')
                  .apply(agg)
                  .dropna(subset=['open','high','low','close'])
        )
        parts.append(res_day)

    if not parts:
        return pd.DataFrame()

    res = pd.concat(parts).sort_index().reset_index()
    res.rename(columns={"index":"timestamp"}, inplace=True)
    res["timestamp"] = res["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return res[["timestamp","open","high","low","close","volume"]]

@app.route('/')
def home():
    return "✅ Nifty Options API (Expiry-based dynamic loading) is running."

@app.route('/preview')
def preview():
    files = [f for f in os.listdir() if f.startswith("nifty_data_") and f.endswith(".csv")]
    return jsonify({"available_files": files})

@app.route('/get_chart')
def get_chart():
    timeframe = request.args.get('timeframe', '1m')
    strike = request.args.get('strike')
    opt_type = request.args.get('type', 'PE')
    expiry = request.args.get('expiry')  # e.g. 04nov25

    if not expiry:
        return jsonify({"error": "Expiry parameter is required"}), 400

    file_name = f"nifty_data_{expiry.lower()}.csv"
    if not os.path.exists(file_name):
        return jsonify({"error": f"File {file_name} not found"}), 404

    try:
        df = pd.read_csv(file_name)
        df.columns = [c.strip() for c in df.columns]
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['tradingsymbol'] = df['tradingsymbol'].astype(str)
        df['_strike_num'] = df['tradingsymbol'].apply(lambda s: extract_strike_and_type(s)[0])
        df['_opt_type']  = df['tradingsymbol'].apply(lambda s: extract_strike_and_type(s)[1])
        df['expiry'] = df.get('expiry', pd.Series(dtype=str)).astype(str)
    except Exception as e:
        return jsonify({"error": f"Error reading CSV: {e}"}), 500

    if not strike:
        return jsonify({"error": "Strike parameter required"}), 400

    strike = str(int(strike))
    mask = (df['_strike_num'] == strike) & (df['_opt_type'] == opt_type.upper())
    df_slice = df[mask].copy()

    if df_slice.empty:
        return jsonify({"error": "No data found for the given strike/type"}), 404

    try:
        resampled = resample_ohlcv(df_slice, timeframe)
    except Exception as e:
        return jsonify({"error": f"Resample error: {e}"}), 500

    return jsonify(resampled.to_dict(orient='records'))

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
