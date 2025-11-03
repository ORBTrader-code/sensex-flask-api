# app.py
from flask import Flask, request, jsonify
import pandas as pd
import os

app = Flask(__name__)

# Look for CSV path (Render secret files are available at /etc/secrets/<filename>)
DEFAULT_SECRET_PATH = "/etc/secrets/nifty_data.csv"
LOCAL_FALLBACK = "nifty_data.csv"  # for local dev only (do NOT push this to GitHub)

csv_path = os.environ.get("NIFTY_CSV_PATH", DEFAULT_SECRET_PATH)
if not os.path.exists(csv_path) and os.path.exists(LOCAL_FALLBACK):
    csv_path = LOCAL_FALLBACK

if not os.path.exists(csv_path):
    # Service will still start but API will return an error until file is uploaded
    print("WARNING: CSV not found at", csv_path)
    df = pd.DataFrame(columns=["date","tradingsymbol","expiry","timestamp","open","high","low","close","volume"])
else:
    # Ensure we parse timestamp column as datetime
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])

# Map timeframe keywords to pandas offset aliases
TF_MAP = {"1m":"1T","3m":"3T","5m":"5T","15m":"15T","30m":"30T","1h":"1H"}

def resample_timeframe(dataframe, tf):
    if tf not in TF_MAP:
        tf = "1m"
    rule = TF_MAP[tf]
    # Make sure timestamp is datetime and is the index
    df_local = dataframe.copy()
    if df_local.empty:
        return df_local
    df_local = df_local.set_index("timestamp")
    # Convert columns to numeric (safe)
    for c in ["open","high","low","close","volume"]:
        df_local[c] = pd.to_numeric(df_local[c], errors="coerce")
    ohlc = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }
    res = df_local.resample(rule).apply(ohlc).dropna().reset_index()
    # Convert timestamp to ISO-like string for the frontend
    res["timestamp"] = res["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return res

@app.route("/")
def home():
    return "✅ Nifty Options API (Flask) is running."

@app.route("/get_chart")
def get_chart():
    symbol = request.args.get("symbol")
    timeframe = request.args.get("timeframe", "1m")

    if not symbol:
        return jsonify({"error":"missing symbol param"}), 400

    # Filter by tradingsymbol exactly (case-sensitive as in your data)
    data = df[df["tradingsymbol"] == symbol]
    if data.empty:
        return jsonify({"error":"symbol not found or CSV not loaded"}), 404

    out = resample_timeframe(data, timeframe)
    records = out.to_dict(orient="records")
    return jsonify(records)

if __name__ == "__main__":
    # port Render provides is in PORT env var; default to 10000 locally
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
