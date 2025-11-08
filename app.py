from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import re
import os
from cachetools import LRUCache # Import the LRU Cache
import pyarrow.parquet as pq
import datetime # <<<--- IMPORTED DATETIME

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

# === Multi-Level Cache Setup ===
resample_cache = LRUCache(maxsize=200)


def resample_ohlcv(df_slice, timeframe):
    """
    Resamples the data.
    """
    if df_slice.empty:
        return df_slice

    rule = TF_MAP.get(timeframe, "1T")
    df_slice = df_slice.copy()
    
    df_slice['timestamp'] = pd.to_datetime(df_slice['timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df_slice.dropna(subset=['timestamp'], inplace=True)

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
    files = [f for f in os.listdir() if f.startswith("nifty_data_") and (f.endswith(".csv") or f.endswith(".parquet"))]
    
    expiries = set()
    for f in files:
        name = f.replace("nifty_data_", "").replace(".csv", "").replace(".parquet", "")
        expiries.add(name)
        
    # --- NEW: Create a structured list for the calendar ---
    available_expiries = []
    for name in sorted(list(expiries)):
        try:
            # Try to parse the date from the name (e.g., "04jul24")
            dt = datetime.datetime.strptime(name, "%d%b%y")
            
            # Format for the calendar (YYYY-MM-DD)
            date_str = dt.strftime("%Y-%m-%d")
            
            # Format for the label (e.g., "04 Jul 2024")
            label_str = dt.strftime("%d %b %Y")
            
            available_expiries.append({
                "value": name,      # The value the API understands (04jul24)
                "label": label_str,   # A human-friendly label
                "date": date_str      # The machine-readable date for the calendar
            })
        except ValueError:
            # Handle cases where the name isn't a date (e.g., "weekly1")
            # For this app, we'll just log it and skip it
            print(f"Could not parse date from filename: {name}")
            
    return jsonify({"available_expiries": available_expiries})
    # --- END MODIFICATION ---

@app.route('/get_chart')
def get_chart():
    timeframe = request.args.get('timeframe', '1m')
    strike = request.args.get('strike')
    opt_type = request.args.get('type', 'PE').upper()
    expiry = request.args.get('expiry')

    if not expiry:
        return jsonify({"error": "Expiry parameter is required"}), 400
    if not strike:
        return jsonify({"error": "Strike parameter required"}), 400

    strike_str = str(int(strike))

    # --- L2 CACHE CHECK ---
    cache_key = (expiry, strike_str, opt_type, timeframe)
    if cache_key in resample_cache:
        print(f"CACHE HIT (L2): Returning cached resample for {cache_key}")
        return jsonify(resample_cache[cache_key])

    print(f"CACHE MISS (L2): Processing request for {cache_key}")

    parquet_file = f"nifty_data_{expiry.lower()}.parquet"
    csv_file = f"nifty_data_{expiry.lower()}.csv"
    
    file_name = ""
    read_mode = ""

    if os.path.exists(parquet_file):
        file_name = parquet_file
        read_mode = "parquet"
    elif os.path.exists(csv_file):
        file_name = csv_file
        read_mode = "csv"
    else:
        return jsonify({"error": f"File for expiry {expiry} not found"}), 404

    COLS_TO_USE = ['timestamp', 'tradingsymbol', 'open', 'high', 'low', 'close', 'volume']
    DTYPES = {
        'tradingsymbol': 'str', 'open': 'float32', 'high': 'float32',
        'low': 'float32', 'close': 'float32', 'volume': 'int32'
    }
    
    chunksize = 100000 
    filtered_chunks = []
    
    try:
        if read_mode == "parquet":
            print(f"Starting memory-safe PARQUET read for {file_name}...")
            pf = pq.ParquetFile(file_name)
            
            for batch in pf.iter_batches(batch_size=chunksize, columns=COLS_TO_USE):
                chunk = batch.to_pandas()
                chunk.columns = [c.strip() for c in chunk.columns]
                chunk[['_strike_num', '_opt_type']] = chunk['tradingsymbol'].str.extract(r'(\d{5})(CE|PE)', expand=True)
                
                matching_rows = chunk.loc[
                    (chunk['_strike_num'] == strike_str) & 
                    (chunk['_opt_type'] == opt_type)
                ]
                
                if not matching_rows.empty:
                    filtered_chunks.append(matching_rows[COLS_TO_USE])

        elif read_mode == "csv":
            print(f"Starting memory-safe CSV read for {file_name}...")
            for chunk in pd.read_csv(
                file_name,
                usecols=COLS_TO_USE,
                dtype=DTYPES,
                chunksize=chunksize
            ):
                chunk.columns = [c.strip() for c in chunk.columns]
                chunk[['_strike_num', '_opt_type']] = chunk['tradingsymbol'].str.extract(r'(\d{5})(CE|PE)', expand=True)
                
                matching_rows = chunk.loc[
                    (chunk['_strike_num'] == strike_str) & 
                    (chunk['_opt_type'] == opt_type)
                ]
                
                if not matching_rows.empty:
                    filtered_chunks.append(matching_rows[COLS_TO_USE])

        print(f"Chunked read finished. Found {len(filtered_chunks)} matching chunks.")

    except Exception as e:
        print(f"Error during chunked read: {e}")
        return jsonify({"error": f"Error reading file: {e}"}), 500

    if not filtered_chunks:
        return jsonify({"error": "No data found for the given strike/type"}), 404

    try:
        df_slice = pd.concat(filtered_chunks, ignore_index=True)
    except Exception as e:
        return jsonify({"error": f"Error concatenating chunks: {e}"}), 500

    if df_slice.empty:
        return jsonify({"error": "No data found for the given strike/type"}), 404

    try:
        resampled = resample_ohlcv(df_slice, timeframe)
        result_dict = resampled.to_dict(orient='records')

        resample_cache[cache_key] = result_dict
        print(f"CACHE SET (L2): Stored resample for {cache_key}")

    except Exception as e:
        return jsonify({"error": f"Resample error: {e}"}), 500

    return jsonify(result_dict)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)