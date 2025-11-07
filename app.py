from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import re
import os
from cachetools import LRUCache # Import the LRU Cache

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

# L1 Cache: Stores *processed* DataFrames for each expiry.
# This avoids re-reading and re-processing the CSV file.
# We use an LRUCache to manage memory, especially on platforms like Render.
# maxsize=10 means it will hold the 10 most recently used expiry files.
# ADJUST maxsize based on your server's RAM and average file size.
data_cache = LRUCache(maxsize=2) # <<< MODIFICATION: Reduced cache size for low RAM

# L2 Cache: Stores the *final resampled data* (the JSON response).
# This makes identical queries (same expiry, strike, type, tf) instantaneous.
# maxsize=200 stores the 200 most recent unique chart requests.
resample_cache = LRUCache(maxsize=200)


def extract_strike_and_type(sym):
    m = re.search(r'(\d{5})(CE|PE)', str(sym))
    if m:
        return m.group(1), m.group(2)
    return None, None

def get_processed_data(expiry):
    """
    Custom function to load, process, and cache the DataFrame for a given expiry.
    This populates the L1 cache (data_cache).
    """
    # 1. Check L1 Cache first
    if expiry in data_cache:
        return data_cache[expiry]

    # 2. If not in cache, load from disk
    file_name = f"nifty_data_{expiry.lower()}.csv"
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File {file_name} not found")

    # 3. This is the slow part: I/O and heavy processing
    print(f"CACHE MISS (L1): Loading and processing {file_name}...")
    df = pd.read_csv(file_name)
    df.columns = [c.strip() for c in df.columns]
    df['timestamp'] = pd.to_datetime(df['timestamp']) # Convert timestamp ONCE
    df['tradingsymbol'] = df['tradingsymbol'].astype(str)

    # --- MODIFICATION: Optimized .apply() ---
    # Combine both regex extractions into a single pass for speed
    df[['_strike_num', '_opt_type']] = df['tradingsymbol'].apply(
        lambda s: pd.Series(extract_strike_and_type(s))
    )
    # --- END MODIFICATION ---

    df['expiry'] = df.get('expiry', pd.Series(dtype=str)).astype(str)

    # 4. --- OPTIMIZATION: Index the data for fast lookups ---
    # Drop rows where strike/type couldn't be parsed
    df.dropna(subset=['_strike_num', '_opt_type'], inplace=True)
    # Set a MultiIndex for *dramatically* faster filtering
    df.set_index(['_strike_num', '_opt_type'], inplace=True)
    df.sort_index(inplace=True) # Sorting is crucial for .loc performance

    # 5. Store the processed DataFrame in the L1 cache
    data_cache[expiry] = df
    print(f"CACHE SET (L1): Stored processed data for {expiry}")
    return df


def resample_ohlcv(df_slice, timeframe):
    """
    Resamples the data.
    NOTE: We removed the redundant pd.to_datetime conversion,
    as it's now handled in get_processed_data.
    """
    if df_slice.empty:
        return df_slice

    rule = TF_MAP.get(timeframe, "1T")
    df_slice = df_slice.copy()
    # df_slice["timestamp"] = pd.to_datetime(df_slice["timestamp"]) # No longer needed
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
    opt_type = request.args.get('type', 'PE').upper()
    expiry = request.args.get('expiry')

    if not expiry:
        return jsonify({"error": "Expiry parameter is required"}), 400
    if not strike:
        return jsonify({"error": "Strike parameter required"}), 400

    strike_str = str(int(strike)) # Clean the strike input

    # --- L2 CACHE CHECK ---
    # Create a unique key for this *specific* request
    cache_key = (expiry, strike_str, opt_type, timeframe)
    if cache_key in resample_cache:
        print(f"CACHE HIT (L2): Returning cached resample for {cache_key}")
        return jsonify(resample_cache[cache_key]) # Instant response!

    print(f"CACHE MISS (L2): Processing request for {cache_key}")

    try:
        # --- STEP 1: Get base data (from L1 cache or load/process) ---
        df = get_processed_data(expiry)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": f"Error reading or processing file: {e}"}), 500

    try:
        # --- STEP 2: Filter data (now much faster) ---
        # We use .loc on the MultiIndex for high-speed filtering
        df_slice = df.loc[(strike_str, opt_type)].copy()
    except KeyError:
        return jsonify({"error": "No data found for the given strike/type"}), 404
    except Exception as e:
        return jsonify({"error": f"Error filtering data: {e}"}), 500

    if df_slice.empty:
        return jsonify({"error": "No data found for the given strike/type"}), 404

    try:
        # --- STEP 3: Resample (only on the filtered slice) ---
        resampled = resample_ohlcv(df_slice, timeframe)
        result_dict = resampled.to_dict(orient='records')

        # --- STEP 4: Store in L2 Cache ---
        resample_cache[cache_key] = result_dict
        print(f"CACHE SET (L2): Stored resample for {cache_key}")

    except Exception as e:
        return jsonify({"error": f"Resample error: {e}"}), 500

    return jsonify(result_dict)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)