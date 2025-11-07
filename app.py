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

# --- REMOVED L1 CACHE ---
# The L1 cache (data_cache) is removed.
# We cannot store the whole processed file in memory as it's > 512MB.

# L2 Cache: Stores the *final resampled data* (the JSON response).
# This is now our ONLY cache. It makes *identical* queries instantaneous.
# maxsize=200 stores the 200 most recent unique chart requests.
resample_cache = LRUCache(maxsize=200)


def extract_strike_and_type(sym):
    m = re.search(r'(\d{5})(CE|PE)', str(sym))
    if m:
        return m.group(1), m.group(2)
    return None, None

# --- REMOVED get_processed_data() FUNCTION ---
# We will do all processing inside get_chart() in a memory-safe way.


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
    
    # --- MODIFICATION: Timestamp conversion must happen here ---
    # We are now passing in a non-indexed, non-datetime-converted DataFrame
    df_slice['timestamp'] = pd.to_datetime(df_slice['timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df_slice.dropna(subset=['timestamp'], inplace=True)
    # --- END MODIFICATION ---

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

    # --- NEW MEMORY-SAFE CHUNKING PROCESS ---
    
    file_name = f"nifty_data_{expiry.lower()}.csv"
    if not os.path.exists(file_name):
        return jsonify({"error": f"File {file_name} not found"}), 404

    # Define ONLY the columns we need, to speed up reading.
    COLS_TO_USE = ['timestamp', 'tradingsymbol', 'open', 'high', 'low', 'close', 'volume']
    # Define data types to reduce memory and speed up parsing.
    DTYPES = {
        'tradingsymbol': 'str',
        'open': 'float32',
        'high': 'float32',
        'low': 'float32',
        'close': 'float32',
        'volume': 'int32' # Assuming volume is integer
    }
    
    # We will read the file in chunks to avoid OOM errors
    chunksize = 100000  # Read 100k rows at a time
    filtered_chunks = []

    print(f"Starting memory-safe chunked read for {file_name}...")
    
    try:
        # Loop through the file in chunks
        for chunk in pd.read_csv(
            file_name,
            usecols=COLS_TO_USE,
            dtype=DTYPES,
            # engine='pyarrow', # <<< REMOVED: This engine doesn't support chunksize
            chunksize=chunksize
        ):
            # Process this chunk
            chunk.columns = [c.strip() for c in chunk.columns]
            
            # --- Extract strike/type for this chunk ---
            chunk[['_strike_num', '_opt_type']] = chunk['tradingsymbol'].apply(
                lambda s: pd.Series(extract_strike_and_type(s))
            )
            
            # --- Filter the chunk ---
            # This is the most important step: we only keep rows that match
            matching_rows = chunk.loc[
                (chunk['_strike_num'] == strike_str) & 
                (chunk['_opt_type'] == opt_type)
            ]
            
            # If we found matching rows, save this small piece
            if not matching_rows.empty:
                # We only need the original columns for resampling
                filtered_chunks.append(matching_rows[COLS_TO_USE])
        
        print(f"Chunked read finished. Found {len(filtered_chunks)} matching chunks.")

    except Exception as e:
        print(f"Error during chunked read: {e}")
        return jsonify({"error": f"Error reading file: {e}"}), 500

    # If no chunks were found, no data exists
    if not filtered_chunks:
        return jsonify({"error": "No data found for the given strike/type"}), 404

    # --- STEP 2: Combine the small filtered pieces ---
    # This df_slice is now very small (only one strike) and fits in memory
    try:
        df_slice = pd.concat(filtered_chunks, ignore_index=True)
    except Exception as e:
        return jsonify({"error": f"Error concatenating chunks: {e}"}), 500

    if df_slice.empty:
        return jsonify({"error": "No data found for the given strike/type"}), 404

    # --- STEP 3: Resample (only on the filtered slice) ---
    try:
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