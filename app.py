from flask import Flask, jsonify
import pandas as pd
import re

app = Flask(__name__)

# Load data once (improves performance)
CSV_FILE = "nifty_data.csv"
df = pd.read_csv(CSV_FILE)

# --- Extract strike from tradingsymbol column ---
def extract_strike(symbol):
    match = re.search(r'(\d{5})(CE|PE)', symbol)
    if match:
        return match.group(1) + match.group(2)
    return None

df['strike'] = df['tradingsymbol'].apply(extract_strike)

@app.route('/')
def home():
    return "✅ Nifty Options API is live!"

@app.route('/preview')
def preview():
    return df.head(10).to_json(orient="records")

@app.route('/strike/<strike>')
def get_strike_data(strike):
    strike = strike.upper()
    filtered = df[df['strike'] == strike]
    if filtered.empty:
        return jsonify({"error": f"No data found for strike {strike}"}), 404
    return filtered.to_json(orient="records")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
