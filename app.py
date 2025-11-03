from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

# Load CSV once when the app starts
try:
    df = pd.read_csv("nifty_data.csv")
except Exception as e:
    df = None
    print(f"❌ Error loading CSV: {e}")

@app.route('/')
def home():
    return "✅ Nifty Data API is Running!"

@app.route('/preview')
def preview_data():
    if df is not None:
        # Return first 5 rows as JSON
        return jsonify(df.head(5).to_dict(orient="records"))
    else:
        return jsonify({"error": "Data not loaded."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
