from flask import Flask, request, jsonify, render_template
import requests
import joblib
import pandas as pd
import numpy as np
import logging
import time
import sys
from datetime import datetime

# Advanced monitoring imports
from advanced_monitoring.history_manager import add_reading
from advanced_monitoring.advanced_engine import advanced_monitor

app = Flask(__name__)

# 🔕 Disable request logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

print("🌾 Smart Soil AI System ONLINE")

API_KEY = "YOUR_API_KEY"
LAT = "9.4981"
LON = "76.3388"

model = joblib.load("crop_model.pkl")

prediction_done = False
latest_sensor_data = None

# ✅ NEW GLOBAL VARIABLES
predicted_crops = []
selected_crop = None


# ---------------- Dashboard Route ----------------
@app.route('/')
def home():
    return render_template("dashboard.html")


# ---------------- Hash Loading Bar ----------------
def hash_stage(title, duration=1.5):
    print(f"\n{title}")
    for i in range(25):
        time.sleep(duration / 25)
        sys.stdout.write("#")
        sys.stdout.flush()
    print(" Done!")


# ---------------- Nutrient Level ----------------
def classify_level(value):
    value = float(value)
    if value < 25:
        return "LOW"
    elif value < 50:
        return "MEDIUM"
    else:
        return "HIGH"


# ---------------- Rainfall ----------------
def get_weekly_rainfall():
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        data = response.json()
        rain_1h = data.get("rain", {}).get("1h", 0)
        return float(round(rain_1h * 24 * 7, 2))
    except:
        return 0.0


# ---------------- SENSOR ROUTE ----------------
@app.route('/sensor', methods=['POST'])
def receive_data():
    global prediction_done, latest_sensor_data, predicted_crops

    data = request.get_json()
    print("\nReceived JSON:", data)

    try:
        N = float(data.get('N', 0))
        P = float(data.get('P', 0))
        K = float(data.get('K', 0))
        temp = float(data.get('temp', 0))
        moisture = float(data.get('humidity', 0))
        ph = float(data.get('ph', 7))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    latest_sensor_data = {
        "N": N,
        "P": P,
        "K": K,
        "temp": temp,
        "humidity": moisture,
        "ph": ph
    }

    if prediction_done:
        return jsonify({"status": "Prediction already completed"}), 200

    rainfall = get_weekly_rainfall()

    input_data = pd.DataFrame([{
        "N": N,
        "P": P,
        "K": K,
        "temperature": temp,
        "moisture": moisture,
        "ph": ph,
        "rainfall": rainfall
    }])

    probabilities = model.predict_proba(input_data)[0]
    top3_indices = np.argsort(probabilities)[-3:][::-1]

    top3_crops = model.classes_[top3_indices]
    top3_conf = probabilities[top3_indices] * 100

    # ✅ STORE PREDICTED CROPS
    predicted_crops = list(top3_crops)

    prediction_done = True

    return jsonify({
        "crop1": str(top3_crops[0]),
        "confidence1": round(float(top3_conf[0]), 2),
        "crop2": str(top3_crops[1]),
        "confidence2": round(float(top3_conf[1]), 2),
        "crop3": str(top3_crops[2]),
        "confidence3": round(float(top3_conf[2]), 2)
    }), 200


# ---------------- GET RECOMMENDED CROPS ----------------
@app.route('/get_crops', methods=['GET'])
def get_crops():
    global predicted_crops

    if not predicted_crops:
        return jsonify({"status": "waiting"}), 200

    return jsonify({
        "status": "ok",
        "crops": predicted_crops
    }), 200


# ---------------- SELECT CROP ----------------
@app.route('/select_crop', methods=['POST'])
def select_crop():
    global selected_crop, predicted_crops

    data = request.get_json()
    crop = data.get("crop")

    if crop not in predicted_crops:
        return jsonify({"error": "Invalid crop selection"}), 400

    selected_crop = crop

    return jsonify({
        "status": "Crop selected successfully",
        "selected_crop": selected_crop
    }), 200


# ---------------- MAINTENANCE ROUTE ----------------
@app.route('/maintain', methods=['GET'])
def maintain():
    global selected_crop, latest_sensor_data

    if selected_crop is None:
        return jsonify({"error": "No crop selected"}), 400

    if latest_sensor_data is None:
        return jsonify({"error": "No sensor data available"}), 400

    # Add current reading to history
    add_reading({
        "N": latest_sensor_data["N"],
        "P": latest_sensor_data["P"],
        "K": latest_sensor_data["K"]
    })

    result = advanced_monitor(selected_crop)

    return jsonify(result), 200


# ---------------- LATEST SENSOR DATA ----------------
@app.route('/latest', methods=['GET'])
def get_latest():
    global latest_sensor_data

    if latest_sensor_data is None:
        return jsonify({"status": "waiting"}), 200

    return jsonify({
        "status": "ok",
        "data": latest_sensor_data
    }), 200


# ---------------- RESET ----------------
@app.route('/reset', methods=['GET'])
def reset_prediction():
    global prediction_done, predicted_crops, selected_crop

    prediction_done = False
    predicted_crops = []
    selected_crop = None

    return jsonify({"status": "System reset"}), 200


# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)