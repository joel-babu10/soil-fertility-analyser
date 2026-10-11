from flask import Flask, request, jsonify
import requests
import joblib
import pandas as pd
import numpy as np
import logging
import time
import sys
from datetime import datetime

app = Flask(__name__)

# 🔕 Disable request logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

print("🌾 Smart Soil AI System ONLINE")

API_KEY = "3c466ca4c6d8e0a29788218f2dc189b2"
LAT = "9.4981"
LON = "76.3388"

model = joblib.load("crop_model.pkl")
prediction_done = False


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


# ---------------- Fertility Analyzer ----------------
def calculate_fertility_level(n, p, k, ph):

    n = float(n)
    p = float(p)
    k = float(k)
    ph = float(ph)

    fertility = (
        min(n / 100, 1) * 25 +
        min(p / 100, 1) * 25 +
        min(k / 100, 1) * 25
    )

    if 6 <= ph <= 7.5:
        fertility += 25
    elif 5 <= ph <= 8:
        fertility += 15
    else:
        fertility += 5

    fertility = round(fertility, 2)

    if fertility >= 80:
        level = "Very Fertile"
    elif fertility >= 60:
        level = "Fertile"
    elif fertility >= 40:
        level = "Moderately Fertile"
    else:
        level = "Poor Soil"

    return fertility, level


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


# ---------------- Soil Health ----------------
def calculate_soil_health(n, p, k, ph):
    score = 0
    score += min(n / 100, 1) * 25
    score += min(p / 100, 1) * 25
    score += min(k / 100, 1) * 25

    if 6 <= ph <= 7.5:
        score += 25
    elif 5 <= ph <= 8:
        score += 15
    else:
        score += 5

    return float(round(score, 2))


# ---------------- Seasonal Filter ----------------
def seasonal_filter(crops):
    month = datetime.now().month

    kharif = ["rice", "maize", "cotton"]
    rabi = ["wheat", "barley", "mustard"]

    crops = [str(c) for c in crops]

    if 6 <= month <= 10:
        filtered = [c for c in crops if c in kharif]
    elif month >= 11 or month <= 3:
        filtered = [c for c in crops if c in rabi]
    else:
        filtered = crops

    return filtered if filtered else crops


# ---------------- Improvement Plan ----------------
def improvement_plan(n, p, k, ph):

    plan = []

    if n < 25:
        plan.append("Add Nitrogen fertilizer (Urea)")
    if p < 25:
        plan.append("Add Phosphorus fertilizer (DAP)")
    if k < 25:
        plan.append("Add Potassium fertilizer (MOP)")
    if ph < 6:
        plan.append("Add Lime to reduce soil acidity")
    if ph > 8:
        plan.append("Add organic compost to reduce alkalinity")

    if not plan:
        plan.append("Maintain balanced fertilization")

    while len(plan) < 2:
        plan.append("Monitor soil monthly")

    return plan[:2]


@app.route('/sensor', methods=['POST'])
def receive_data():
    global prediction_done

    if prediction_done:
        return jsonify({"status": "Prediction already completed"}), 200

    data = request.json

    N = float(data['N'])
    P = float(data['P'])
    K = float(data['K'])
    temp = float(data['temp'])
    moisture = float(data['humidity'])
    ph = float(data['ph'])

    print("\n--- SOIL ANALYSIS STARTED ---")

    hash_stage("Analyzing Soil Nutrients...", 1.5)
    hash_stage("Fetching Rainfall Data...", 1.5)
    rainfall = get_weekly_rainfall()

    hash_stage("Running AI Crop Model...", 2)

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

    hash_stage("Generating Soil Improvement Plan...", 1.5)

    soil_health = calculate_soil_health(N, P, K, ph)
    fertility_score, fertility_level = calculate_fertility_level(N, P, K, ph)
    plan = improvement_plan(N, P, K, ph)

    prediction_done = True

    # ---------------- PRINT PARAMETERS ----------------
    print("\n📊 SOIL PARAMETERS:")
    print(f"Nitrogen: {N} mg/kg ({classify_level(N)})")
    print(f"Phosphorus: {P} mg/kg ({classify_level(P)})")
    print(f"Potassium: {K} mg/kg ({classify_level(K)})")
    print(f"Temperature: {temp} °C")
    print(f"Moisture: {moisture} %")
    print(f"pH Level: {ph}")
    print(f"Weekly Rainfall: {rainfall} mm")
    print(f"Soil Health Score: {soil_health}/100")

    print("\n🌱 SOIL FERTILITY ANALYSIS:")
    print(f"Fertility Score: {fertility_score}/100")
    print(f"Fertility Level: {fertility_level}")

    print("\n🌾 TOP 3 CROP RECOMMENDATIONS:")
    for i in range(3):
        print(f"{i+1}. {top3_crops[i]} ({round(top3_conf[i],2)}%)")

    print("\n🛠 IMPROVEMENT SUGGESTIONS:")
    print("1.", plan[0])
    print("2.", plan[1])

    return jsonify({
        "nitrogen_level": str(classify_level(N)),
        "phosphorus_level": str(classify_level(P)),
        "potassium_level": str(classify_level(K)),

        "fertility_score": fertility_score,
        "fertility_level": fertility_level,

        "crop1": str(top3_crops[0]),
        "confidence1": float(round(top3_conf[0], 2)),
        "crop2": str(top3_crops[1]),
        "confidence2": float(round(top3_conf[1], 2)),
        "crop3": str(top3_crops[2]),
        "confidence3": float(round(top3_conf[2], 2)),

        "soil_health_score": soil_health,
        "plan1": plan[0],
        "plan2": plan[1],
        "rainfall_used": rainfall
    }), 200


@app.route('/reset', methods=['GET'])
def reset_prediction():
    global prediction_done
    prediction_done = False
    print("\n🔄 System Reset – Ready for New Prediction\n")
    return jsonify({"status": "Ready for new prediction"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
