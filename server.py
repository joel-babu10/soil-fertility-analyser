from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import joblib
import pandas as pd
import numpy as np
import logging
import time
import sys
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ---------------- SAFE ADVANCED IMPORTS ----------------
try:
    from advanced_monitoring.history_manager import add_reading
    from advanced_monitoring.advanced_engine import advanced_monitor
    ADVANCED_AVAILABLE = True
except:
    ADVANCED_AVAILABLE = False

    def add_reading(data):
        pass

    def advanced_monitor(crop):
        return {
            "status": "basic_mode",
            "crop": crop,
            "message": "Advanced monitoring module not installed"
        }

app = Flask(__name__)
CORS(app)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

print("🌾 Smart Terrace Farming AI ONLINE")

# ---------------- WEATHER API ----------------
API_KEY = "YOUR_API_KEY"
LAT = "9.4981"
LON = "76.3388"

# ---------------- LOAD MODEL ----------------
model = joblib.load("crop_model.pkl")

# ---------------- GLOBAL VARIABLES ----------------
latest_sensor_data = None
predicted_crops = []
selected_crop = None

# ---------------- LEAF DISEASE MODULE ----------------
try:
    from disease import bp as disease_bp, init_disease
    init_disease(lambda: latest_sensor_data)
    app.register_blueprint(disease_bp)
    print("🍃 Leaf disease module loaded")
except Exception as e:
    print("⚠ Leaf disease module not loaded:", e)

# ============================================================
#  CROP REQUIREMENTS DATABASE
#  ideal ranges: N, P, K (mg/kg), temp (°C), moisture (%), pH
# ============================================================
CROP_REQUIREMENTS = {
    # ---- field crops ----
    "rice":        {"N": (80,120), "P": (40,60),  "K": (40,60),  "temp": (20,35), "moisture": (60,80), "ph": (5.5,7.0)},
    "wheat":       {"N": (60,100), "P": (30,50),  "K": (30,50),  "temp": (15,25), "moisture": (50,70), "ph": (6.0,7.5)},
    "maize":       {"N": (80,120), "P": (40,60),  "K": (50,70),  "temp": (20,35), "moisture": (55,75), "ph": (5.8,7.0)},
    "chickpea":    {"N": (20,40),  "P": (40,60),  "K": (20,40),  "temp": (15,30), "moisture": (40,60), "ph": (6.0,8.0)},
    "kidneybeans": {"N": (20,40),  "P": (40,60),  "K": (20,40),  "temp": (18,30), "moisture": (50,70), "ph": (6.0,7.5)},
    "pigeonpeas":  {"N": (20,40),  "P": (30,50),  "K": (20,40),  "temp": (20,35), "moisture": (40,60), "ph": (5.5,7.0)},
    "mothbeans":   {"N": (10,30),  "P": (20,40),  "K": (10,30),  "temp": (25,40), "moisture": (30,50), "ph": (6.0,8.0)},
    "mungbean":    {"N": (20,40),  "P": (30,50),  "K": (20,40),  "temp": (25,35), "moisture": (50,70), "ph": (6.0,7.5)},
    "blackgram":   {"N": (20,40),  "P": (30,50),  "K": (20,40),  "temp": (25,35), "moisture": (50,70), "ph": (6.0,7.5)},
    "lentil":      {"N": (20,40),  "P": (30,50),  "K": (20,40),  "temp": (18,28), "moisture": (40,60), "ph": (6.0,8.0)},
    # ---- fruits ----
    "pomegranate": {"N": (50,80),  "P": (30,50),  "K": (50,80),  "temp": (25,38), "moisture": (30,50), "ph": (5.5,7.5)},
    "banana":      {"N": (80,120), "P": (40,60),  "K": (80,120), "temp": (26,35), "moisture": (60,80), "ph": (5.5,7.0)},
    "mango":       {"N": (50,80),  "P": (30,50),  "K": (50,80),  "temp": (24,38), "moisture": (40,60), "ph": (5.5,7.5)},
    "grapes":      {"N": (40,70),  "P": (30,50),  "K": (60,90),  "temp": (15,35), "moisture": (40,60), "ph": (5.5,7.0)},
    "watermelon":  {"N": (60,90),  "P": (30,50),  "K": (50,80),  "temp": (25,38), "moisture": (50,70), "ph": (6.0,7.5)},
    "muskmelon":   {"N": (60,90),  "P": (30,50),  "K": (50,80),  "temp": (25,38), "moisture": (50,70), "ph": (6.0,7.5)},
    "apple":       {"N": (50,80),  "P": (30,50),  "K": (50,80),  "temp": (10,25), "moisture": (40,60), "ph": (5.5,7.0)},
    "orange":      {"N": (50,80),  "P": (30,50),  "K": (60,90),  "temp": (22,35), "moisture": (45,65), "ph": (6.0,7.5)},
    "papaya":      {"N": (60,90),  "P": (40,60),  "K": (60,90),  "temp": (22,35), "moisture": (55,75), "ph": (5.5,7.0)},
    "coconut":     {"N": (50,80),  "P": (30,50),  "K": (80,120), "temp": (25,35), "moisture": (50,70), "ph": (5.5,8.0)},
    "coffee":      {"N": (40,70),  "P": (30,50),  "K": (40,70),  "temp": (18,28), "moisture": (50,70), "ph": (5.0,6.5)},
    # ---- terrace/pot crops mapped from TERRACE_MAP ----
    "tomato":      {"N": (60,100), "P": (40,60),  "K": (60,90),  "temp": (20,30), "moisture": (55,75), "ph": (6.0,7.0)},
    "chilli":      {"N": (50,80),  "P": (30,50),  "K": (40,70),  "temp": (20,35), "moisture": (50,70), "ph": (5.5,7.0)},
    "spinach":     {"N": (60,100), "P": (30,50),  "K": (40,60),  "temp": (10,25), "moisture": (55,75), "ph": (6.0,7.5)},
    "coriander":   {"N": (40,70),  "P": (20,40),  "K": (30,50),  "temp": (15,25), "moisture": (45,65), "ph": (6.0,7.5)},
    "fenugreek":   {"N": (40,70),  "P": (20,40),  "K": (20,40),  "temp": (15,30), "moisture": (45,65), "ph": (6.0,7.5)},
    "mint":        {"N": (50,80),  "P": (20,40),  "K": (30,50),  "temp": (18,28), "moisture": (55,75), "ph": (6.0,7.5)},
    "okra":        {"N": (50,80),  "P": (30,50),  "K": (40,60),  "temp": (22,35), "moisture": (50,70), "ph": (6.0,7.5)},
    "brinjal":     {"N": (60,100), "P": (40,60),  "K": (50,80),  "temp": (22,35), "moisture": (55,75), "ph": (5.5,7.0)},
    "lettuce":     {"N": (50,80),  "P": (30,50),  "K": (30,50),  "temp": (12,22), "moisture": (55,75), "ph": (6.0,7.5)},
    "radish":      {"N": (40,70),  "P": (30,50),  "K": (30,50),  "temp": (10,20), "moisture": (50,70), "ph": (6.0,7.5)},
    "basil":       {"N": (40,70),  "P": (20,40),  "K": (30,50),  "temp": (18,30), "moisture": (50,70), "ph": (6.0,7.5)},
    "cucumber":    {"N": (60,90),  "P": (40,60),  "K": (50,80),  "temp": (20,32), "moisture": (60,80), "ph": (6.0,7.5)},
    "beans":       {"N": (20,40),  "P": (40,60),  "K": (20,40),  "temp": (18,30), "moisture": (50,70), "ph": (6.0,7.5)},
    "ginger":      {"N": (50,80),  "P": (30,50),  "K": (50,80),  "temp": (20,30), "moisture": (60,80), "ph": (5.5,6.5)},
    "turmeric":    {"N": (60,90),  "P": (30,50),  "K": (60,90),  "temp": (20,35), "moisture": (60,80), "ph": (5.5,7.0)},
    "microgreens": {"N": (30,60),  "P": (20,40),  "K": (20,40),  "temp": (15,25), "moisture": (50,70), "ph": (6.0,7.5)},
    "amaranthus":  {"N": (40,80),  "P": (20,40),  "K": (30,60),  "temp": (20,35), "moisture": (40,60), "ph": (6.0,8.0)},
    # ---- missing terrace-mapped crops ----
    "peas":              {"N": (20,40),  "P": (40,60),  "K": (20,40),  "temp": (10,24), "moisture": (50,70), "ph": (6.0,7.5)},
    "green gram":        {"N": (20,40),  "P": (30,50),  "K": (20,40),  "temp": (25,35), "moisture": (50,70), "ph": (6.0,7.5)},
    "pole beans":        {"N": (20,40),  "P": (40,60),  "K": (20,40),  "temp": (18,30), "moisture": (50,70), "ph": (6.0,7.5)},
    "bush beans":        {"N": (20,40),  "P": (40,60),  "K": (20,40),  "temp": (18,30), "moisture": (50,70), "ph": (6.0,7.5)},
    "cowpea":            {"N": (20,40),  "P": (30,50),  "K": (20,40),  "temp": (25,38), "moisture": (40,60), "ph": (5.5,7.5)},
    "cluster beans":     {"N": (20,40),  "P": (30,50),  "K": (20,40),  "temp": (25,38), "moisture": (40,60), "ph": (6.0,7.5)},
    "green gram sprouts":{"N": (20,40),  "P": (30,50),  "K": (20,40),  "temp": (25,35), "moisture": (50,70), "ph": (6.0,7.5)},
    "leafy greens":      {"N": (50,80),  "P": (30,50),  "K": (30,50),  "temp": (12,25), "moisture": (55,75), "ph": (6.0,7.5)},
    "dwarf pomegranate": {"N": (50,80),  "P": (30,50),  "K": (50,80),  "temp": (25,38), "moisture": (30,50), "ph": (5.5,7.5)},
    "curry leaf":        {"N": (40,70),  "P": (20,40),  "K": (30,60),  "temp": (20,35), "moisture": (40,60), "ph": (6.0,7.5)},
    "climbing beans":    {"N": (20,40),  "P": (40,60),  "K": (20,40),  "temp": (18,30), "moisture": (50,70), "ph": (6.0,7.5)},
    "bitter gourd":      {"N": (50,80),  "P": (30,50),  "K": (40,70),  "temp": (25,38), "moisture": (55,75), "ph": (6.0,7.5)},
    "bottle gourd":      {"N": (50,80),  "P": (30,50),  "K": (40,70),  "temp": (25,38), "moisture": (55,75), "ph": (6.0,7.5)},
    "ridge gourd":       {"N": (50,80),  "P": (30,50),  "K": (40,70),  "temp": (25,38), "moisture": (55,75), "ph": (6.0,7.5)},
    "snake gourd":       {"N": (50,80),  "P": (30,50),  "K": (40,70),  "temp": (25,38), "moisture": (55,75), "ph": (6.0,7.5)},
    "pumpkin":           {"N": (60,90),  "P": (40,60),  "K": (50,80),  "temp": (22,35), "moisture": (55,75), "ph": (6.0,7.5)},
    "strawberry":        {"N": (40,70),  "P": (30,50),  "K": (40,70),  "temp": (15,26), "moisture": (55,75), "ph": (5.5,6.5)},
    "lemon":             {"N": (50,80),  "P": (30,50),  "K": (60,90),  "temp": (22,35), "moisture": (45,65), "ph": (6.0,7.5)},
    "shade herbs":       {"N": (30,60),  "P": (20,40),  "K": (20,40),  "temp": (15,28), "moisture": (50,70), "ph": (6.0,7.5)},
}

TERRACE_MAP = {
    "rice": ["spinach","coriander","fenugreek"],
    "maize": ["tomato","chilli","okra"],
    "chickpea": ["beans","peas","green gram"],
    "kidneybeans": ["pole beans","bush beans","cowpea"],
    "pigeonpeas": ["brinjal","cluster beans","cowpea"],
    "mothbeans": ["microgreens","mint","spinach"],
    "mungbean": ["green gram sprouts","microgreens","spinach"],
    "blackgram": ["amaranthus","spinach","mint"],
    "lentil": ["leafy greens","lettuce","fenugreek"],
    "pomegranate": ["chilli","tomato","dwarf pomegranate"],
    "banana": ["okra","brinjal","cucumber"],
    "mango": ["curry leaf","mint","basil"],
    "grapes": ["climbing beans","cucumber","bitter gourd"],
    "watermelon": ["cucumber","bottle gourd","ridge gourd"],
    "muskmelon": ["cucumber","snake gourd","pumpkin"],
    "apple": ["lettuce","spinach","strawberry"],
    "orange": ["lemon","mint","coriander"],
    "papaya": ["tomato","chilli","okra"],
    "coconut": ["curry leaf","turmeric","ginger"],
    "cotton": ["mint","basil","spinach"],
    "jute": ["amaranthus","spinach","lettuce"],
    "coffee": ["curry leaf","mint","shade herbs"]
}

# ============================================================
#  CROP HEALTH ENGINE
# ============================================================
def get_crop_requirements(crop_name):
    """Return requirements dict for a crop (case-insensitive, partial match)."""
    name = crop_name.lower().strip()
    if name in CROP_REQUIREMENTS:
        return CROP_REQUIREMENTS[name]
    # partial match
    for key in CROP_REQUIREMENTS:
        if key in name or name in key:
            return CROP_REQUIREMENTS[key]
    return None

def score_param(val, lo, hi):
    """Score a single parameter: 100 if in range, penalised if outside."""
    if lo <= val <= hi:
        return 100
    elif val < lo:
        deficit = (lo - val) / lo
        return max(0, round(100 - deficit * 100))
    else:
        excess = (val - hi) / hi
        return max(0, round(100 - excess * 100))

def fertilizer_advice(crop, N, P, K, ph, req):
    """Generate targeted fertilizer advice."""
    advice = []
    N_lo, N_hi = req["N"]
    P_lo, P_hi = req["P"]
    K_lo, K_hi = req["K"]
    ph_lo, ph_hi = req["ph"]

    # Nitrogen
    if N < N_lo:
        deficit = N_lo - N
        urea_g = round(deficit * 0.003 * 12, 1)
        advice.append({
            "nutrient": "Nitrogen (N)",
            "status": "LOW",
            "action": f"Add {urea_g}g urea OR 150–200g kitchen compost per 12L pot",
            "organic": "Compost tea (50 ml/week), dried neem leaves, banana peel compost",
            "urgency": "HIGH" if N < N_lo * 0.5 else "MEDIUM"
        })
    elif N > N_hi:
        advice.append({
            "nutrient": "Nitrogen (N)",
            "status": "EXCESS",
            "action": "Stop nitrogen fertilizers. Flush pot with plain water 2–3 times.",
            "organic": "Add carbon-rich material like dry leaves to balance",
            "urgency": "MEDIUM"
        })
    else:
        advice.append({"nutrient": "Nitrogen (N)", "status": "OK", "action": "Maintain with monthly compost top-dressing", "organic": "—", "urgency": "LOW"})

    # Phosphorus
    if P < P_lo:
        dap_g = round((P_lo - P) * 0.002 * 12, 1)
        advice.append({
            "nutrient": "Phosphorus (P)",
            "status": "LOW",
            "action": f"Add {dap_g}g DAP OR mix bone meal into top 5 cm of soil",
            "organic": "Banana peel powder, rock phosphate, wood ash",
            "urgency": "HIGH" if P < P_lo * 0.5 else "MEDIUM"
        })
    elif P > P_hi:
        advice.append({"nutrient": "Phosphorus (P)", "status": "EXCESS", "action": "Avoid phosphate fertilizers. Use plain water flush.", "organic": "—", "urgency": "LOW"})
    else:
        advice.append({"nutrient": "Phosphorus (P)", "status": "OK", "action": "Good — no action needed", "organic": "—", "urgency": "LOW"})

    # Potassium
    if K < K_lo:
        mop_g = round((K_lo - K) * 0.0025 * 12, 1)
        advice.append({
            "nutrient": "Potassium (K)",
            "status": "LOW",
            "action": f"Add {mop_g}g MOP (muriate of potash) OR 2 tbsp wood ash per pot",
            "organic": "Banana peel ash, coconut shell ash, seaweed extract",
            "urgency": "HIGH" if K < K_lo * 0.5 else "MEDIUM"
        })
    elif K > K_hi:
        advice.append({"nutrient": "Potassium (K)", "status": "EXCESS", "action": "Skip potash. Water regularly to dilute.", "organic": "—", "urgency": "LOW"})
    else:
        advice.append({"nutrient": "Potassium (K)", "status": "OK", "action": "Good — maintain with compost", "organic": "—", "urgency": "LOW"})

    # pH correction
    ph_advice = {}
    if ph < ph_lo:
        lime_g = round((ph_lo - ph) * 5 * 12, 1)
        ph_advice = {
            "status": "TOO_ACIDIC",
            "action": f"Add {lime_g}g garden lime or wood ash",
            "organic": "Crushed eggshells, wood ash, oyster shell powder",
            "note": f"{crop.title()} needs pH {ph_lo}–{ph_hi}. Current: {ph}"
        }
    elif ph > ph_hi:
        sulfur_g = round((ph - ph_hi) * 3 * 12, 1)
        ph_advice = {
            "status": "TOO_ALKALINE",
            "action": f"Add {sulfur_g}g sulfur powder OR 2 tbsp coffee grounds",
            "organic": "Coffee grounds, peat moss, vinegar solution (1 tsp/litre)",
            "note": f"{crop.title()} needs pH {ph_lo}–{ph_hi}. Current: {ph}"
        }
    else:
        ph_advice = {
            "status": "OK",
            "action": "pH is within optimal range — maintain with compost",
            "organic": "Continue organic top-dressing",
            "note": f"pH {ph} is ideal for {crop.title()}"
        }

    return advice, ph_advice

def watering_advice(crop, moisture, temp, req):
    mo_lo, mo_hi = req["moisture"]
    t_lo, t_hi = req["temp"]
    if moisture < mo_lo:
        if temp > t_hi:
            return {"schedule": "Water TWICE daily", "amount": "Until water drains from bottom", "tip": f"Heat stress + dry soil — {crop} needs immediate watering. Move to shade if possible.", "urgency": "CRITICAL"}
        return {"schedule": "Water today", "amount": "Until water drains from bottom", "tip": f"{crop.title()} prefers moisture {mo_lo}–{mo_hi}%. Water early morning at base of plant.", "urgency": "HIGH"}
    elif moisture > mo_hi:
        return {"schedule": "Skip 1–2 days", "amount": "No watering needed", "tip": f"Soil too wet for {crop}. Check pot drainage. Remove saucers with standing water.", "urgency": "MEDIUM"}
    else:
        if temp > t_hi:
            return {"schedule": "Every day", "amount": "Light watering", "tip": f"Good moisture but hot. Water in evening to reduce evaporation.", "urgency": "LOW"}
        return {"schedule": "Every 2 days", "amount": "When top 2 cm is dry", "tip": f"Conditions balanced for {crop}. Check soil with finger before watering.", "urgency": "LOW"}

def overall_health(scores):
    avg = sum(scores.values()) / len(scores)
    if avg >= 80:
        return "EXCELLENT", avg
    elif avg >= 60:
        return "GOOD", avg
    elif avg >= 40:
        return "FAIR", avg
    else:
        return "POOR", avg

# ============================================================
#  ROUTES
# ============================================================

@app.route('/')
def home():
    return render_template("dashboard.html")

def hash_stage(title, duration=1.2):
    print(f"\n{title}")
    for i in range(25):
        time.sleep(duration / 25)
        sys.stdout.write("#")
        sys.stdout.flush()
    print(" Done!")

def classify_level(value):
    value = float(value)
    if value < 25:
        return "LOW"
    elif value < 50:
        return "MEDIUM"
    else:
        return "HIGH"

def get_weekly_rainfall():
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
        response = requests.get(url, timeout=5)
        data = response.json()
        rain_1h = data.get("rain", {}).get("1h", 0)
        return float(round(rain_1h * 24 * 7, 2))
    except:
        return 0.0

def calculate_fertility_score(N, P, K, ph):
    score = 0
    score += min(N / 140, 1) * 30
    score += min(P / 80, 1) * 20
    score += min(K / 80, 1) * 20
    if 6 <= ph <= 7.5:
        score += 30
    elif 5 <= ph <= 8:
        score += 20
    else:
        score += 10
    return round(score)

def calculate_soil_health(N, P, K, moisture, ph):
    score = 0
    score += min(N / 120, 1) * 25
    score += min(P / 60, 1) * 20
    score += min(K / 60, 1) * 20
    score += min(moisture / 100, 1) * 15
    if 6 <= ph <= 7.5:
        score += 20
    else:
        score += 10
    return round(score)

def improvement_plan(N, P, K, ph):
    plans = []
    if N < 30:
        plans.append("Add compost or nitrogen fertilizer.")
    if P < 30:
        plans.append("Use bone meal or phosphorus fertilizer.")
    if K < 30:
        plans.append("Add wood ash or potassium fertilizer.")
    if ph < 6:
        plans.append("Increase pH using lime or wood ash.")
    elif ph > 8:
        plans.append("Reduce pH using compost or sulfur.")
    if len(plans) == 0:
        plans.append("Soil condition looks healthy.")
    return plans


@app.route('/sensor', methods=['POST'])
def receive_data():
    global latest_sensor_data, predicted_crops
    data = request.get_json()
    print("\n📥 Received JSON:", data)
    try:
        N = float(data.get('N', 0))
        P = float(data.get('P', 0))
        K = float(data.get('K', 0))
        temp = float(data.get('temp', 0))
        if temp == -127:
            temp = 25
        moisture = float(data.get('moisture') or data.get('humidity') or 0)
        ph = float(data.get('ph', 7))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    latest_sensor_data = {"N": N, "P": P, "K": K, "temp": temp, "moisture": moisture, "ph": ph}
    rainfall = get_weekly_rainfall()

    input_data = pd.DataFrame([{"N": N, "P": P, "K": K, "temperature": temp, "moisture": moisture, "ph": ph, "rainfall": rainfall}])
    probabilities = model.predict_proba(input_data)[0]
    top3_indices = np.argsort(probabilities)[-3:][::-1]
    top3_crops = model.classes_[top3_indices]
    top3_conf = probabilities[top3_indices] * 100

    terrace_crops = []
    for crop in top3_crops:
        crop_name = str(crop).lower()
        if crop_name in TERRACE_MAP:
            terrace_crops.extend(TERRACE_MAP[crop_name])
        else:
            terrace_crops.append(crop_name)
    predicted_crops = list(dict.fromkeys(terrace_crops))[:3]

    fertility_score = calculate_fertility_score(N, P, K, ph)
    soil_health_score = calculate_soil_health(N, P, K, moisture, ph)
    nitrogen_level = classify_level(N)
    phosphorus_level = classify_level(P)
    potassium_level = classify_level(K)

    if fertility_score < 40:
        fertility_level = "LOW"
    elif fertility_score < 70:
        fertility_level = "MEDIUM"
    else:
        fertility_level = "HIGH"

    plans = improvement_plan(N, P, K, ph)
    while len(plans) < 2:
        plans.append("Maintain regular watering and composting.")

    add_reading({"N": N, "P": P, "K": K, "temp": temp, "moisture": moisture, "ph": ph, "timestamp": str(datetime.now())})

    print("\n🌱 Terrace Crop Suggestions:", predicted_crops)

    return jsonify({
        "crop1": predicted_crops[0],
        "confidence1": round(float(top3_conf[0]), 2),
        "crop2": predicted_crops[1],
        "confidence2": round(float(top3_conf[1]), 2),
        "crop3": predicted_crops[2],
        "confidence3": round(float(top3_conf[2]), 2),
        "fertility_score": fertility_score,
        "fertility_level": fertility_level,
        "soil_health_score": soil_health_score,
        "nitrogen_level": nitrogen_level,
        "phosphorus_level": phosphorus_level,
        "potassium_level": potassium_level,
        "plan1": plans[0],
        "plan2": plans[1],
        "rainfall_used": rainfall
    }), 200


@app.route('/get_crops', methods=['GET'])
def get_crops():
    global predicted_crops
    if not predicted_crops:
        return jsonify({"status": "waiting"}), 200
    return jsonify({"status": "ok", "crops": predicted_crops}), 200


@app.route('/select_crop', methods=['POST'])
def select_crop():
    global selected_crop, predicted_crops
    data = request.get_json()
    crop = data.get("crop")
    if crop not in predicted_crops:
        return jsonify({"error": "Invalid crop selection"}), 400
    selected_crop = crop
    return jsonify({"status": "Crop selected successfully", "selected_crop": selected_crop}), 200


# ============================================================
#  NEW: /crop_health  — full monitoring report for selected crop
# ============================================================
@app.route('/crop_health', methods=['POST'])
def crop_health():
    """
    POST body: { "crop": "tomato" }
    Uses latest_sensor_data to generate a full health + fertilizer report.
    """
    global latest_sensor_data, selected_crop

    data = request.get_json()
    crop = data.get("crop") or selected_crop

    if not crop:
        return jsonify({"error": "No crop specified"}), 400
    if not latest_sensor_data:
        return jsonify({"error": "No sensor data available"}), 400

    crop = str(crop).lower().strip()
    selected_crop = crop  # auto-select

    N        = float(latest_sensor_data.get("N", 0))
    P        = float(latest_sensor_data.get("P", 0))
    K        = float(latest_sensor_data.get("K", 0))
    temp     = float(latest_sensor_data.get("temp", 25))
    moisture = float(latest_sensor_data.get("moisture", 50))
    ph       = float(latest_sensor_data.get("ph", 7))

    req = get_crop_requirements(crop)

    if req is None:
        # No requirement data — return basic response
        return jsonify({
            "crop": crop,
            "status": "NO_DATA",
            "message": f"No specific requirement data for {crop}. Using general guidelines.",
            "general_advice": [
                "Maintain soil pH between 6.0 and 7.0 for most crops.",
                "Keep moisture between 50–70% for best growth.",
                "Apply balanced NPK fertilizer (19:19:19) once a month.",
                "Water at base of plant early morning.",
                "Check for pests weekly — use neem oil spray as prevention."
            ]
        }), 200

    # --- score each parameter ---
    scores = {
        "N":        score_param(N,        *req["N"]),
        "P":        score_param(P,        *req["P"]),
        "K":        score_param(K,        *req["K"]),
        "temp":     score_param(temp,     *req["temp"]),
        "moisture": score_param(moisture, *req["moisture"]),
        "ph":       score_param(ph,       *req["ph"]),
    }

    health_status, health_avg = overall_health(scores)

    fert_advice, ph_advice = fertilizer_advice(crop, N, P, K, ph, req)
    water = watering_advice(crop, moisture, temp, req)

    # ideal ranges for display
    ideal = {
        "N":        f"{req['N'][0]}–{req['N'][1]} mg/kg",
        "P":        f"{req['P'][0]}–{req['P'][1]} mg/kg",
        "K":        f"{req['K'][0]}–{req['K'][1]} mg/kg",
        "temp":     f"{req['temp'][0]}–{req['temp'][1]} °C",
        "moisture": f"{req['moisture'][0]}–{req['moisture'][1]} %",
        "ph":       f"{req['ph'][0]}–{req['ph'][1]}",
    }

    current = {"N": N, "P": P, "K": K, "temp": temp, "moisture": moisture, "ph": ph}

    # urgency alerts
    alerts = []
    for item in fert_advice:
        if item["urgency"] == "HIGH":
            alerts.append(f"🚨 {item['nutrient']} is critically {item['status']} for {crop.title()}")
    if ph_advice["status"] == "TOO_ACIDIC":
        alerts.append(f"⚠ pH too acidic — {crop.title()} prefers {req['ph'][0]}–{req['ph'][1]}")
    if ph_advice["status"] == "TOO_ALKALINE":
        alerts.append(f"⚠ pH too alkaline — {crop.title()} prefers {req['ph'][0]}–{req['ph'][1]}")
    if water["urgency"] == "CRITICAL":
        alerts.append(f"🌡 Critical heat + dry soil — water {crop.title()} immediately!")

    print(f"\n🌿 Crop Health Report for {crop}: {health_status} ({round(health_avg)})")

    return jsonify({
        "crop": crop,
        "health_status": health_status,
        "health_score": round(health_avg),
        "scores": scores,
        "ideal": ideal,
        "current": current,
        "fertilizer_advice": fert_advice,
        "ph_advice": ph_advice,
        "watering": water,
        "alerts": alerts,
        "timestamp": str(datetime.now())
    }), 200



# ============================================================
#  WEATHER FORECAST — real 7-day from OpenWeatherMap
# ============================================================
@app.route('/weather', methods=['GET'])
def get_weather():
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric&cnt=56"
        r = requests.get(url, timeout=6)
        data = r.json()
        if r.status_code != 200:
            raise Exception(data.get("message", "API error"))
        # Group by day
        days = {}
        for item in data["list"]:
            date = item["dt_txt"][:10]
            if date not in days:
                days[date] = {"temps": [], "rain": 0, "icons": [], "descs": []}
            days[date]["temps"].append(item["main"]["temp"])
            days[date]["rain"] += item.get("rain", {}).get("3h", 0)
            days[date]["icons"].append(item["weather"][0]["main"])
            days[date]["descs"].append(item["weather"][0]["description"])
        result = []
        for date, v in list(days.items())[:7]:
            dominant = max(set(v["icons"]), key=v["icons"].count)
            icon_map = {"Clear": "☀️", "Clouds": "🌤", "Rain": "🌧", "Drizzle": "🌦", "Thunderstorm": "⛈", "Snow": "❄️", "Mist": "🌫", "Haze": "🌫"}
            result.append({
                "date": date,
                "icon": icon_map.get(dominant, "🌤"),
                "desc": v["descs"][0].capitalize(),
                "min": round(min(v["temps"]), 1),
                "max": round(max(v["temps"]), 1),
                "rain_mm": round(v["rain"], 1)
            })
        total_rain = sum(d["rain_mm"] for d in result)
        return jsonify({"days": result, "total_rain_mm": round(total_rain, 1)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
#  CROP ROTATION SUGGESTER
# ============================================================
ROTATION_DB = {
    "tomato":    {"family": "Solanaceae", "next": ["beans", "basil", "spinach", "lettuce"],       "avoid": ["chilli", "brinjal", "potato"], "reason": "Rotates nitrogen-heavy soil; legumes restore N"},
    "chilli":    {"family": "Solanaceae", "next": ["beans", "coriander", "spinach"],               "avoid": ["tomato", "brinjal"], "reason": "Solanaceae crop — follow with legumes or leafy greens"},
    "brinjal":   {"family": "Solanaceae", "next": ["coriander", "fenugreek", "beans"],             "avoid": ["tomato", "chilli"], "reason": "Heavy feeder — follow with nitrogen-fixers"},
    "spinach":   {"family": "Amaranthaceae", "next": ["tomato", "beans", "cucumber"],              "avoid": ["lettuce", "amaranthus"], "reason": "Light feeder — heavy feeders benefit from the rested soil"},
    "coriander": {"family": "Apiaceae", "next": ["tomato", "brinjal", "okra"],                     "avoid": ["fennel"], "reason": "Neutral crop — opens soil for heavy feeders"},
    "beans":     {"family": "Leguminosae", "next": ["tomato", "chilli", "brinjal", "cucumber"],    "avoid": ["peas", "cowpea"], "reason": "Fixes nitrogen — ideal before heavy feeders"},
    "okra":      {"family": "Malvaceae", "next": ["coriander", "spinach", "fenugreek"],            "avoid": [], "reason": "Deep roots improve soil structure"},
    "cucumber":  {"family": "Cucurbitaceae", "next": ["beans", "coriander", "lettuce"],            "avoid": ["pumpkin", "bottle gourd"], "reason": "Exhausts soil — follow with light or N-fixing crops"},
    "lettuce":   {"family": "Asteraceae", "next": ["tomato", "beans", "cucumber"],                 "avoid": [], "reason": "Shallow feeder — great before deep-rooted crops"},
    "radish":    {"family": "Brassicaceae", "next": ["beans", "tomato", "spinach"],                "avoid": [], "reason": "Loosens compact soil — good before beans"},
    "mint":      {"family": "Lamiaceae", "next": ["tomato", "coriander", "brinjal"],               "avoid": [], "reason": "Repels pests — great companion before fruiting crops"},
    "ginger":    {"family": "Zingiberaceae", "next": ["coriander", "spinach", "beans"],            "avoid": [], "reason": "Depletes K — follow with K-restoring legumes"},
    "turmeric":  {"family": "Zingiberaceae", "next": ["spinach", "coriander", "lettuce"],          "avoid": ["ginger"], "reason": "Similar K depletion as ginger — rotate to leafy greens"},
    "microgreens":{"family": "Various", "next": ["tomato", "chilli", "beans"],                     "avoid": [], "reason": "Minimal soil impact — any crop can follow"},
    "fenugreek": {"family": "Leguminosae", "next": ["tomato", "brinjal", "okra"],                  "avoid": [], "reason": "Nitrogen fixer — prime soil for heavy feeders"},
    "basil":     {"family": "Lamiaceae", "next": ["tomato", "beans", "cucumber"],                  "avoid": [], "reason": "Pest repellent — great before fruiting crops"},
    "amaranthus":{"family": "Amaranthaceae", "next": ["beans", "coriander", "tomato"],             "avoid": ["spinach"], "reason": "Heavy N user — follow with N-fixing legumes"},
    "pumpkin":   {"family": "Cucurbitaceae", "next": ["beans", "peas", "spinach"],                 "avoid": ["cucumber", "bottle gourd"], "reason": "Depletes soil — restore with legumes"},
    "bitter gourd":{"family": "Cucurbitaceae", "next": ["beans", "coriander", "lettuce"],          "avoid": ["cucumber"], "reason": "Heavy feeder — follow with light crops"},
    "bottle gourd":{"family": "Cucurbitaceae", "next": ["beans", "spinach", "coriander"],          "avoid": ["cucumber", "pumpkin"], "reason": "Exhausts potassium — legumes restore balance"},
}

@app.route('/crop_rotation', methods=['POST'])
def crop_rotation():
    data = request.get_json()
    history_crops = data.get("history", [])  # list of last 1-3 crops grown
    if not history_crops:
        return jsonify({"error": "No crop history provided"}), 400
    last = str(history_crops[-1]).lower().strip()
    info = ROTATION_DB.get(last)
    if not info:
        return jsonify({
            "last_crop": last,
            "suggestions": ["beans", "spinach", "coriander"],
            "avoid": [],
            "reason": "General rotation: follow heavy feeders with legumes or leafy greens.",
            "family": "Unknown"
        }), 200
    # filter out crops already in recent history
    recent = [c.lower() for c in history_crops[-3:]]
    suggestions = [c for c in info["next"] if c not in recent][:4]
    return jsonify({
        "last_crop": last,
        "family": info["family"],
        "suggestions": suggestions,
        "avoid": info["avoid"],
        "reason": info["reason"]
    }), 200


# ============================================================
#  PEST & DISEASE ALERT
# ============================================================
PEST_RULES = [
    {"cond": lambda t, m: t > 30 and m > 75, "risk": "HIGH",   "name": "Fungal Disease (Leaf Blight / Powdery Mildew)", "action": "Reduce watering, improve air circulation, spray neem oil 5ml/litre", "organic": "Neem oil spray, copper fungicide, remove affected leaves"},
    {"cond": lambda t, m: t > 28 and m > 70, "risk": "HIGH",   "name": "Damping Off (Pythium)", "action": "Stop overhead watering, water only at base, ensure drainage", "organic": "Cinnamon powder on soil surface, chamomile tea drench"},
    {"cond": lambda t, m: t > 35 and m < 40, "risk": "HIGH",   "name": "Spider Mites", "action": "Mist plants daily, isolate affected pots, spray neem oil", "organic": "Neem oil + soap spray, introduce predatory mites"},
    {"cond": lambda t, m: t > 25 and m > 65, "risk": "MEDIUM", "name": "Aphids / Whiteflies", "action": "Check leaf undersides weekly, use yellow sticky traps", "organic": "Garlic spray, neem oil, blast with water"},
    {"cond": lambda t, m: t < 20 and m > 70, "risk": "MEDIUM", "name": "Root Rot (Phytophthora)", "action": "Reduce watering, check drainage, repot if roots are brown", "organic": "Trichoderma powder in soil, well-draining potting mix"},
    {"cond": lambda t, m: t > 32 and m > 60, "risk": "MEDIUM", "name": "Bacterial Wilt", "action": "Remove affected plants immediately, sanitize tools", "organic": "Avoid wounding plants, use disease-free seeds"},
    {"cond": lambda t, m: 25 < t < 35 and 50 < m < 70, "risk": "LOW", "name": "Caterpillars / Leaf Miners", "action": "Inspect weekly, handpick larvae, use Bt spray if severe", "organic": "Bacillus thuringiensis (Bt) spray, neem cake in soil"},
]

@app.route('/pest_alert', methods=['GET'])
def pest_alert():
    global latest_sensor_data
    if not latest_sensor_data:
        return jsonify({"error": "No sensor data"}), 400
    temp = float(latest_sensor_data.get("temp", 25))
    moisture = float(latest_sensor_data.get("moisture", 50))
    alerts = []
    for rule in PEST_RULES:
        if rule["cond"](temp, moisture):
            alerts.append({"risk": rule["risk"], "name": rule["name"], "action": rule["action"], "organic": rule["organic"]})
    return jsonify({"temp": temp, "moisture": moisture, "alerts": alerts, "count": len(alerts)}), 200


# ============================================================
#  FERTILIZER SCHEDULE CALENDAR
# ============================================================
@app.route('/fert_schedule', methods=['POST'])
def fert_schedule():
    global latest_sensor_data
    data = request.get_json()
    crop = str(data.get("crop", "")).lower().strip()
    if not crop or not latest_sensor_data:
        return jsonify({"error": "Missing crop or sensor data"}), 400
    req = get_crop_requirements(crop)
    if not req:
        return jsonify({"error": f"No data for {crop}"}), 400
    N = float(latest_sensor_data.get("N", 0))
    P = float(latest_sensor_data.get("P", 0))
    K = float(latest_sensor_data.get("K", 0))
    ph = float(latest_sensor_data.get("ph", 7))
    schedule = []
    # Week 1
    w1 = []
    if N < req["N"][0]: w1.append(f"Apply {round((req['N'][0]-N)*0.003*12,1)}g Urea OR compost tea 50ml")
    if P < req["P"][0]: w1.append(f"Mix {round((req['P'][0]-P)*0.002*12,1)}g DAP into top soil")
    if not w1: w1.append("No fertilizer needed — soil is balanced")
    schedule.append({"week": "Week 1", "tasks": w1, "priority": "HIGH" if len(w1) > 1 else "LOW"})
    # Week 2
    w2 = ["Check soil moisture daily", "Foliar spray: 1 tsp seaweed extract per litre water"]
    if K < req["K"][0]: w2.insert(0, f"Add {round((req['K'][0]-K)*0.0025*12,1)}g MOP (Muriate of Potash)")
    schedule.append({"week": "Week 2", "tasks": w2, "priority": "MEDIUM"})
    # Week 3
    w3 = ["Top-dress with 100g kitchen compost", "Check for pests on leaf undersides"]
    if ph < req["ph"][0]: w3.insert(0, f"Add {round((req['ph'][0]-ph)*5*12,1)}g garden lime for pH correction")
    elif ph > req["ph"][1]: w3.insert(0, f"Add {round((ph-req['ph'][1])*3*12,1)}g sulfur powder for pH correction")
    schedule.append({"week": "Week 3", "tasks": w3, "priority": "MEDIUM"})
    # Week 4
    w4 = ["Repeat Week 1 NPK application at half dose", "Prune dead leaves, mulch surface with dry leaves"]
    schedule.append({"week": "Week 4", "tasks": w4, "priority": "LOW"})
    return jsonify({"crop": crop, "schedule": schedule}), 200


@app.route('/maintain', methods=['GET'])
def maintain():
    global selected_crop, latest_sensor_data
    if selected_crop is None:
        return jsonify({"error": "No crop selected"}), 400
    if latest_sensor_data is None:
        return jsonify({"error": "No sensor data available"}), 400
    result = advanced_monitor(selected_crop)
    return jsonify({"selected_crop": selected_crop, "sensor_data": latest_sensor_data, "monitoring": result, "timestamp": str(datetime.now())}), 200


@app.route('/latest', methods=['GET'])
def get_latest():
    global latest_sensor_data
    if latest_sensor_data is None:
        return jsonify({"status": "waiting"}), 200
    return jsonify({"status": "ok", "data": latest_sensor_data}), 200


@app.route('/reset', methods=['GET'])
def reset_prediction():
    global predicted_crops, selected_crop
    predicted_crops = []
    selected_crop = None
    return jsonify({"status": "System reset"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)