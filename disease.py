"""Disease module: soil-based risk + leaf-image prediction.
Usage in server.py:
    from disease import bp, init_disease
    init_disease(lambda: latest_sensor_data)
    app.register_blueprint(bp)
"""
import io, json, os
import numpy as np
from flask import Blueprint, request, jsonify

bp = Blueprint("disease", __name__)
_get_sensor = lambda: None
_model = None
_classes = None
_B = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_B, "models", "leaf_model.pt")
CLASS_PATH = os.path.join(_B, "models", "leaf_classes.json")


def init_disease(sensor_getter):
    global _get_sensor
    _get_sensor = sensor_getter


# ---------------- SOIL -> DISEASE RISK (rule engine) ----------------
def soil_disease_risks(d):
    N, P, K = d["N"], d["P"], d["K"]
    t, m, ph = d["temp"], d["moisture"], d["ph"]
    r = []

    def add(name, kind, risk, why, fix):
        r.append({"disease": name, "type": kind, "risk": min(100, round(risk)),
                  "why": why, "action": fix})

    if m > 70:  # wet soil
        s = 40 + (m - 70) * 2 + (15 if 20 <= t <= 30 else 0)
        add("Root rot / Damping-off", "Fungal (Pythium, Fusarium)", s,
            f"Moisture {m}% is high; waterlogged roots lack oxygen.",
            "Stop watering, improve drainage, drench with Trichoderma or 1% Bordeaux.")
    if m > 65 and 18 <= t <= 28:
        add("Late blight / Downy mildew", "Fungal / Oomycete",
            35 + (m - 65) * 1.5 + (10 if N > 100 else 0),
            f"Cool-warm ({t}°C) and humid conditions favour spore spread.",
            "Avoid overhead watering, space plants, spray copper fungicide or neem oil.")
    if t > 30 and m < 35:
        add("Spider mites / Heat scorch", "Pest / Abiotic", 35 + (t - 30) * 4,
            f"Hot ({t}°C) and dry soil ({m}%) stress plants and favour mites.",
            "Mulch, shade at noon, water early morning, spray neem oil.")
    if N < 30:
        add("Nitrogen deficiency (yellowing)", "Nutrient", 40 + (30 - N) * 2,
            f"N is {N} mg/kg; older leaves yellow first.", "Add compost, compost tea or urea.")
    if N > 120:
        add("Excess N: soft growth, aphid/blight prone", "Nutrient", 30 + (N - 120),
            f"N is {N} mg/kg; lush tissue attracts pests.", "Stop N feeding, flush with water.")
    if P < 20:
        add("Phosphorus deficiency (purple leaves)", "Nutrient", 40 + (20 - P) * 2.5,
            f"P is {P} mg/kg; poor root and flower development.", "Bone meal or DAP.")
    if K < 25:
        add("Potassium deficiency (leaf-edge scorch)", "Nutrient", 40 + (25 - K) * 2,
            f"K is {K} mg/kg; weak resistance to disease.", "Wood ash or MOP.")
    if ph > 7.8:
        add("Iron chlorosis", "Nutrient lock-out", 35 + (ph - 7.8) * 25,
            f"pH {ph} locks up iron and zinc.", "Add sulfur / coffee grounds / peat.")
    if ph < 5.5:
        add("Acid stress / Fusarium wilt", "Soil chemistry / Fungal", 35 + (5.5 - ph) * 25,
            f"pH {ph} is too acidic; Fusarium thrives.", "Add lime or wood ash.")
    if m < 25:
        add("Drought wilt", "Abiotic", 40 + (25 - m) * 2, f"Moisture only {m}%.", "Water deeply, mulch.")
    r.sort(key=lambda x: -x["risk"])
    return r


@bp.route("/disease_soil", methods=["GET"])
def disease_soil():
    d = _get_sensor()
    if not d:
        return jsonify({"status": "waiting"}), 200
    risks = soil_disease_risks(d)
    top = risks[0]["risk"] if risks else 0
    level = "HIGH" if top >= 65 else "MEDIUM" if top >= 40 else "LOW"
    return jsonify({"status": "ok", "overall": level, "risks": risks, "sensor": d})


# ---------------- LEAF IMAGE -> DISEASE ----------------
TREATMENTS = [
    ("healthy", "Leaf looks healthy. Keep regular watering and monthly compost."),
    ("late_blight", "Remove infected leaves, avoid wetting foliage, spray copper fungicide."),
    ("early_blight", "Remove lower infected leaves, mulch, spray mancozeb/copper, rotate crops."),
    ("blight", "Remove infected leaves, improve airflow, apply copper fungicide."),
    ("rust", "Remove affected leaves, avoid overhead watering, apply sulfur spray."),
    ("powdery_mildew", "Improve airflow, spray diluted milk (1:9) or sulfur."),
    ("mildew", "Improve airflow, reduce humidity, apply neem oil."),
    ("leaf_mold", "Lower humidity, ventilate, remove infected leaves, copper spray."),
    ("bacterial", "Remove infected parts, avoid splashing water, copper-based bactericide."),
    ("spot", "Remove spotted leaves, water at base, apply neem oil or copper fungicide."),
    ("scab", "Remove fallen leaves, prune for airflow, apply sulfur fungicide."),
    ("rot", "Remove affected parts, reduce watering, improve drainage."),
    ("mite", "Spray neem oil / soapy water on leaf undersides, raise humidity."),
    ("virus", "No cure. Remove and destroy plant, control whiteflies/aphids, use clean seed."),
    ("curl", "Likely whitefly-borne virus: remove plant, use yellow sticky traps, neem spray."),
    ("greening", "No cure. Remove tree/plant, control psyllids."),
    ("scorch", "Water consistently, mulch, add potassium."),
]


def treatment_for(label):
    low = label.lower()
    for key, tip in TREATMENTS:
        if key in low:
            return tip
    return "Isolate the plant, remove affected leaves and consult a local agri-extension officer."


def _load():
    global _model, _classes
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError("Leaf model missing. Run 03_train.py first.")
        import torch, torch.nn as nn
        from torchvision import models
        ck = torch.load(MODEL_PATH, map_location="cpu")
        _classes = ck["classes"]
        m = models.mobilenet_v2(weights=None)
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, len(_classes)))
        m.load_state_dict(ck["state"]); m.eval()
        _model = m
    return _model, _classes


@bp.route("/predict_leaf", methods=["POST"])
def predict_leaf():
    f = request.files.get("image")
    if not f:
        return jsonify({"error": "No image uploaded (field name 'image')"}), 400
    try:
        import torch
        from PIL import Image
        from torchvision import transforms as T
        model, classes = _load()
        tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        x = tf(Image.open(io.BytesIO(f.read())).convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            p = torch.softmax(model(x), 1)[0].numpy()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Could not process image: {e}"}), 400

    idx = np.argsort(p)[-3:][::-1]
    preds = []
    for i in idx:
        crop, _, disease = classes[i].partition("___")
        preds.append({"label": classes[i], "crop": crop.replace("_", " "),
                      "disease": disease.replace("_", " ") or "healthy",
                      "confidence": round(float(p[i]) * 100, 1)})
    best = preds[0]
    soil = _get_sensor()
    soil_hint = None
    if soil and "healthy" not in best["label"].lower():
        risks = soil_disease_risks(soil)[:2]
        soil_hint = [f"{r['disease']} ({r['risk']}%): {r['why']}" for r in risks] or None
    return jsonify({
        "predictions": preds,
        "is_healthy": "healthy" in best["label"].lower(),
        "low_confidence": best["confidence"] < 60,
        "treatment": treatment_for(best["label"]),
        "soil_factors": soil_hint,
    })