# advanced_monitoring/advanced_engine.py

import json
from advanced_monitoring.history_manager import get_last_readings

with open("crop_profiles.json") as f:
    crop_profiles = json.load(f)

fertilizer_map = {
    "N": "Urea",
    "P": "DAP",
    "K": "MOP"
}

def deficiency_severity(ideal, readings):

    avg_value = sum(readings) / len(readings)
    deficit_percent = ((ideal - avg_value) / ideal) * 100

    if deficit_percent < 20:
        return "Mild"
    elif deficit_percent < 40:
        return "Moderate"
    else:
        return "Severe"

def advanced_monitor(crop, tolerance=0.15):

    profile = crop_profiles[crop]
    result = {}

    last_readings = get_last_readings(3)

    if len(last_readings) < 3:
        return {"status": "Insufficient Data"}

    for nutrient in ["N", "P", "K"]:

        ideal = profile[nutrient]
        lower = ideal * (1 - tolerance)

        nutrient_values = [r[nutrient] for r in last_readings]

        if all(value < lower for value in nutrient_values):

            severity = deficiency_severity(ideal, nutrient_values)

            result[nutrient] = {
                "status": "Persistent Deficiency",
                "severity": severity,
                "recommended_fertilizer": fertilizer_map[nutrient],
                "application_strategy":
                    "Apply in split doses over 2–3 weeks" 
                    if nutrient == "N"
                    else "Apply as basal dose"
            }

        else:
            result[nutrient] = {
                "status": "Stable"
            }

    return result