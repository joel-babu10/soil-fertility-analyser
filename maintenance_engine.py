import json

# Load crop profiles
with open("crop_profiles.json") as f:
    crop_profiles = json.load(f)

# Fertilizer reference
fertilizer_data = {
    "N": {"name": "Urea", "percent": 0.46, "efficiency": 0.7},
    "P": {"name": "DAP", "percent": 0.46, "efficiency": 0.5},
    "K": {"name": "MOP", "percent": 0.60, "efficiency": 0.6}
}

def check_deficiency(ideal, current):
    tolerance = 0.15
    lower = ideal * (1 - tolerance)
    upper = ideal * (1 + tolerance)

    if current < lower:
        return "Low"
    elif current > upper:
        return "High"
    else:
        return "Optimal"


def calculate_fertilizer(nutrient, deficit):

    fert = fertilizer_data[nutrient]

    corrected_deficit = deficit / fert["efficiency"]
    fertilizer_required = corrected_deficit / fert["percent"]

    return fert["name"], round(fertilizer_required, 2)


def monitor_crop(crop, current_data):

    profile = crop_profiles[crop]
    result = {}

    for nutrient in ["N", "P", "K"]:

        status = check_deficiency(profile[nutrient], current_data[nutrient])

        if status == "Low":

            deficit = profile[nutrient] - current_data[nutrient]
            fert_name, amount = calculate_fertilizer(nutrient, deficit)

            result[nutrient] = {
                "status": "Deficient",
                "fertilizer": fert_name,
                "recommended_kg_per_ha": amount
            }

        else:
            result[nutrient] = {"status": status}

    # Moisture check
    if current_data["moisture"] < 0.4:
        result["moisture"] = "Low - Irrigate"
    elif current_data["moisture"] > 0.8:
        result["moisture"] = "High - Reduce irrigation"
    else:
        result["moisture"] = "Optimal"

    # Temperature check
    ideal_temp = profile["temperature"]

    if current_data["temperature"] > ideal_temp + 5:
        result["temperature"] = "Heat Stress Risk"
    elif current_data["temperature"] < ideal_temp - 5:
        result["temperature"] = "Cold Stress Risk"
    else:
        result["temperature"] = "Normal"

    return result