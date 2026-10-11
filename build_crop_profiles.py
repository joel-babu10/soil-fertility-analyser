import pandas as pd
import json

# Load dataset
df = pd.read_csv("Crop_recommendation.csv")

# Group by crop
grouped = df.groupby("label").mean()

crop_profiles = {}

for crop in grouped.index:
    crop_profiles[crop] = {
        "N": grouped.loc[crop, "N"],
        "P": grouped.loc[crop, "P"],
        "K": grouped.loc[crop, "K"],
        "ph": grouped.loc[crop, "ph"],
        "temperature": grouped.loc[crop, "temperature"]
    }

# Save as JSON
with open("crop_profiles.json", "w") as f:
    json.dump(crop_profiles, f, indent=4)

print("Crop nutrient profiles generated successfully!")