"""Shared settings + folder-name -> canonical class mapping."""
import re

SEED = 42
IMG_EXT = (".jpg", ".jpeg", ".png")
MODEL_DIR = "models"
MANIFEST = "manifest.csv"

CLASSES = [
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Mosaic_virus", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites", "Tomato___Target_Spot",
    "Tomato___Yellow_Leaf_Curl_Virus", "Tomato___healthy",
]

# keyword (prefix of a word in the folder name) -> class. Handles PlantVillage (GitHub + Kaggle) and PlantDoc names.
RULES = [("bacterial", "Bacterial_spot"), ("septoria", "Septoria_leaf_spot"), ("early", "Early_blight"),
         ("late", "Late_blight"), ("mold", "Leaf_Mold"), ("mould", "Leaf_Mold"), ("mite", "Spider_mites"),
         ("target", "Target_Spot"), ("yellow", "Yellow_Leaf_Curl_Virus"), ("curl", "Yellow_Leaf_Curl_Virus"),
         ("mosaic", "Mosaic_virus"), ("healthy", "healthy")]


def canon(folder_name):
    words = re.findall(r"[a-z]+", folder_name.lower())
    for key, label in RULES:
        if any(w.startswith(key) for w in words):
            return "Tomato___" + label
    if words and set(words) <= {"tomato", "leaf", "leaves"}:  # PlantDoc: plain "Tomato leaf" = healthy
        return "Tomato___healthy"
    return None
