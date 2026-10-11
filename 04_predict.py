r"""Step 4: test the trained model (no web page needed).
  python 04_predict.py leaf.jpg              # one photo
  python 04_predict.py "C:\photos\my_leaves"  # a whole folder -> table + results.csv
If the folder has class subfolders (e.g. Tomato_Late_blight/), it also reports accuracy.
"""
import os, sys, csv, collections, torch, torch.nn as nn
from PIL import Image
from torchvision import models, transforms as T
from config import CLASSES, MODEL_DIR, IMG_EXT, canon

TREAT = {
    "Bacterial_spot": "Remove infected leaves, avoid wetting foliage, copper-based bactericide.",
    "Early_blight": "Remove lower infected leaves, mulch, copper fungicide, rotate crops.",
    "Late_blight": "Remove infected leaves/plants fast, avoid overhead watering, copper fungicide.",
    "Leaf_Mold": "Lower humidity, improve ventilation, remove infected leaves, copper spray.",
    "Mosaic_virus": "No cure. Remove and destroy plant, disinfect tools, control aphids.",
    "Septoria_leaf_spot": "Remove spotted leaves, water at the base, copper fungicide, mulch.",
    "Spider_mites": "Spray neem oil / soapy water under leaves, raise humidity.",
    "Target_Spot": "Remove infected leaves, improve airflow, copper or chlorothalonil fungicide.",
    "Yellow_Leaf_Curl_Virus": "No cure. Remove plant, control whiteflies (yellow sticky traps, neem).",
    "healthy": "Looks healthy. Keep regular watering and monthly compost.",
}
LOW = 60.0  # % below this = "not sure"

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ck = torch.load(os.path.join(MODEL_DIR, "leaf_model.pt"), map_location=dev)
classes = ck["classes"]
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, len(classes)))
model.load_state_dict(ck["state"]); model.to(dev).eval()
tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def predict(path):
    x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(dev)
    with torch.no_grad(): p = torch.softmax(model(x), 1)[0].cpu()
    v, i = p.topk(3)
    return [(classes[j], float(c) * 100) for c, j in zip(v, i)]

def short(c): return c.split("___")[-1]

target = sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__)
if os.path.isfile(target):
    top = predict(target)
    print(f"\n{os.path.basename(target)}")
    for c, conf in top: print(f"  {short(c):28s} {conf:5.1f}%")
    print("\nTreatment:", TREAT.get(short(top[0][0]), "-"))
    if top[0][1] < LOW: print(f"Low confidence (<{LOW:.0f}%): retake photo - single leaf, good light, plain background.")
    sys.exit()

files = [os.path.join(dp, f) for dp, _, fn in os.walk(target) for f in fn if f.lower().endswith(IMG_EXT)]
assert files, f"No images found in {target}"
rows, ok, n_lab, per = [], 0, 0, collections.defaultdict(lambda: [0, 0])
for f in sorted(files):
    top = predict(f); pred, conf = top[0]
    true = canon(os.path.basename(os.path.dirname(f)))
    mark = ""
    if true:
        n_lab += 1; per[true][1] += 1
        if pred == true: ok += 1; per[true][0] += 1; mark = "OK"
        else: mark = "WRONG"
    rows.append([f, true or "", pred, f"{conf:.1f}", "LOW" if conf < LOW else "", mark])
    print(f"{os.path.basename(f)[:34]:34s} -> {short(pred):24s} {conf:5.1f}% {'(low) ' if conf < LOW else ''}{mark}")
with open("results.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["file", "true", "predicted", "confidence", "low", "result"]); w.writerows(rows)
print(f"\n{len(files)} images -> results.csv   (low-confidence: {sum(1 for r in rows if r[4])})")
if n_lab:
    print(f"\nAccuracy on labelled images: {ok}/{n_lab} = {ok/n_lab*100:.1f}%")
    for c in sorted(per): print(f"  {short(c):28s} {per[c][0]}/{per[c][1]}")
