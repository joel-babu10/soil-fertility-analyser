"""Step 7: one-command self-test. Builds a test set from the images the model NEVER saw in training
(the 'test_lab' split of manifest.csv), runs predictions, prints accuracy.
  python 07_quick_test.py              # 30 images per class, clean + 'hard' (field-like) versions
  python 07_quick_test.py --n 50       # 50 per class
  python 07_quick_test.py --no-hard    # clean only
Creates test_clean/ and test_hard/ (so you can look at the images) and quick_test_results.csv.
'hard' = the same photos with random blur, dark/bright light, rotation, zoom, noise and JPEG damage,
a rough stress test for messy real-world photos. It does NOT replace real garden photos.
"""
import os, io, sys, csv, random, shutil, collections
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from config import CLASSES, MANIFEST, MODEL_DIR, SEED

LOW = 60.0
def folder(c): return c.replace("___", "_")
def short(c): return c.split("___")[-1]

def sample_test(n):
    if not os.path.exists(MANIFEST): sys.exit("manifest.csv missing - run 02_prepare_data.py first")
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(MANIFEST)):
        if r["split"] == "test_lab": by[r["label"]].append(r["path"])
    out = []
    for c in CLASSES:
        ps = sorted(by.get(c, [])); random.Random(SEED).shuffle(ps)
        out += [(p, c) for p in ps[:n]]
    return out

def degrade(img, rng):
    """Apply 3-4 random real-world-like degradations."""
    ops = ["rotate", "light", "blur", "zoom", "noise", "jpeg", "contrast"]
    for op in rng.sample(ops, rng.choice([3, 4])):
        if op == "rotate": img = img.rotate(rng.uniform(-35, 35), resample=Image.BILINEAR, fillcolor=(rng.randint(30, 200),) * 3)
        elif op == "light": img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.45, 1.5))
        elif op == "contrast": img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.6, 1.4))
        elif op == "blur": img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.8, 2.2)))
        elif op == "zoom":
            w, h = img.size; s = rng.uniform(0.6, 0.9); cw, ch = int(w * s), int(h * s)
            x, y = rng.randint(0, w - cw), rng.randint(0, h - ch); img = img.crop((x, y, x + cw, y + ch)).resize((w, h))
        elif op == "noise":
            a = np.asarray(img, dtype=np.float32) + np.random.RandomState(rng.randint(0, 10**6)).normal(0, rng.uniform(8, 22), (img.size[1], img.size[0], 3))
            img = Image.fromarray(a.clip(0, 255).astype(np.uint8))
        elif op == "jpeg":
            b = io.BytesIO(); img.save(b, "JPEG", quality=rng.randint(15, 45)); b.seek(0); img = Image.open(b).convert("RGB")
    return img

def build(items, out, hard):
    shutil.rmtree(out, ignore_errors=True); rng = random.Random(SEED); paths = []
    for i, (p, c) in enumerate(items):
        d = os.path.join(out, folder(c)); os.makedirs(d, exist_ok=True)
        img = Image.open(p).convert("RGB")
        if hard: img = degrade(img, rng)
        dst = os.path.join(d, f"{i:04d}.jpg"); img.save(dst, quality=92); paths.append((dst, c))
    return paths

def load_model():
    import torch, torch.nn as nn
    from torchvision import models, transforms as T
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(MODEL_DIR, "leaf_model.pt"), map_location=dev)
    classes = ck["classes"]; m = models.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, len(classes)))
    m.load_state_dict(ck["state"]); m.to(dev).eval()
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    def topk(path, k=3):
        x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(dev)
        with torch.no_grad(): p = torch.softmax(m(x), 1)[0].cpu()
        v, i = p.topk(k); return [(classes[int(j)], float(c) * 100) for c, j in zip(v, i)]
    def predict(path): return topk(path, 1)[0]
    predict.topk = topk
    return predict

def evaluate(name, paths, predict):
    per = collections.defaultdict(lambda: [0, 0]); conf = collections.Counter(); rows = []; low = 0
    for p, true in paths:
        pred, c = predict(p); ok = pred == true
        per[true][1] += 1; per[true][0] += ok; low += c < LOW
        if not ok: conf[(true, pred)] += 1
        rows.append([name, p, true, pred, f"{c:.1f}", "OK" if ok else "WRONG"])
    tot = sum(v[1] for v in per.values()); right = sum(v[0] for v in per.values())
    return per, conf, rows, right / tot, low, tot

if __name__ == "__main__":
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 30
    items = sample_test(n); assert items, "No test_lab images in manifest"
    print(f"{len(items)} unseen images ({n} per class max). Loading model...")
    predict = load_model(); results = {}; allrows = []
    modes = [("clean", False)] + ([] if "--no-hard" in sys.argv else [("hard", True)])
    for name, hard in modes:
        print(f"Building test_{name}/ and predicting...")
        results[name] = evaluate(name, build(items, f"test_{name}", hard), predict); allrows += results[name][2]
    print(f"\n{'class':34s}" + "".join(f"{k:>10s}" for k in results))
    for c in CLASSES:
        print(f"{short(c):34s}" + "".join(f"{(r[0][c][0]/max(r[0][c][1],1))*100:9.1f}%" for r in results.values()))
    print(f"{'OVERALL':34s}" + "".join(f"{r[3]*100:9.1f}%" for r in results.values()))
    for k, r in results.items():
        print(f"\n[{k}] {int(r[3]*r[5])}/{r[5]} correct, {r[4]} low-confidence (<{LOW:.0f}%)")
        for (t, p), v in r[1].most_common(5): print(f"   {short(t)} -> predicted {short(p)}: {v}x")
    with open("quick_test_results.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["set", "file", "true", "predicted", "confidence", "result"]); w.writerows(allrows)
    print("\nSaved quick_test_results.csv. Look at test_hard/ to see what the degraded photos look like.")
