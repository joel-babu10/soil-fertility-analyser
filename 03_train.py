"""Step 3: train MobileNetV2 (PyTorch, GPU) on manifest.csv.  python 03_train.py"""
import os, csv, json, random
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms as T
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from config import *

BS, FIELD_REPEAT = 32, 4      # real-world photos are rare -> repeat them in training
EPOCHS_HEAD, EPOCHS_FINE = 5, 8
NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
TRAIN_T = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                     T.RandomRotation(25), T.ColorJitter(0.35, 0.35, 0.25, 0.03), T.ToTensor(), NORM])
EVAL_T = T.Compose([T.Resize((224, 224)), T.ToTensor(), NORM])
CID = {c: i for i, c in enumerate(CLASSES)}

class DS(Dataset):
    def __init__(s, rows, tf): s.r, s.tf = rows, tf
    def __len__(s): return len(s.r)
    def __getitem__(s, i):
        p, y = s.r[i]; return s.tf(Image.open(p).convert("RGB")), y

def epoch(model, dl, dev, opt=None, lossf=None, scaler=None):
    tr = opt is not None; model.train(tr); n = ok = 0; ls = 0.0; preds = []
    for x, y in dl:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.set_grad_enabled(tr), torch.autocast(dev.type, enabled=dev.type == "cuda"):
            out = model(x); loss = lossf(out, y) if lossf else None
        if tr:
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if loss is not None: ls += loss.item() * len(y)
        p = out.argmax(1); ok += (p == y).sum().item(); n += len(y); preds += p.cpu().tolist()
    return ls / n, ok / n, preds

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "(NO GPU - slow!)")

    sp = {k: [] for k in ("train", "val", "test_lab", "test_field")}
    for r in csv.DictReader(open(MANIFEST)):
        item = (r["path"], CID[r["label"]])
        sp[r["split"]] += [item] * (FIELD_REPEAT if r["split"] == "train" and r["source"] == "field" else 1)
    print({k: len(v) for k, v in sp.items()})

    mk = lambda rows, tf, sh: DataLoader(DS(rows, tf), BS, shuffle=sh, num_workers=4, pin_memory=True, persistent_workers=True)
    tdl, vdl = mk(sp["train"], TRAIN_T, True), mk(sp["val"], EVAL_T, False)

    cnt = np.bincount([y for _, y in sp["train"]], minlength=len(CLASSES)).clip(min=1)
    lossf = nn.CrossEntropyLoss(weight=torch.tensor(len(sp["train"]) / (len(CLASSES) * cnt), dtype=torch.float32, device=dev))

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, len(CLASSES)))
    model.to(dev); scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")
    best = 0.0; ckpt = os.path.join(MODEL_DIR, "leaf_model.pt")

    def stage(name, epochs, lr, freeze):
        nonlocal best
        print(name)
        for p in model.features.parameters(): p.requires_grad = not freeze
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
        for e in range(epochs):
            _, ta, _ = epoch(model, tdl, dev, opt, lossf, scaler)
            vl, va, _ = epoch(model, vdl, dev, lossf=lossf)
            print(f"  epoch {e+1}/{epochs}  train acc {ta:.3f}  val acc {va:.3f}  val loss {vl:.3f}")
            if va > best: best = va; torch.save({"state": model.state_dict(), "classes": CLASSES}, ckpt)
    stage("Stage 1: new classifier head", EPOCHS_HEAD, 1e-3, True)
    stage("Stage 2: fine-tune all layers", EPOCHS_FINE, 1e-4, False)

    model.load_state_dict(torch.load(ckpt, map_location=dev)["state"])
    json.dump(CLASSES, open(os.path.join(MODEL_DIR, "leaf_classes.json"), "w"))
    report = []
    for name in ("test_lab", "test_field"):
        rows = sp[name]
        if not rows: report.append(f"\n== {name}: no images =="); continue
        _, acc, pred = epoch(model, mk(rows, EVAL_T, False), dev)
        y = [r[1] for r in rows]; labs = sorted(set(y) | set(pred))
        report.append(f"\n== {name}  ({len(rows)} images)  accuracy {acc:.4f} ==\n" +
                      classification_report(y, pred, labels=labs, target_names=[CLASSES[i] for i in labs], digits=3, zero_division=0) +
                      "\n" + str(confusion_matrix(y, pred, labels=labs)))
    txt = "\n".join(report); print(txt)
    open(os.path.join(MODEL_DIR, "metrics.txt"), "w").write(txt)
    print("\nSaved models/leaf_model.pt, leaf_classes.json, metrics.txt")
    print("test_lab = clean PlantVillage photos (easy).  test_field = PlantDoc real photos (the number that matters).")

if __name__ == "__main__":
    main()
