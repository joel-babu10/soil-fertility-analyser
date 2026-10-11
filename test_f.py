r"""Step 9: make a test folder with N images per class taken from your dataset folder.
  python 09_make_test_folder.py "C:\path\to\dataset"            # 10 per class -> test_folder/
  python 09_make_test_folder.py "C:\path\to\dataset" --n 20 --out my_test --seed 7
If manifest.csv exists, images the model trained on (train/val) are skipped, so the samples are unseen.
Then:  python 08_viewer.py test_folder
"""
import os, sys, csv, random, shutil, collections
from config import CLASSES, IMG_EXT, MANIFEST, canon

args = sys.argv[1:]
if not args or args[0].startswith("--"): sys.exit(__doc__)
src = args[0]
def opt(flag, default, cast=str): return cast(args[args.index(flag) + 1]) if flag in args else default
n, out, seed = opt("--n", 10, int), opt("--out", "test_folder"), opt("--seed", 42, int)
assert os.path.isdir(src), f"Folder not found: {src}"

norm = lambda p: os.path.normcase(os.path.abspath(p))
used = set()
if os.path.exists(MANIFEST):
    used = {norm(r["path"]) for r in csv.DictReader(open(MANIFEST)) if r["split"] in ("train", "val")}
    print(f"manifest.csv found: skipping {len(used)} images used for training/validation")
else:
    print("NOTE: manifest.csv not found - samples may include images the model trained on (accuracy will look higher).")

pool = collections.defaultdict(list); skipped = []
for dp, dn, fn in os.walk(src):
    low = dp.lower()
    if "grayscale" in low or "segmented" in low: dn[:] = []; continue
    imgs = [f for f in fn if f.lower().endswith(IMG_EXT)]
    if not imgs or not os.path.basename(dp).lower().startswith("tomato"): continue
    c = canon(os.path.basename(dp))
    if not c: skipped.append(os.path.basename(dp)); continue
    pool[c] += [os.path.join(dp, f) for f in imgs if norm(os.path.join(dp, f)) not in used]

shutil.rmtree(out, ignore_errors=True); total = 0
for c in CLASSES:
    ps = sorted(pool.get(c, [])); random.Random(seed).shuffle(ps); pick = ps[:n]
    d = os.path.join(out, c.replace("___", "_")); os.makedirs(d, exist_ok=True)
    for p in pick: shutil.copy2(p, os.path.join(d, os.path.basename(p)))
    total += len(pick)
    print(f"  {c.replace('___', '_'):38s} {len(pick):3d}" + ("   <-- fewer than requested" if len(pick) < n else ""))
if skipped: print("Skipped unmapped folders:", sorted(set(skipped)))
print(f"\nCreated {os.path.abspath(out)}  ({total} images)\nTest it:  python 08_viewer.py {out}")