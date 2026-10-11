"""Step 8: test viewer window (built-in tkinter, no extra install).   python 08_viewer.py
 - Run test: samples unseen test images from your dataset (clean or 'hard' field-like) and shows every
   photo as a card: green = correct, red = wrong, amber = correct but low confidence.
 - Open photos / Open folder: predict your own images (folder names like Tomato_Late_blight = scored).
 - Click any card for a big view with the top-3 predictions and a treatment tip.
Run it from the leaf-ai folder (it needs models/leaf_model.pt and manifest.csv).
"""
import os, sys, threading, queue, importlib.util, collections
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
from config import CLASSES, IMG_EXT, canon
spec = importlib.util.spec_from_file_location("qt", os.path.join(HERE, "07_quick_test.py"))
qt = importlib.util.module_from_spec(spec); spec.loader.exec_module(qt)

BG, PANEL, BORDER, TXT, MUTED = "#080d08", "#111a11", "#1f331f", "#e2ffe2", "#6a9a6a"
GREEN, RED, AMBER, TEAL = "#3ddc84", "#f87171", "#f59e0b", "#2dd4bf"
LOW, THUMB = qt.LOW, 150
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
short = lambda c: c.split("___")[-1].replace("_", " ")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Tomato Leaf Disease - Test Viewer"); self.geometry("1320x820"); self.configure(bg=BG)
        self.predict = None; self.results = []; self.q = queue.Queue(); self.busy = True; self.photos = []
        st = ttk.Style(self); st.theme_use("clam")
        st.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=TXT, rowheight=24, borderwidth=0)
        st.configure("Treeview.Heading", background=BORDER, foreground=TXT, borderwidth=0)
        st.configure("TCombobox", fieldbackground=PANEL, background=BORDER, foreground=TXT, arrowcolor=TXT, bordercolor=BORDER)
        st.map("TCombobox", fieldbackground=[("readonly", PANEL)], foreground=[("readonly", TXT)],
               selectbackground=[("readonly", PANEL)], selectforeground=[("readonly", TXT)])
        self.option_add("*TCombobox*Listbox.background", PANEL); self.option_add("*TCombobox*Listbox.foreground", TXT)
        self.option_add("*TCombobox*Listbox.selectBackground", GREEN); self.option_add("*TCombobox*Listbox.selectForeground", "#021002")
        st.configure("Horizontal.TProgressbar", troughcolor=PANEL, background=GREEN)
        self.mode, self.n = tk.StringVar(value="clean"), tk.StringVar(value="30")
        self.flt, self.cls = tk.StringVar(value="All"), tk.StringVar(value="All classes")
        self._ui(); self.after(100, self._poll)
        threading.Thread(target=self._load, daemon=True).start()

    # ---------- layout ----------
    def _btn(self, parent, text, cmd, accent=False):
        return tk.Button(parent, text=text, command=cmd, bg=GREEN if accent else BORDER, fg="#021002" if accent else TXT,
                         activebackground=TEAL, relief="flat", padx=12, pady=5, font=("Segoe UI", 10, "bold"), cursor="hand2")

    def _ui(self):
        bar = tk.Frame(self, bg=BG); bar.pack(fill="x", padx=14, pady=(12, 6))
        tk.Label(bar, text="Per class", bg=BG, fg=MUTED).pack(side="left")
        tk.Spinbox(bar, from_=5, to=200, width=4, textvariable=self.n, bg=PANEL, fg=TXT, relief="flat").pack(side="left", padx=(4, 12))
        ttk.Combobox(bar, textvariable=self.mode, values=["clean", "hard"], width=7, state="readonly").pack(side="left")
        self.run_b = self._btn(bar, "Run test", self.run_test, True); self.run_b.pack(side="left", padx=8)
        self._btn(bar, "Open photos...", self.open_photos).pack(side="left", padx=2)
        self._btn(bar, "Open folder...", self.open_folder).pack(side="left", padx=2)
        ttk.Combobox(bar, textvariable=self.cls, values=["All classes"] + [short(c) for c in CLASSES], width=22, state="readonly").pack(side="right")
        ttk.Combobox(bar, textvariable=self.flt, values=["All", "Wrong only", "Low confidence", "Correct only"], width=14, state="readonly").pack(side="right", padx=6)
        tk.Label(bar, text="Show", bg=BG, fg=MUTED).pack(side="right")
        for v in (self.flt, self.cls): v.trace_add("write", lambda *a: self.render())

        self.pb = ttk.Progressbar(self, mode="determinate"); self.pb.pack(fill="x", padx=14)
        self.status = tk.Label(self, text="Loading model...", bg=BG, fg=MUTED, anchor="w"); self.status.pack(fill="x", padx=14, pady=(2, 6))

        body = tk.Frame(self, bg=BG); body.pack(fill="both", expand=True, padx=14, pady=(0, 12))
        side = tk.Frame(body, bg=PANEL, width=300); side.pack(side="left", fill="y"); side.pack_propagate(False)
        self.big = tk.Label(side, text="-", bg=PANEL, fg=GREEN, font=("Segoe UI", 34, "bold")); self.big.pack(pady=(16, 0))
        self.sub = tk.Label(side, text="overall accuracy", bg=PANEL, fg=MUTED); self.sub.pack()
        self.tree = ttk.Treeview(side, columns=("c", "ok", "n", "acc"), show="headings", height=12)
        for k, t, w in (("c", "Class", 130), ("ok", "OK", 36), ("n", "N", 36), ("acc", "Acc", 56)):
            self.tree.heading(k, text=t); self.tree.column(k, width=w, anchor="w" if k == "c" else "e")
        self.tree.pack(fill="x", padx=8, pady=12)
        for t, c in (("good", GREEN), ("mid", AMBER), ("bad", RED)): self.tree.tag_configure(t, foreground=c)
        self.conf = tk.Label(side, text="", bg=PANEL, fg=TXT, justify="left", anchor="nw", wraplength=280); self.conf.pack(fill="x", padx=10)

        right = tk.Frame(body, bg=BG); right.pack(side="left", fill="both", expand=True, padx=(12, 0))
        self.cv = tk.Canvas(right, bg=BG, highlightthickness=0); sb = tk.Scrollbar(right, command=self.cv.yview)
        self.cv.configure(yscrollcommand=sb.set); sb.pack(side="right", fill="y"); self.cv.pack(side="left", fill="both", expand=True)
        self.grid_f = tk.Frame(self.cv, bg=BG); self.win = self.cv.create_window((0, 0), window=self.grid_f, anchor="nw")
        self.grid_f.bind("<Configure>", lambda e: self.cv.configure(scrollregion=self.cv.bbox("all")))
        self.cv.bind("<Configure>", lambda e: self._resize(e))
        self.bind_all("<MouseWheel>", lambda e: self.cv.yview_scroll(int(-e.delta / 120), "units"))
        self._last_w = 0

    def _resize(self, e):
        self.cv.itemconfig(self.win, width=e.width)
        if abs(e.width - self._last_w) > 80 and self.results: self._last_w = e.width; self.render()

    # ---------- model + jobs ----------
    def _load(self):
        try: self.predict = qt.load_model(); self.q.put(("ready",))
        except Exception as e: self.q.put(("error", f"Could not load model: {e}\nRun 03_train.py first."))

    def _start(self, items):
        if self.busy or not items: return
        self.busy = True; self.run_b.config(state="disabled"); self.pb["value"] = 0
        threading.Thread(target=self._worker, args=(items,), daemon=True).start()

    def _worker(self, items):
        res = []
        try:
            for i, (p, true) in enumerate(items):
                top = self.predict.topk(p, 3)
                th = Image.open(p).convert("RGB"); th.thumbnail((THUMB, THUMB))
                res.append({"path": p, "true": true, "top": top, "thumb": th})
                self.q.put(("prog", i + 1, len(items)))
            self.q.put(("done", res))
        except Exception as e: self.q.put(("error", str(e)))

    def run_test(self):
        if self.busy: return
        try: n = max(1, int(self.n.get()))
        except ValueError: return messagebox.showerror("Per class", "Enter a number")
        hard = self.mode.get() == "hard"
        self.status.config(text=f"Building {'hard' if hard else 'clean'} test set..."); self.busy = True; self.update_idletasks(); self.busy = False
        try: items = qt.sample_test(n); paths = qt.build(items, "test_hard" if hard else "test_clean", hard)
        except SystemExit as e: return messagebox.showerror("Missing data", str(e))
        self._start(paths)

    def open_photos(self):
        fs = filedialog.askopenfilenames(title="Choose leaf photos", filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if fs: self._start([(f, canon(os.path.basename(os.path.dirname(f)))) for f in fs]); self._single = len(fs) == 1

    def open_folder(self):
        d = filedialog.askdirectory(title="Choose a folder of photos (subfolders like Tomato_Late_blight = scored)")
        if not d: return
        fs = [os.path.join(dp, f) for dp, _, fn in os.walk(d) for f in sorted(fn) if f.lower().endswith(IMG_EXT)]
        if not fs: return messagebox.showinfo("Empty", "No images found")
        self._single = False; self._start([(f, canon(os.path.basename(os.path.dirname(f)))) for f in fs])

    def _poll(self):
        try:
            while True:
                m = self.q.get_nowait()
                if m[0] == "ready": self.busy = False; self.status.config(text="Model ready. Press Run test or open your own photos.")
                elif m[0] == "prog": self.pb["value"] = m[1] / m[2] * 100; self.status.config(text=f"Predicting {m[1]}/{m[2]}...")
                elif m[0] == "done":
                    self.results = m[1]; self.busy = False; self.run_b.config(state="normal"); self.summary(); self.render()
                    if len(self.results) == 1 and getattr(self, "_single", False): self.detail(self.results[0])
                elif m[0] == "error": self.busy = False; self.run_b.config(state="normal"); self.status.config(text="Error"); messagebox.showerror("Error", m[1])
        except queue.Empty: pass
        self.after(100, self._poll)

    # ---------- display ----------
    def verdict(self, r):
        pred, conf = r["top"][0]
        if r["true"] is None: return "unknown", AMBER if conf < LOW else TEAL
        if pred != r["true"]: return "wrong", RED
        return ("low", AMBER) if conf < LOW else ("ok", GREEN)

    def summary(self):
        per = collections.defaultdict(lambda: [0, 0]); conf = collections.Counter(); lows = 0
        for r in self.results:
            lows += r["top"][0][1] < LOW
            if r["true"] is None: continue
            per[r["true"]][1] += 1; ok = r["top"][0][0] == r["true"]; per[r["true"]][0] += ok
            if not ok: conf[(r["true"], r["top"][0][0])] += 1
        self.tree.delete(*self.tree.get_children())
        for c in CLASSES:
            if c in per:
                o, n = per[c]; a = o / n * 100
                self.tree.insert("", "end", values=(short(c), o, n, f"{a:.0f}%"), tags=("good" if a >= 95 else "mid" if a >= 80 else "bad",))
        tot = sum(v[1] for v in per.values()); ok = sum(v[0] for v in per.values())
        if tot: self.big.config(text=f"{ok / tot * 100:.1f}%", fg=GREEN if ok / tot >= .95 else AMBER if ok / tot >= .8 else RED); self.sub.config(text=f"{ok}/{tot} correct")
        else: self.big.config(text="-", fg=MUTED); self.sub.config(text="no labelled images (unscored)")
        self.conf.config(text=(f"Low confidence (<{LOW:.0f}%): {lows}\n\nTop mix-ups:\n" +
                               "\n".join(f"{short(t)} -> {short(p)}: {v}x" for (t, p), v in conf.most_common(6))) if self.results else "")
        self.status.config(text=f"{len(self.results)} images. Click a card for details.")

    def render(self):
        for w in self.grid_f.winfo_children(): w.destroy()
        self.photos = []
        f, c = self.flt.get(), self.cls.get()
        shown = []
        for r in self.results:
            v, _ = self.verdict(r)
            if c != "All classes" and short(r["true"] or r["top"][0][0]) != c: continue
            if f == "Wrong only" and v != "wrong": continue
            if f == "Correct only" and v not in ("ok", "low"): continue
            if f == "Low confidence" and r["top"][0][1] >= LOW: continue
            shown.append(r)
        cols = max(1, (self.cv.winfo_width() or 900) // (THUMB + 18))
        for i, r in enumerate(shown[:600]):
            v, col = self.verdict(r); pred, conf = r["top"][0]
            card = tk.Frame(self.grid_f, bg=PANEL, highlightthickness=3, highlightbackground=col, cursor="hand2")
            card.grid(row=i // cols, column=i % cols, padx=5, pady=5)
            ph = ImageTk.PhotoImage(r["thumb"]); self.photos.append(ph)
            lab = [tk.Label(card, image=ph, bg=PANEL)]
            lab.append(tk.Label(card, text=f"{short(pred)}  {conf:.0f}%", bg=PANEL, fg=col, font=("Segoe UI", 9, "bold"), wraplength=THUMB))
            if r["true"] and v == "wrong": lab.append(tk.Label(card, text=f"true: {short(r['true'])}", bg=PANEL, fg=MUTED, font=("Segoe UI", 8), wraplength=THUMB))
            for l in lab: l.pack(); l.bind("<Button-1>", lambda e, r=r: self.detail(r))
            card.bind("<Button-1>", lambda e, r=r: self.detail(r))
        if len(shown) > 600: self.status.config(text=f"Showing first 600 of {len(shown)} (use the filters).")
        self.cv.yview_moveto(0)

    def detail(self, r):
        w = tk.Toplevel(self); w.title(os.path.basename(r["path"])); w.configure(bg=BG); w.geometry("+160+60")
        img = Image.open(r["path"]).convert("RGB"); img.thumbnail((520, 520)); ph = ImageTk.PhotoImage(img)
        l = tk.Label(w, image=ph, bg=BG); l.image = ph; l.pack(side="left", padx=14, pady=14)
        p = tk.Frame(w, bg=BG); p.pack(side="left", fill="both", padx=(0, 16), pady=14)
        v, col = self.verdict(r); pred, conf = r["top"][0]
        tk.Label(p, text=short(pred), bg=BG, fg=col, font=("Segoe UI", 22, "bold")).pack(anchor="w")
        msg = {"ok": "Correct", "low": "Correct but low confidence", "wrong": f"WRONG - true class: {short(r['true'] or '')}", "unknown": "Not scored (no label)"}[v]
        tk.Label(p, text=msg, bg=BG, fg=col).pack(anchor="w", pady=(0, 10))
        for c, pc in r["top"]:
            row = tk.Frame(p, bg=BG); row.pack(fill="x", pady=2)
            tk.Label(row, text=short(c), bg=BG, fg=TXT, width=24, anchor="w").pack(side="left")
            cv = tk.Canvas(row, width=170, height=12, bg=PANEL, highlightthickness=0); cv.pack(side="left")
            cv.create_rectangle(0, 0, 170 * pc / 100, 12, fill=GREEN if c == pred else MUTED, width=0)
            tk.Label(row, text=f"{pc:.1f}%", bg=BG, fg=TXT, width=7).pack(side="left")
        tk.Label(p, text="Treatment", bg=BG, fg=MUTED).pack(anchor="w", pady=(16, 2))
        tk.Label(p, text=TREAT.get(pred.split("___")[-1], "-"), bg=BG, fg=TXT, wraplength=330, justify="left").pack(anchor="w")
        if conf < LOW: tk.Label(p, text="Low confidence: retake the photo (single leaf, good light, plain background).", bg=BG, fg=AMBER, wraplength=330, justify="left").pack(anchor="w", pady=10)
        tk.Label(p, text=r["path"], bg=BG, fg=MUTED, font=("Segoe UI", 8), wraplength=330, justify="left").pack(anchor="w", pady=(14, 0))


if __name__ == "__main__":
    App().mainloop()
