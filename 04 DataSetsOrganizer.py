#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import csv
import shutil
import random
from pathlib import Path
from collections import defaultdict, Counter

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

APP_TITLE = "YOLO Dataset Builder"
RANDOM_SEED = 42


IMG_EXTS = {".jpg", ".jpeg", ".png"}

IGNORED_FILENAMES = {"desktop.ini", "thumbs.db", ".ds_store"}
IGNORED_DIRNAME_TOKENS = {"__MACOSX"}


def is_hidden_or_ignored(fname: str, dirpath: str) -> bool:
    if fname.startswith("."):
        return True
    if fname.lower() in IGNORED_FILENAMES:
        return True
    if any(tok in dirpath.upper() for tok in IGNORED_DIRNAME_TOKENS):
        return True
    return False



def validate_label_file(path: Path, num_classes: int | None):
    EPS = 1e-4

    try:
        txt = path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        txt = path.read_text(encoding="latin-1").strip()
    except Exception as e:
        return False, f"label read error: {e}", None

    if txt == "":
        return False, "label file empty", None

    class_ids = []
    warnings = 0

    for i, raw in enumerate(txt.splitlines(), start=1):
        line = raw.strip()
        if line == "":
            continue

        parts = line.split()
        if len(parts) < 5:
            return False, f"line {i}: expected >=5 columns, got {len(parts)}", None


        try:
            cls = int(float(parts[0]))
            if cls < 0:
                return False, f"line {i}: negative class id", None
            if num_classes is not None and not (0 <= cls < num_classes):
                return False, f"line {i}: class id {cls} out of range [0,{num_classes-1}]", None
        except Exception:
            return False, f"line {i}: class id not an integer", None


        try:
            x, y, w, h = map(float, parts[1:5])
        except Exception:
            return False, f"line {i}: non-numeric bbox", None


        for val, name in zip([x, y, w, h], ["x", "y", "w", "h"]):
            if not (-EPS <= val <= 1.0 + EPS):
                return False, f"line {i}: {name} out of [0,1] (too far)", None


        if (x - w/2) < -EPS or (x + w/2) > 1.0 + EPS or (y - h/2) < -EPS or (y + h/2) > 1.0 + EPS:
            warnings += 1  # ale nie uznajemy za błąd

        class_ids.append(cls)

    msg = None
    if warnings > 0:
        msg = f"{warnings} bbox near border (tolerated)"
    return True, msg, class_ids



def top_level_folder(root: Path, path: Path) -> str:
    rel = path.parent.relative_to(root)
    parts = rel.parts
    return parts[0] if len(parts) > 0 else "."


def scan_source_det(root: Path, num_classes: int | None):

    temp = defaultdict(lambda: {"image": None, "label": None, "any_path": None})
    file_locations = defaultdict(set)
    class_hist = Counter()

    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if is_hidden_or_ignored(fname, dirpath):
                continue
            p = Path(dirpath) / fname
            stem = p.stem
            ext = p.suffix.lower()
            if ext not in IMG_EXTS:
                continue

            file_locations[fname].add(str(p))
            if temp[stem]["any_path"] is None:
                temp[stem]["any_path"] = str(p)
            temp[stem]["image"] = p


            if "images" in p.parts:
                try:
                    i = p.parts.index("images")
                    lbl_path = Path(*p.parts[:i], "labels", *p.parts[i + 1:]).with_suffix(".txt")
                except Exception:
                    lbl_path = p.with_suffix(".txt")
            else:
                lbl_path = p.with_suffix(".txt")

            if lbl_path.exists():
                temp[stem]["label"] = lbl_path
                file_locations[lbl_path.name].add(str(lbl_path))

    series_ok, incomplete = {}, {}
    for base, packs in temp.items():
        img, lbl = packs["image"], packs["label"]
        problems, ok = [], True
        any_p = Path(packs["any_path"]) if packs["any_path"] else root
        folder_name = top_level_folder(root, any_p)

        if img is None:
            ok = False
            problems.append("no image")
        if lbl is None:
            ok = False
            problems.append("no label DET (.txt)")
        else:
            valid, err, cls_list = validate_label_file_det(lbl, num_classes)
            if not valid:
                ok = False
                problems.append(f"DET: {err}")
            else:
                if err:

                    problems.append(f"INFO: {err}")
                class_hist.update(cls_list)

        if ok:
            series_ok[base] = {"image": img, "label": lbl, "folder": folder_name}
        else:
            incomplete[base] = {"folder": folder_name, "problems": problems}

    duplicates = {fn: sorted(paths) for fn, paths in file_locations.items() if len(paths) > 1}
    return series_ok, incomplete, duplicates, class_hist



def ensure_structure(dst_root: Path):
    for kind in ["images", "labels"]:
        for split in ["train", "valid", "test"]:
            (dst_root / "RGB" / kind / split).mkdir(parents=True, exist_ok=True)


def compute_target_counts(n_series: int, pct_train: float, pct_valid: float, pct_test: float):
    if abs((pct_train + pct_valid + pct_test) - 100.0) > 1e-6:
        raise ValueError("The sum of the percentages must be 100.")

    raw = {
        "train": n_series * pct_train / 100.0,
        "valid": n_series * pct_valid / 100.0,
        "test":  n_series * pct_test  / 100.0,
    }
    rounded = {k: int(round(v)) for k, v in raw.items()}
    delta = sum(rounded.values()) - n_series

    if delta != 0:

        order = sorted(raw.keys(), key=lambda k: abs(raw[k] - rounded[k]), reverse=True)
        i = 0
        while delta != 0 and i < 10:
            k = order[i % 3]
            rounded[k] -= 1 if delta > 0 else -1
            delta = sum(rounded.values()) - n_series
            i += 1
    return rounded


def compute_split_counts(n_series: int, pct_train: float, pct_valid: float, pct_test: float):

    if abs((pct_train + pct_valid + pct_test) - 100.0) > 1e-6:
        raise ValueError("The sum of the percentages must be 100.")
    t = int(n_series * pct_train / 100.0)
    v = int(n_series * pct_valid / 100.0)
    s = n_series - t - v
    return {"train": t, "valid": v, "test": s}


def pick_and_assign_per_folder(series_by_folder: dict, pct_train: float, pct_valid: float, pct_test: float):
    assign = {}
    for folder, bases in series_by_folder.items():
        counts = compute_split_counts(len(bases), pct_train, pct_valid, pct_test)
        shuffled = bases[:]
        rnd = random.Random(f"{RANDOM_SEED}-{folder}")
        rnd.shuffle(shuffled)
        i = 0
        for split in ["train", "valid", "test"]:
            k = counts[split]
            for _ in range(k):
                if i >= len(shuffled):
                    break
                assign[shuffled[i]] = split
                i += 1

        while i < len(shuffled):
            assign[shuffled[i]] = "train"
            i += 1
    return assign


def adjust_assign_to_exact(assign_map: dict, series_by_folder: dict, target: dict, topper_folder: str | None,
                           reduce_to_fit: bool):

    current = Counter(assign_map.values())
    bases_all = set(assign_map.keys())
    excluded = set()
    log = []

    def candidates(from_split=None, only_folder=None):
        for b in bases_all - excluded:
            if from_split is not None and assign_map[b] != from_split:
                continue
            if only_folder is not None and only_folder != folder_of(b):
                continue
            yield b


    folder_by_base = {}
    for f, items in series_by_folder.items():
        for b in items:
            folder_by_base[b] = f

    def folder_of(b):
        return folder_by_base.get(b, "?")


    total_now = sum(current.values())
    total_target = sum(target.values())
    if reduce_to_fit and total_now > total_target:
        to_drop = total_now - total_target
        log.append(f"[TRIM] Cut {to_drop} series to obtain exact proportions.")
        # najpierw z folderu do dobijania
        pref = list(candidates(only_folder=topper_folder)) if topper_folder else []
        rest = [b for b in candidates() if b not in pref]
        for b in pref + rest:
            if to_drop == 0:
                break
            excluded.add(b)
            current[assign_map[b]] -= 1
            to_drop -= 1
        if to_drop > 0:
            log.append("[WARN] Not enough series to trim - proportions may vary.")


    def move_one(b, to_split):
        frm = assign_map[b]
        if frm == to_split:
            return False
        assign_map[b] = to_split
        current[frm] -= 1
        current[to_split] += 1
        log.append(f"[MOVE] {b}: {frm} -> {to_split} (folder={folder_of(b)})")
        return True


    for split in ["train", "valid", "test"]:
        while current[split] < target[split]:

            found = False

            for b in candidates(only_folder=topper_folder):
                if b in excluded or assign_map[b] == split:
                    continue
                move_one(b, split)
                found = True
                break
            if not found:
                # dowolny inny folder
                for b in candidates():
                    if b in excluded or assign_map[b] == split:
                        continue
                    move_one(b, split)
                    found = True
                    break
            if not found:
                log.append(f"[WARN] Failed to reach target for {split}.")
                break


    for split in ["train", "valid", "test"]:
        while current[split] > target[split]:

            chosen = None

            for b in candidates(from_split=split, only_folder=topper_folder):
                if b not in excluded:
                    chosen = b
                    break
            if chosen is None:
                for b in candidates(from_split=split):
                    if b not in excluded:
                        chosen = b
                        break
            if chosen is None:
                break

            deficits = {k: target[k] - current[k] for k in target}
            to_split = max(deficits, key=deficits.get)
            if deficits[to_split] <= 0:
                break
            move_one(chosen, to_split)

    return assign_map, excluded, log


def move_series(assign_map, series, dst_root: Path, excluded: set[str]):
    ensure_structure(dst_root)
    for base, pack in series.items():
        if base in excluded:
            continue
        split = assign_map[base]
        img_src = pack["image"]
        lbl_src = pack["label"]
        img_dst = dst_root / "RGB" / "images" / split / img_src.name
        lbl_dst = dst_root / "RGB" / "labels" / split / (img_src.with_suffix(".txt").name)
        if img_dst.exists() or lbl_dst.exists():
            raise FileExistsError(f"File exists: {img_dst} or {lbl_dst}")
        shutil.move(str(img_src), str(img_dst))
        shutil.move(str(lbl_src), str(lbl_dst))


def write_split_report_csv(assign_map, series, dst_root: Path, excluded: set[str]):
    report_path = dst_root / "report_split.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["base_name", "split", "folder", "used"])
        for base in sorted(series.keys()):
            used = "no" if base in excluded else "yes"
            split = assign_map.get(base, "-")
            folder = series[base]["folder"]
            w.writerow([base, split, folder, used])
    return report_path


def write_classes_report_csv(class_hist: Counter, dst_root: Path):
    report_path = dst_root / "report_classes.csv"
    with open(report_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class_id", "count"])
        for cls, cnt in sorted(class_hist.items()):
            w.writerow([cls, cnt])
    return report_path


# -------------------- GUI --------------------

def show_scrollable_message(title: str, content: str):
    win = tk.Toplevel()
    win.title(title)
    win.geometry("900x500")
    win.transient()
    win.grab_set()

    frm = ttk.Frame(win)
    frm.pack(fill="both", expand=True)

    yscroll = ttk.Scrollbar(frm, orient="vertical")
    txt = tk.Text(
        frm,
        wrap="none",
        yscrollcommand=yscroll.set,
        bg="#1e1e1e",
        fg="#d4d4d4",
        insertbackground="#d4d4d4",
        selectbackground="#3a7afe"
    )
    yscroll.config(command=txt.yview)
    yscroll.pack(side="right", fill="y")
    txt.pack(side="left", fill="both", expand=True)

    txt.insert("1.0", content)
    txt.config(state="disabled")

    ttk.Button(win, text="OK", command=win.destroy).pack(pady=8)
    win.wait_window()

def apply_dark_theme(root: tk.Tk):
    style = ttk.Style(root)

    # wymuś motyw, który pozwala na kolory
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    bg = "#1e1e1e"
    fg = "#d4d4d4"
    field = "#2d2d2d"
    accent = "#3a7afe"

    root.configure(bg=bg)

    style.configure(".", background=bg, foreground=fg)
    style.configure("TFrame", background=bg)
    style.configure("TLabel", background=bg, foreground=fg)
    style.configure("TLabelframe", background=bg, foreground=fg)
    style.configure("TLabelframe.Label", background=bg, foreground=fg)

    style.configure(
        "TButton",
        background="#333333",
        foreground=fg,
        padding=6
    )
    style.map(
        "TButton",
        background=[("active", "#444444")]
    )

    style.configure(
        "TEntry",
        fieldbackground=field,
        background=field,
        foreground=fg,
        insertcolor=fg
    )

    style.configure(
        "TCombobox",
        fieldbackground=field,
        background=field,
        foreground=fg
    )

    style.configure(
        "TCheckbutton",
        background=bg,
        foreground=fg
    )

    # Treeview (tabele)
    style.configure(
        "Treeview",
        background=field,
        fieldbackground=field,
        foreground=fg,
        rowheight=22
    )
    style.configure(
        "Treeview.Heading",
        background="#2a2a2a",
        foreground=fg
    )
    style.map(
        "Treeview",
        background=[("selected", accent)],
        foreground=[("selected", "white")]
    )

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        apply_dark_theme(self)
        self.title(APP_TITLE)
        self.geometry("1280x820")

        self.src_path = tk.StringVar()
        self.dst_path = tk.StringVar()

        self.n_series = tk.IntVar(value=0)
        self.n_images = tk.IntVar(value=0)
        self.n_labels = tk.IntVar(value=0)

        self.pct_train = tk.DoubleVar(value=70.0)
        self.pct_valid = tk.DoubleVar(value=20.0)
        self.pct_test = tk.DoubleVar(value=10.0)

        self.num_classes = tk.StringVar(value="")  # puste = brak sprawdzania
        self.reduce_to_fit = tk.BooleanVar(value=True)
        self.topper_folder = tk.StringVar(value="")

        self.series = {}
        self.assign_map = {}
        self.series_by_folder = {}
        self.class_hist = Counter()

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        # ── Paths ───────────────────────────────────────────────────────────
        src_frame = ttk.LabelFrame(self, text="Source directory (recursive)")
        src_frame.pack(fill="x", **pad)
        ttk.Entry(src_frame, textvariable=self.src_path).pack(side="left", fill="x", expand=True, padx=6, pady=6)
        ttk.Button(src_frame, text="Select…", command=self.choose_src).pack(side="right", padx=6, pady=6)

        dst_frame = ttk.LabelFrame(self, text="Target directory (RGB structure)")
        dst_frame.pack(fill="x", **pad)
        ttk.Entry(dst_frame, textvariable=self.dst_path).pack(side="left", fill="x", expand=True, padx=6, pady=6)
        ttk.Button(dst_frame, text="Select…", command=self.choose_dst).pack(side="right", padx=6, pady=6)

        # ── Validation parameters ───────────────────────────────────────────────
        val_frame = ttk.LabelFrame(self, text="Label Validation")
        val_frame.pack(fill="x", **pad)
        ttk.Label(val_frame, text="Number of classes (optional):").grid(row=0, column=0, sticky="e")
        ttk.Entry(val_frame, width=8, textvariable=self.num_classes).grid(row=0, column=1, sticky="w")
        ttk.Label(val_frame, text="(checks whether id ∈ [0..N-1])").grid(row=0, column=2, sticky="w")

        # ── Scanning ────────────────────────────────────────────────────────
        scan_frame = ttk.Frame(self)
        scan_frame.pack(fill="x", **pad)
        ttk.Button(scan_frame, text="Scan source", command=self.scan_action).pack(side="left")
        self.summary_lbl = ttk.Label(scan_frame, text="No data available. Please scan the source first..")
        self.summary_lbl.pack(side="left", padx=12)

        # ── Percentages and scoring ─────────────────────────────────────────────
        pct_frame = ttk.LabelFrame(self, text="Percentage breakdown (per subfolder)")
        pct_frame.pack(fill="x", **pad)
        ttk.Label(pct_frame, text="train [%]:").grid(row=0, column=0, sticky="e")
        ttk.Entry(pct_frame, width=6, textvariable=self.pct_train).grid(row=0, column=1, sticky="w")
        ttk.Label(pct_frame, text="valid [%]:").grid(row=0, column=2, sticky="e")
        ttk.Entry(pct_frame, width=6, textvariable=self.pct_valid).grid(row=0, column=3, sticky="w")
        ttk.Label(pct_frame, text="test [%]:").grid(row=0, column=4, sticky="e")
        ttk.Entry(pct_frame, width=6, textvariable=self.pct_test).grid(row=0, column=5, sticky="w")
        ttk.Checkbutton(pct_frame, text="Crop to perfect proportions (if needed)",
                        variable=self.reduce_to_fit).grid(row=0, column=6, padx=10, sticky="w")
        ttk.Button(pct_frame, text="Recalculate the division", command=self.compute_splits).grid(row=0, column=7, padx=10)

        topper_frame = ttk.LabelFrame(self, text="Folder for punching (preferred for proofreading)")
        topper_frame.pack(fill="x", **pad)
        self.topper_combo = ttk.Combobox(topper_frame, textvariable=self.topper_folder, values=[], state="readonly")
        self.topper_combo.pack(side="left", padx=6, pady=6)
        ttk.Label(topper_frame, text="(optional - select after scanning)").pack(side="left")

        # ── Tables ───────────────────────────────────────────────────────────
        table_frame = ttk.LabelFrame(self, text="Global division (sum)")
        table_frame.pack(fill="x", **pad)
        self.tree_global = ttk.Treeview(table_frame, columns=("split", "count", "target"), show="headings", height=4)
        self.tree_global.heading("split", text="Split")
        self.tree_global.heading("count", text="Currently")
        self.tree_global.heading("target", text="Aim")
        self.tree_global.column("split", width=120, anchor="center")
        self.tree_global.column("count", width=140, anchor="e")
        self.tree_global.column("target", width=140, anchor="e")
        self.tree_global.pack(fill="x", padx=6, pady=6)

        per_folder_frame = ttk.LabelFrame(self, text="Breakdown per subfolder (numbers and %)")
        per_folder_frame.pack(fill="both", expand=True, **pad)
        cols = ("folder", "total", "train", "valid", "test", "pct_train", "pct_valid", "pct_test")
        self.tree_folders = ttk.Treeview(per_folder_frame, columns=cols, show="headings", height=12)
        headers = {
            "folder": "Folder",
            "total": "Total",
            "train": "Train",
            "valid": "Valid",
            "test": "Test",
            "pct_train": "%Train",
            "pct_valid": "%Valid",
            "pct_test": "%Test",
        }
        widths = {
            "folder": 260, "total": 80, "train": 80, "valid": 80, "test": 80,
            "pct_train": 80, "pct_valid": 80, "pct_test": 80,
        }
        anchors = {
            "folder": "w", "total": "e", "train": "e", "valid": "e", "test": "e",
            "pct_train": "e", "pct_valid": "e", "pct_test": "e",
        }
        for c in cols:
            self.tree_folders.heading(c, text=headers[c])
            self.tree_folders.column(c, width=widths[c], anchor=anchors[c])
        self.tree_folders.pack(fill="both", expand=True, padx=6, pady=6)

        # ── Histogram klas ───────────────────────────────────────────────────
        cls_frame = ttk.LabelFrame(self, text="Class multiplicities (globally)")
        cls_frame.pack(fill="x", **pad)
        self.tree_classes = ttk.Treeview(cls_frame, columns=("cls", "count"), show="headings", height=6)
        self.tree_classes.heading("cls", text="class_id")
        self.tree_classes.heading("count", text="count")
        self.tree_classes.column("cls", width=120, anchor="e")
        self.tree_classes.column("count", width=160, anchor="e")
        self.tree_classes.pack(side="left", fill="x", expand=True, padx=6, pady=6)
        ttk.Button(cls_frame, text="Export CSV", command=self.export_classes_csv).pack(side="left", padx=10)

        # ── Akcje ────────────────────────────────────────────────────────────
        actions = ttk.Frame(self)
        actions.pack(fill="x", **pad)
        ttk.Button(actions, text="Move to destination", command=self.move_action).pack(side="left")
        ttk.Button(actions, text="Finish", command=self.destroy).pack(side="right")

        # ── Log ──────────────────────────────────────────────────────────────
        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(
            log_frame,
            height=8,
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="#d4d4d4",
            selectbackground="#3a7afe"
        )
        self.log.pack(fill="both", expand=True, padx=6, pady=6)

    # ── Handlery ─────────────────────────────────────────────────────────────
    def choose_src(self):
        path = filedialog.askdirectory(title="Select source directory")
        if path:
            self.src_path.set(path)

    def choose_dst(self):
        path = filedialog.askdirectory(title="Select destination directory")
        if path:
            self.dst_path.set(path)

    def log_write(self, msg: str):
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.update_idletasks()

    def scan_action(self):
        src = self.src_path.get().strip()
        if not src:
            messagebox.showerror(APP_TITLE, "Specify source directory.")
            return
        root = Path(src)
        if not root.exists():
            messagebox.showerror(APP_TITLE, "The source directory does not exist.")
            return

        # liczba klas (opcjonalna)
        nc = None
        if self.num_classes.get().strip() != "":
            try:
                tmp = int(self.num_classes.get().strip())
                if tmp <= 0:
                    raise ValueError
                nc = tmp
            except Exception:
                messagebox.showerror(APP_TITLE, "Number of classes must be a positive integer or empty.")
                return

        self.log_write(f"[SKAN] Start: {root}")
        series_ok, incomplete, duplicates, class_hist = scan_source(root, nc)

        if duplicates:
            details = []
            for fname, paths in sorted(duplicates.items()):
                details.append(fname)
                for p in paths:
                    details.append(f"    {p}")
            show_scrollable_message(APP_TITLE, "Duplicate file names detected:\n\n" + "\n".join(details))
            self.log_write("[ERROR] Duplicate files:\n" + "\n".join(details))
            return

        if incomplete:
            lines = []
            lines.append(f"Number of incomplete series: {len(incomplete)}\n")
            for idx, (base, info) in enumerate(sorted(incomplete.items()), start=1):
                lines.append(f"{idx}) {base}")
                lines.append(f"    folder: {info['folder']}")
                for pr in info["problems"]:
                    lines.append(f"    - {pr}")
                lines.append("")
            show_scrollable_message(APP_TITLE, "Incomplete/incorrect series:\n\n" + "\n".join(lines))
            self.log_write(f"[WARNING] Incomplete series: {len(incomplete)}")
            return

        self.series = series_ok
        self.class_hist = class_hist
        n_series = len(series_ok)
        self.n_series.set(n_series)
        self.n_images.set(n_series)
        self.n_labels.set(n_series)
        self.summary_lbl.config(text=f"Series: {n_series} | Images: {self.n_images.get()} | Labels: {self.n_labels.get()}")
        self.log_write(f"[OK] Complete series found: {n_series}")

        series_by_folder = defaultdict(list)
        for base, data in self.series.items():
            series_by_folder[data["folder"]].append(base)
        self.series_by_folder = dict(sorted(series_by_folder.items(), key=lambda kv: kv[0].lower()))


        self.topper_combo["values"] = list(self.series_by_folder.keys())
        if not self.topper_folder.get() and self.topper_combo["values"]:
            self.topper_folder.set(self.topper_combo["values"][0])


        self.refresh_classes_view()

        self.compute_splits()

    def refresh_classes_view(self):
        for row in self.tree_classes.get_children():
            self.tree_classes.delete(row)
        for cls, cnt in sorted(self.class_hist.items()):
            self.tree_classes.insert("", "end", values=(cls, cnt))

    def compute_splits(self):
        if not self.series:
            return
        try:
            pctT, pctV, pctE = self.pct_train.get(), self.pct_valid.get(), self.pct_test.get()
            _ = compute_split_counts(1, pctT, pctV, pctE)
        except Exception as e:
            messagebox.showerror(APP_TITLE, str(e))
            return


        global_counts = {"train": 0, "valid": 0, "test": 0}
        folder_rows = []
        for folder, bases in self.series_by_folder.items():
            total = len(bases)
            counts = compute_split_counts(total, self.pct_train.get(), self.pct_valid.get(), self.pct_test.get())
            tr, va, te = counts["train"], counts["valid"], counts["test"]
            def pct(x, tot): return 0.0 if tot == 0 else round(100.0 * x / tot, 1)
            folder_rows.append((folder, total, tr, va, te, f"{pct(tr,total):.1f}", f"{pct(va,total):.1f}", f"{pct(te,total):.1f}"))
            for k in global_counts:
                global_counts[k] += counts[k]

        for row in self.tree_folders.get_children():
            self.tree_folders.delete(row)
        for vals in folder_rows:
            self.tree_folders.insert("", "end", values=vals)


        self.assign_map = pick_and_assign_per_folder(self.series_by_folder, self.pct_train.get(), self.pct_valid.get(), self.pct_test.get())


        target = compute_target_counts(len(self.series), self.pct_train.get(), self.pct_valid.get(), self.pct_test.get())


        current_global = Counter(self.assign_map.values())
        for row in self.tree_global.get_children():
            self.tree_global.delete(row)
        for split in ["train", "valid", "test"]:
            self.tree_global.insert("", "end", values=(split, current_global.get(split, 0), target[split]))

        self.log_write("[INFO] Updated per folder and global targets breakdown.")

    def move_action(self):
        if not self.series:
            messagebox.showinfo(APP_TITLE, "No scanned data.")
            return
        dst = self.dst_path.get().strip()
        if not dst:
            messagebox.showerror(APP_TITLE, "Specify destination directory.")
            return
        dst_root = Path(dst)
        dst_root.mkdir(parents=True, exist_ok=True)

        try:
            _ = compute_split_counts(1, self.pct_train.get(), self.pct_valid.get(), self.pct_test.get())
        except Exception as e:
            messagebox.showerror(APP_TITLE, str(e))
            return

        target = compute_target_counts(len(self.series), self.pct_train.get(), self.pct_valid.get(), self.pct_test.get())
        topper = self.topper_folder.get().strip() or None
        reduce_to_fit = bool(self.reduce_to_fit.get())


        new_assign, excluded, adj_log = adjust_assign_to_exact(dict(self.assign_map), self.series_by_folder, target, topper, reduce_to_fit)


        if adj_log:
            show_scrollable_message(APP_TITLE, "Adjustments to achieve targets:\n\n" + "\n".join(adj_log))
            self.log_write("[INFO] Corrections to targets have been made (see details window).")

        try:
            move_series(new_assign, self.series, dst_root, excluded)
        except FileExistsError as e:
            show_scrollable_message(APP_TITLE, f"Collision in target directory:\n\n{e}")
            self.log_write(f"[ERROR] {e}")
            return
        except Exception as e:
            show_scrollable_message(APP_TITLE, f"Transfer error:\n\n{e}")
            self.log_write(f"[ERROR] {e}")
            return

        rep_split = write_split_report_csv(new_assign, self.series, dst_root, excluded)
        rep_classes = write_classes_report_csv(self.class_hist, dst_root)
        messagebox.showinfo(APP_TITLE, f"Transfer completed.\nDivision report: {rep_split}\nClass Report: {rep_classes}")
        self.log_write(f"[OK] Completed. Reports: {rep_split}, {rep_classes}")

    def export_classes_csv(self):
        if not self.class_hist:
            messagebox.showinfo(APP_TITLE, "No class data available. Scan the source first..")
            return
        dst = self.dst_path.get().strip()
        if not dst:
            messagebox.showerror(APP_TITLE, "Specify destination directory (where to save CSV).")
            return
        dst_root = Path(dst)
        dst_root.mkdir(parents=True, exist_ok=True)
        p = write_classes_report_csv(self.class_hist, dst_root)
        messagebox.showinfo(APP_TITLE, f"Save: {p}")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
