import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox


def polygon_to_yolo_seg(points, img_w, img_h):
    """
    LabelMe polygon point conversion -> list of coordinates [x1, y1, x2, y2, ...]
    normalized to the 0–1 range (YOLO-seg format).
    """
    coords = []
    for x, y in points:
        xn = x / img_w
        yn = y / img_h
        # optionally trim to [0,1] just in case
        xn = max(0.0, min(1.0, xn))
        yn = max(0.0, min(1.0, yn))
        coords.extend([xn, yn])
    return coords


def convert():
    img_folder = entry_img.get()
    json_folder = entry_json.get()
    out_folder = entry_out.get()

    if not img_folder or not json_folder or not out_folder:
        messagebox.showerror("Error", "Select all folders.")
        return

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]

    if not json_files:
        messagebox.showerror("Error", "No JSON files in the folder.")
        return

    # Simple mapping: label name -> class ID (0,1,2,...)
    label_to_id = {}
    next_class_id = 0

    for jf in json_files:
        json_path = os.path.join(json_folder, jf)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_name = data.get("imagePath")
        if not img_name:
            print(f"⚠ No 'imagePath' w {jf}, I skip.")
            continue

        img_path = os.path.join(img_folder, img_name)

        if not os.path.exists(img_path):
            print(f"⚠ No image: {img_name} (for {jf}), I skip.")
            continue

        # Get image size from JSON, or if not available, from file
        if "imageWidth" in data and "imageHeight" in data:
            w, h = data["imageWidth"], data["imageHeight"]
        else:
            try:
                from PIL import Image
            except ImportError:
                messagebox.showerror(
                    "Error",
                    "No image dimensions information in JSON,\n"
                    "and the Pillow library (PIL) is not installed.\n"
                    "Install it: pip install pillow"
                )
                return
            img = Image.open(img_path)
            w, h = img.size

        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(out_folder, txt_name)

        lines = []

        for shape in data.get("shapes", []):
            # We are only interested in polygons (segmentation)
            if shape.get("shape_type") != "polygon":
                continue

            label = shape.get("label", "unknown")

            # Mapping class name to ID (0,1,2,...)
            if label not in label_to_id:
                label_to_id[label] = next_class_id
                next_class_id += 1

            class_id = label_to_id[label]

            points = shape.get("points", [])
            if len(points) < 3:
                # too few points to make a sensible training ground
                continue

            seg_coords = polygon_to_yolo_seg(points, w, h)

            # Format YOLO-seg: class_id x1 y1 x2 y2 ...
            line = str(class_id) + " " + " ".join(f"{c:.6f}" for c in seg_coords)
            lines.append(line)

        if lines:
            with open(txt_path, "w", encoding="utf-8") as out:
                out.write("\n".join(lines))
        else:
            # If there are no polygons, you can skip the file or make it blank
            # Here we create an empty file so that YOLO doesn't crash
            open(txt_path, "w").close()

    # Viewing the class map in the console
    print("Label Map -> ID klas:")
    for lbl, cid in label_to_id.items():
        print(f"  {cid}: {lbl}")

    messagebox.showinfo("Ready", "Segmentation conversion to YOLO completed!")


# ==================================
#              GUI
# ==================================

root = tk.Tk()
root.title("LabelMe JSON Converter (Segmentation) → YOLO-seg TXT")
root.geometry("650x260")


def pick_folder(entry):
    folder = filedialog.askdirectory()
    if folder:
        entry.delete(0, tk.END)
        entry.insert(0, folder)


tk.Label(root, text="Folder with images:").pack()
entry_img = tk.Entry(root, width=80)
entry_img.pack()
tk.Button(root, text="Choose", command=lambda: pick_folder(entry_img)).pack()

tk.Label(root, text="Folder with JSON files (LabelMe):").pack()
entry_json = tk.Entry(root, width=80)
entry_json.pack()
tk.Button(root, text="Choose", command=lambda: pick_folder(entry_json)).pack()

tk.Label(root, text="YOLO TXT save folder (segmentation):").pack()
entry_out = tk.Entry(root, width=80)
entry_out.pack()
tk.Button(root, text="Choose", command=lambda: pick_folder(entry_out)).pack()

tk.Button(
    root,
    text="CONVERT (segmentation → YOLO)",
    bg="green",
    fg="white",
    #font=("Arial", 13),
    font=("Orbitron", 13, "bold"),
    command=convert
).pack(pady=12)

root.mainloop()
