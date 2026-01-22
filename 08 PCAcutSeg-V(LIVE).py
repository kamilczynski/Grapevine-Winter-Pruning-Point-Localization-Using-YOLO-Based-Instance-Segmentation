import cv2
import numpy as np
import os
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk



BG_MAIN   = "#0f1117"
BG_PANEL  = "#161a22"
BG_BUTTON = "#1f2430"
FG_TEXT   = "#e6e6e6"
ACCENT    = "#00ff99"

FONT_TITLE = ("Segoe UI", 14, "bold")
FONT_BTN   = ("Segoe UI", 10)
FONT_LABEL = ("Segoe UI", 10)



CLASS_COLORS = {
    0: {"mask": (255, 0, 0),   "line": (0, 255, 0), "point": (0, 255, 0)},
    1: {"mask": (255, 255, 0), "line": (0, 0, 255), "point": (0, 0, 255)},
}

MASK_ALPHA = 0.25


def load_yolo_segmentation_txt(txt_path, img_w, img_h):
    objects = []
    if not os.path.exists(txt_path):
        return objects

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue

            cls = int(parts[0])
            coords = np.array(list(map(float, parts[1:])))
            xs = coords[0::2] * img_w
            ys = coords[1::2] * img_h
            polygon = np.vstack((xs, ys)).T.astype(int)
            objects.append((cls, polygon))

    return objects



def cutline_pca_edge(polygon, img_h, img_w, edge="top"):
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None

    cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    if len(cnt) < 5:
        return None

    mean = cnt.mean(axis=0)
    pts = cnt - mean

    _, _, Vt = np.linalg.svd(pts, full_matrices=False)
    axis_main = Vt[0]

   
    proj_test = pts @ axis_main
    if np.corrcoef(proj_test, cnt[:, 1])[0, 1] > 0:
        axis_main = -axis_main

    proj_main = pts @ axis_main

  
    if edge == "top":
        edge_pts = cnt[proj_main >= np.percentile(proj_main, 90)]
    else:
        edge_pts = cnt[proj_main <= np.percentile(proj_main, 10)]

    if len(edge_pts) < 2:
        return None

    vx, vy, x0, y0 = cv2.fitLine(edge_pts, cv2.DIST_L2, 0, 0.01, 0.01)
    v = np.array([vx[0], vy[0]], dtype=np.float32)
    v /= np.linalg.norm(v)

    proj = (edge_pts - np.array([x0[0], y0[0]])) @ v
    t1, t2 = proj.min(), proj.max()

    x1 = int(x0 + v[0] * t1)
    y1 = int(y0 + v[1] * t1)
    x2 = int(x0 + v[0] * t2)
    y2 = int(y0 + v[1] * t2)

    return (
        int(np.clip(x1, 0, img_w - 1)),
        int(np.clip(y1, 0, img_h - 1)),
        int(np.clip(x2, 0, img_w - 1)),
        int(np.clip(y2, 0, img_h - 1)),
    )


def midpoint_of_line(line):
    x1, y1, x2, y2 = line
    return (x1 + x2) // 2, (y1 + y2) // 2



class CutlineGUI:
    def __init__(self, master):
        master.title("YOLO-Seg Cutting Line Visualizer – PCA")
        master.geometry("1100x760")
        master.configure(bg=BG_MAIN)

        master.bind("<Left>",  lambda e: self.show_prev())
        master.bind("<Right>", lambda e: self.show_next())

        panel = Frame(master, bg=BG_PANEL, width=240)
        panel.pack(side=LEFT, fill=Y)

        main = Frame(master, bg=BG_MAIN)
        main.pack(side=RIGHT, fill=BOTH, expand=True)

        Label(panel, text="PCAcutSeg-V", fg=ACCENT, bg=BG_PANEL, font=FONT_TITLE).pack(pady=18)

        self.btn(panel, "📁 Folder Images", self.select_image_folder)
        self.btn(panel, "📄 Folder TXT", self.select_txt_folder)
        self.btn(panel, "◀ Previous", self.show_prev)
        self.btn(panel, "Next ▶", self.show_next)
        self.btn(panel, "💾 Save", self.save_output)

        self.info_label = Label(main, fg=FG_TEXT, bg=BG_MAIN, font=FONT_LABEL)
        self.info_label.pack(pady=6)

        self.canvas = Canvas(main, width=900, height=600, bg="#000000", highlightthickness=0)
        self.canvas.pack(padx=10, pady=10)

        self.img_folder = None
        self.txt_folder = None
        self.images = []
        self.index = 0
        self.current_img_cv = None

    def btn(self, parent, text, cmd):
        Button(
            parent, text=text, command=cmd,
            bg=BG_BUTTON, fg=FG_TEXT,
            font=FONT_BTN, relief=FLAT,
            activebackground=ACCENT, activeforeground="#000000",
            padx=10, pady=10
        ).pack(fill=X, padx=14, pady=6)



    def select_image_folder(self):
        self.img_folder = filedialog.askdirectory()
        self.images = sorted(f for f in os.listdir(self.img_folder)
                             if f.lower().endswith((".jpg", ".jpeg", ".png")))
        self.index = 0
        self.try_show_image()

    def select_txt_folder(self):
        self.txt_folder = filedialog.askdirectory()
        self.try_show_image()

    def try_show_image(self):
        if self.img_folder and self.txt_folder:
            self.show_image()

    def show_prev(self):
        if not self.images:
            return
        self.index = (self.index - 1) % len(self.images)
        self.show_image()

    def show_next(self):
        if not self.images:
            return
        self.index = (self.index + 1) % len(self.images)
        self.show_image()

    def show_image(self):
        name = self.images[self.index]
        img = cv2.imread(os.path.join(self.img_folder, name))
        if img is None:
            return

        h, w = img.shape[:2]
        self.info_label.config(text=f"[{self.index+1}/{len(self.images)}] {name}")

        objects = load_yolo_segmentation_txt(
            os.path.join(self.txt_folder, os.path.splitext(name)[0] + ".txt"), w, h
        )

        for cls, polygon in objects:
            c = CLASS_COLORS.get(cls)
            if c is None:
                continue

            overlay = img.copy()
            cv2.fillPoly(overlay, [polygon], c["mask"])
            cv2.addWeighted(overlay, MASK_ALPHA, img, 1 - MASK_ALPHA, 0, img)

            edge = "top" if cls == 0 else "bottom"
            res = cutline_pca_edge(polygon, h, w, edge)

            if res:
                cv2.line(img, res[:2], res[2:], c["line"], 6, cv2.LINE_AA)
                cx, cy = midpoint_of_line(res)
                cv2.circle(img, (cx, cy), 4, (0, 0, 0), -1)
                cv2.circle(img, (cx, cy), 3, c["point"], -1)

        self.current_img_cv = img
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb).resize((900, 600))
        self.tk_img = ImageTk.PhotoImage(img_pil)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=NW, image=self.tk_img)

    def save_output(self):
        out = os.path.join(self.img_folder, f"cutline_{self.images[self.index]}")
        cv2.imwrite(out, self.current_img_cv)
        messagebox.showinfo("Saved", out)



if __name__ == "__main__":
    root = Tk()
    CutlineGUI(root)
    root.mainloop()
