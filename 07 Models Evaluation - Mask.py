
import os
import cv2
import yaml
import numpy as np
from ultralytics import YOLO
from pathlib import Path



model_path = "C:/Users/weights/best.pt"
data_yaml  = "C:/Users/data_cut.yaml"
save_dir   = "C:/"


img_dir = Path(save_dir) / "images"
lbl_dir = Path(save_dir) / "labels"
img_dir.mkdir(parents=True, exist_ok=True)
lbl_dir.mkdir(parents=True, exist_ok=True)


with open(data_yaml, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

root = Path(data["path"])
test_dir = root / data["test"]
assert test_dir.exists(), f"❌ Brak folderu test: {test_dir}"

image_list = list(test_dir.glob("*.*"))
print(f"[INFO] Test images: {len(image_list)}")


model = YOLO(model_path)
print("[INFO] Model task:", model.task)
assert model.task == "segment", "❌ The model is NOT segmented!"


for img_path in image_list:

    print(f"[PROCESS] {img_path.name}")

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    name = img_path.stem

    result = model.predict(
        source=img,
        conf=0.01,
        iou=0.5,
        verbose=False
    )[0]

   
    plotted = result.plot(
        boxes=False,     
        conf=False,      
        labels=True      
    )

  
    cv2.imwrite(str(img_dir / f"{name}.jpg"), plotted)


    label_lines = []

    if result.masks is not None:
        masks = result.masks.xy      
        classes = result.boxes.cls.cpu().numpy().astype(int)

        h, w = img.shape[:2]

        for poly, cls in zip(masks, classes):
            poly_norm = []
            for x, y in poly:
                poly_norm.append(x / w)
                poly_norm.append(y / h)

            label_lines.append(
                f"{cls} " + " ".join(f"{p:.6f}" for p in poly_norm)
            )

    with open(lbl_dir / f"{name}.txt", "w", encoding="utf-8") as f:
        for line in label_lines:
            f.write(line + "\n")

print("✅ READY — style ULTRALYTICS, maski only, no bbox/conf")
