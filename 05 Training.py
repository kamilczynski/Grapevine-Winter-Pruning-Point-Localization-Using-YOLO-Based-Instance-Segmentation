!pip install -U ultralytics opencv-python

import sys, torch
print("PY:", sys.executable)
print("Torch:", torch.__version__, "| CUDA:", torch.version.cuda, "| GPU:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

from pathlib import Path


DATASET_NOBILISSEGMENTACJA = Path(r"C:\Users\")
# Checking the data structure
for p in [
    DATASET_CUT,
    DATASET_CUT / "images/train",
    DATASET_CUT / "images/val",    
    DATASET_CUT / "images/test", 
    DATASET_CUT / "labels/train",
    DATASET_CUT / "labels/val",
    DATASET_CUT / "labels/test"
]:
    print(p, "→", "OK ✅" if p.exists() else "❌ DOES NOT EXIST")


data_yaml = r"""
path: C:\Users
train: images/train
val: images/val
test: images/test
names:
  0: headingCut
  1: rejectingCut
"""
with open("data_cut.yaml", "w", encoding="utf-8") as f:
    f.write(data_yaml)
print("✔ Save data_cut.yaml")

from ultralytics import YOLO
model = YOLO("yolo-seg.pt")
results = model.train(
    #data="data_cut.yaml",        
    data="data_cut.yaml",        
    epochs=200,
    batch=32,
    imgsz=640,
    optimizer='SGD',
    momentum=0.937,
    weight_decay=0.0005,
    lr0=0.01,
    lrf=0.01,
    seed=0,
    augment=True,
    workers=8,  
    patience=0,
    device='cuda',
    #project="runs/train_data_cut",     
    #name="yolov26n_data_cut",          
    project="runs/train_data_cut",     
    name="yolov26n_data_cut",           
)
