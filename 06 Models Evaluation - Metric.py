from ultralytics import YOLO
import torch
import os
import csv


model_path = r"C:/Users/weights/best.pt"
data_yaml  = r"C:/Users/data_cut.yaml"


save_dir   = r"C:/Users"
run_name   = "testset_eval"   

os.makedirs(save_dir, exist_ok=True)

# ================== CUDA ==================
device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    raise SystemError("❌ Brak CUDA – ewaluacja wymaga GPU")

print(f"\n🔥 SEGMENTATION EVALUATION – TEST SET | GPU: {torch.cuda.get_device_name(0)}")

# ================== MODEL ==================
model = YOLO(model_path)
class_names = model.names  # dict: {id: name}


results = model.val(
    data=data_yaml,
    split="test",
    imgsz=640,
    device=device,
    verbose=True,
    plots=True,     
    save=True,      
    project=save_dir,
    name=run_name,
    exist_ok=True
)


out_dir = os.path.join(save_dir, run_name)

# ================== GLOBAL METRICS (SEG) ===================
# Note: For segmentation in Ultralytics, metrics are in results.seg
if results.seg is None:
    raise RuntimeError("❌ results.seg is None — it looks like model/val is not in segmentation mode.")

metrics_global = {
    "Precision_mean": float(results.seg.p.mean()),
    "Recall_mean":    float(results.seg.r.mean()),
    "F1_mean":        float(results.seg.f1.mean()),
    "mAP50_mean":     float(results.seg.map50),
    "mAP50-95_mean":  float(results.seg.map),
}


metrics_per_class = []
for cls_id, cls_name in class_names.items():
    metrics_per_class.append({
        "Class_ID": int(cls_id),
        "Class_Name": str(cls_name),
        "Precision": float(results.seg.p[cls_id]),
        "Recall":    float(results.seg.r[cls_id]),
        "F1":        float(results.seg.f1[cls_id]),
        "mAP50":     float(results.seg.ap50[cls_id]),
        "mAP50-95":  float(results.seg.ap[cls_id]),
    })


csv_path = os.path.join(out_dir, "segmentation_metrics_testset.csv")

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

  
    writer.writerow(["=== GLOBAL METRICS (TEST SET) ==="])
    writer.writerow(["Metric", "Value"])
    for k, v in metrics_global.items():
        writer.writerow([k, f"{v:.6f}"])

    writer.writerow([])


    writer.writerow(["=== PER CLASS METRICS (TEST SET) ==="])
    writer.writerow([
        "Class_ID", "Class_Name",
        "Precision", "Recall", "F1",
        "mAP@0.5", "mAP@0.5:0.95"
    ])

    for m in metrics_per_class:
        writer.writerow([
            m["Class_ID"],
            m["Class_Name"],
            f"{m['Precision']:.6f}",
            f"{m['Recall']:.6f}",
            f"{m['F1']:.6f}",
            f"{m['mAP50']:.6f}",
            f"{m['mAP50-95']:.6f}",
        ])

print(f"\n✅ CSV save to:\n{csv_path}")


expected = [
    "MaskPR_curve.png",
    "MaskP_curve.png",
    "MaskR_curve.png",
    "MaskF1_curve.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png"
]

print("\n📌 Expected graphs (Ultralytics) in the folder:")
print(out_dir)

missing = []
for fn in expected:
    p = os.path.join(out_dir, fn)
    if os.path.exists(p):
        print(f"  ✅ {fn}")
    else:
        missing.append(fn)
        print(f"  ⚠️ lack: {fn}")

if missing:
    print("\n⚠️ If something is missing, the most common causes:")
    print(" - too few predictions (e.g. conf too high → detects almost nothing)")
    print(" - dataset/labels have a problem (e.g. empty/incorrect classes)")
    print(" - validation was started without correct detection results")
    print("Then try running val with a different conf/IoU (see below).")

print("\n=== 🟢 TEST SET EVALUATION COMPLETED ===")
