import sys
import time
import cv2
import numpy as np


sys.path.append("/opt/pyorbbecsdk/build")
from pyorbbecsdk import *  # noqa

from ultralytics import YOLO


ENGINE_PATH = "//workspace//weights/best.engine"
WINDOW_NAME = "Orbbec RGB + YOLO-SEG + PCAcutSeg-V (LIVE)"

CLASS_HEADING_CUT   = 0
CLASS_REJECTING_CUT = 1

CLASS_COLORS = {
    0: {"mask": (255, 0, 0),   "line": (0, 255, 0), "point": (0, 255, 0)},
    1: {"mask": (255, 255, 0), "line": (0, 0, 255), "point": (0, 0, 255)},
}

MASK_ALPHA        = 0.25
LINE_THICKNESS    = 6
POINT_OUTLINE_R   = 4
POINT_R           = 3



def cutline_pca_edge(polygon, img_h, img_w, edge="top"):
    if polygon is None or len(polygon) < 5:
        return None

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
    axis = Vt[0]

 
    if np.corrcoef(pts @ axis, cnt[:, 1])[0, 1] > 0:
        axis = -axis

    proj = pts @ axis


    if edge == "top":
        edge_pts = cnt[proj >= np.percentile(proj, 90)]
    else:
        edge_pts = cnt[proj <= np.percentile(proj, 10)]

    if len(edge_pts) < 2:
        return None

    vx, vy, x0, y0 = cv2.fitLine(edge_pts, cv2.DIST_L2, 0, 0.01, 0.01)
    v = np.array([vx[0], vy[0]], dtype=np.float32)
    v /= np.linalg.norm(v)

    t = (edge_pts - np.array([x0[0], y0[0]])) @ v
    t1, t2 = t.min(), t.max()

    x1 = int(np.clip(x0 + v[0] * t1, 0, img_w - 1))
    y1 = int(np.clip(y0 + v[1] * t1, 0, img_h - 1))
    x2 = int(np.clip(x0 + v[0] * t2, 0, img_w - 1))
    y2 = int(np.clip(y0 + v[1] * t2, 0, img_h - 1))

    return x1, y1, x2, y2


def midpoint(line):
    return (line[0] + line[2]) // 2, (line[1] + line[3]) // 2


def draw_line_and_point(img, line, color):
    cv2.line(img, line[:2], line[2:], color, LINE_THICKNESS, cv2.LINE_AA)
    cx, cy = midpoint(line)
    cv2.circle(img, (cx, cy), POINT_OUTLINE_R, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), POINT_R, color, -1, cv2.LINE_AA)


def main():
    print("✅ LIVE Orbbec + YOLO-SEG + PCAcutSeg-V (SIMPLE)")

    model = YOLO(ENGINE_PATH, task="segment")

    pipeline = Pipeline()
    config = Config()

    profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    profile = next((p for p in profiles if p.get_format() == OBFormat.RGB), None)
    if profile is None:
        print("❌ Brak profilu RGB")
        return

    config.enable_stream(profile)
    pipeline.start(config)

    prev_time = time.time()

    try:
        while True:
            frames = pipeline.wait_for_frames(100)
            if frames is None:
                continue

            cf = frames.get_color_frame()
            if cf is None:
                continue

            w, h = cf.get_width(), cf.get_height()
            data = np.frombuffer(cf.get_data(), dtype=np.uint8)
            if data.size != w * h * 3:
                continue

            img_rgb = data.reshape((h, w, 3))
            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            annotated = img.copy()

            results = model(img, verbose=False)
            res = results[0]

            if res.masks is not None and res.boxes is not None:
                polygons = res.masks.xy
                classes  = res.boxes.cls.cpu().numpy().astype(int)

                for poly, cls in zip(polygons, classes):
                    polygon = poly.astype(np.int32)
                    colors = CLASS_COLORS.get(cls)
                    if colors is None:
                        continue

                    overlay = annotated.copy()
                    cv2.fillPoly(overlay, [polygon], colors["mask"])
                    cv2.addWeighted(overlay, MASK_ALPHA, annotated, 1 - MASK_ALPHA, 0, annotated)

                    edge = "top" if cls == CLASS_HEADING_CUT else "bottom"
                    line = cutline_pca_edge(polygon, h, w, edge=edge)

                    if line:
                        draw_line_and_point(annotated, line, colors["line"])

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(
                annotated, f"FPS: {fps:.1f}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2
            )

            cv2.imshow(WINDOW_NAME, annotated)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
