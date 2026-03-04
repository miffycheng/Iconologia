import json
import cv2
from pathlib import Path

IMG_PATH = Path("./data/Amsterdam_1698_Jean_Baudoin/selected/page_31.jpg")
DET_JSON = Path("./output/res_det.json")   # 改成你的 det json 路徑
OUT_DIR = Path("./crops_simple")
OUT_DIR.mkdir(exist_ok=True)

# 讀圖
img = cv2.imread(str(IMG_PATH))
h, w = img.shape[:2]

# 讀 det JSON（你的格式）
det = json.loads(DET_JSON.read_text(encoding="utf-8"))
polys = det["dt_polys"]
scores = det["dt_scores"]

def poly_to_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    # clamp to image bounds
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    return x1, y1, x2, y2

kept = 0
for i, (poly, sc) in enumerate(zip(polys, scores)):
    x1, y1, x2, y2 = poly_to_bbox(poly)

    # 過濾太小 or 太低分數（可調）
    if (x2 - x1) < 10 or (y2 - y1) < 10:
        continue
    if sc < 0.6:
        continue

    crop = img[y1:y2, x1:x2]
    cv2.imwrite(str(OUT_DIR / f"{IMG_PATH.stem}_box{i:03d}_s{sc:.2f}.jpg"), crop)
    kept += 1

print(f"Done. Saved {kept} crops to: {OUT_DIR}")
