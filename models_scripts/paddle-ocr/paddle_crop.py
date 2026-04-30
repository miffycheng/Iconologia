"""
Crop detected text regions from a page image.

Reads the detection JSON produced by paddle_det.py and saves each bounding box
as a cropped JPG for downstream recognition.

Usage:
    python paddle_crop.py

Edit the paths at the top of this file before running.
Run paddle_det.py first to produce the detection JSON.
"""

import json
import cv2
from pathlib import Path

# ── Configure these paths ──────────────────────────────────────────────────────
IMAGE_PATH = Path("../../data/Amsterdam_1698_Jean_Baudoin/JPG/JPG/page_31.jpg")
DET_JSON   = Path("../../output/res_det.json")
OUTPUT_DIR = Path("../../crops_simple")
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(exist_ok=True)

img = cv2.imread(str(IMAGE_PATH))
h, w = img.shape[:2]

det = json.loads(DET_JSON.read_text(encoding="utf-8"))
polys  = det["dt_polys"]
scores = det["dt_scores"]


def poly_to_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)
    return x1, y1, x2, y2


kept = 0
for i, (poly, sc) in enumerate(zip(polys, scores)):
    x1, y1, x2, y2 = poly_to_bbox(poly)
    if (x2 - x1) < 10 or (y2 - y1) < 10:
        continue
    if sc < 0.6:
        continue
    crop = img[y1:y2, x1:x2]
    cv2.imwrite(str(OUTPUT_DIR / f"{IMAGE_PATH.stem}_box{i:03d}_s{sc:.2f}.jpg"), crop)
    kept += 1

print(f"Done. Saved {kept} crops to: {OUTPUT_DIR}")
