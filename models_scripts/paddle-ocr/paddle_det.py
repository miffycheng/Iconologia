"""
Text detection with PaddleOCR PP-OCRv5.

Runs detection on a single image and saves:
  - annotated image  → OUTPUT_DIR/
  - bounding box JSON → OUTPUT_DIR/res_det.json

Usage:
    python paddle_det.py

Edit the paths at the top of this file before running.
"""

from pathlib import Path
from paddleocr import TextDetection

# ── Configure these paths ──────────────────────────────────────────────────────
IMAGE_PATH = Path("../../data/Amsterdam_1698_Jean_Baudoin/JPG/JPG/page_31.jpg")
OUTPUT_DIR = Path("../../output")
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = TextDetection(model_name="PP-OCRv5_mobile_det")
output = model.predict(input=str(IMAGE_PATH), batch_size=1)

for res in output:
    res.print()
    res.save_to_img(save_path=str(OUTPUT_DIR) + "/")
    res.save_to_json(save_path=str(OUTPUT_DIR / "res_det.json"))
