"""
Text recognition on cropped images with PaddleOCR PP-OCRv5.

Reads crops produced by paddle_crop.py and saves recognition results to JSON.

Usage:
    python paddle_rec.py

Edit the paths at the top of this file before running.
Run paddle_det.py and paddle_crop.py first.
"""

import json
from pathlib import Path
from paddleocr import TextRecognition

# ── Configure these paths ──────────────────────────────────────────────────────
CROPS_DIR  = Path("../../crops_simple")
OUTPUT_DIR = Path("../../output")
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = TextRecognition(model_name="PP-OCRv5_mobile_rec")
output = model.predict(input=str(CROPS_DIR), batch_size=2)

all_results = []
for res in output:
    all_results.append(res.json)

out_path = OUTPUT_DIR / "rec_all.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"Saved {len(all_results)} results to {out_path}")
