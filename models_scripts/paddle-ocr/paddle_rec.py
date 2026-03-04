from paddleocr import TextRecognition
import json
from pathlib import Path

model = TextRecognition(model_name="PP-OCRv5_mobile_rec")
output = model.predict(input="./crops_simple", batch_size=2)

all_results = []
for res in output:
    all_results.append(res.json)   # ✅ 這就是 dict，可直接 dump

out_path = Path("./output/rec_all.json")
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"Saved {len(all_results)} results to {out_path}")
