import json
from pathlib import Path
from paddleocr import PaddleOCR

IMG_DIR = Path("./data/Amsterdam_1698_Jean_Baudoin/selected/page_31.jpg")  # 放 page_XX.jpg 的資料夾
OUT_DIR = Path("./output_detrec")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ocr = PaddleOCR(use_angle_cls=False, lang="en")  # 先用 en，之後可換多語/其他

for img_path in sorted(IMG_DIR.glob("*.jpg")):
    result = ocr.ocr(str(img_path), cls=False)

    # result 格式：result[0] = [ [poly, (text, score)], ... ]
    items = []
    for i, item in enumerate(result[0] if result else []):
        poly = item[0]               # [[x,y], [x,y], [x,y], [x,y]]
        text, score = item[1]        # text, confidence
        items.append({
            "id": i,
            "polygon": [[int(x), int(y)] for x, y in poly],
            "text": text,
            "score": float(score),
        })

    out_json = OUT_DIR / f"{img_path.stem}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {"input_path": str(img_path), "n_items": len(items), "items": items},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[OK] {img_path.name}: {len(items)} text boxes -> {out_json}")
