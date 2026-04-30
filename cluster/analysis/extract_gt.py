import json
import re
from pathlib import Path
from lxml import etree

# 圖片名稱前綴 → XML 檔名對照表
BOOK_NAME_MAP = {
    "1026A428_pdf_1-510": "1026A428_pdf_1-510",
    "Amsterdam_1644_Dirck_Pieter_Pers": "Amsterdam_1644_Dirck_Pieter_Pers",
    "Amsterdam_1657_ca__Cornelis_Danckerts": "Amsterdam_1657_ca._Cornelis_Danckerts",
    "Augsburg_1760_ca__Georg_Hertel": "Augsburg_1760_ca._Georg_Hertel",
    "Delft_1743-50_Hubert_Korneliszoon_Poot": "Delft_1743-50_Hubert_Korneliszoon_Poot",
    "London_1709_P__Tempest": "London_1709_P._Tempest",
    "London_1777-1778_George_Richardsonv1": "London_1777-1778_George_Richardsonv1",
    "London_1777-1778_George_Richardsonv2": "London_1777-1778_George_Richardsonv2",
    "Padova_1611_Cesare_Ripa": "Padova_1611_Cesare_Ripa",
    "Padova_1625_Cesare_Ripa": "Padova_1625_Cesare_Ripa",
    "Parma_1759_J__B__Boudardv1": "Parma_1759_J._B._Boudardv1",
    "Perugia_1764-67_Cesare_Orlandiv1": "Perugia_1764-67_Cesare_Orlandiv1",
}

def get_page_text(xml_path, page_num):
    tree = etree.parse(xml_path)
    root = tree.getroot()

    texts = []
    prefix = f"{page_num}_"

    for elem in root.iter():
        tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
        if tag == 'lb':
            n = elem.get('n', '')
            if n.startswith(prefix):
                if elem.tail and elem.tail.strip():
                    texts.append(elem.tail.strip())

    return ' '.join(texts)

def extract_book_and_page(image_id):
    name = image_id.split('_png.rf.')[0]
    parts = name.rsplit('_', 1)
    book_name = parts[0]
    page_num = int(parts[1]) - 1  # ← 改這裡！
    xml_name = BOOK_NAME_MAP.get(book_name, book_name)
    return xml_name, page_num

# 路徑設定
exp_dir = Path(__file__).resolve().parent.parent  # exp/analysis/ → exp/
xml_dir = exp_dir / "PDF"
results_dir = exp_dir / "results"

# 載入 metadata
with open(exp_dir / 'metadata.json') as f:
    metadata = json.load(f)

# 載入 OCR 結果
ocr_results = {}
for model_file in results_dir.glob("*.json"):
    model_name = model_file.stem
    with open(model_file) as f:
        data = json.load(f)
    ocr_results[model_name] = {item['image']: item['ocr_text'] for item in data}

# 組合結果
final_results = []
for item in metadata:
    image_id = item['id']
    book_name, page_num = extract_book_and_page(image_id)
    xml_path = xml_dir / f"{book_name}.xml"

    if xml_path.exists():
        ground_truth = get_page_text(xml_path, page_num)
    else:
        ground_truth = None
        print(f"[WARN] XML not found: {xml_path}")

    entry = {
        'image': image_id,
        'language': item['language'],
        'layout': item['layout'],
        'ground_truth': ground_truth,
    }

    for model_name, model_data in ocr_results.items():
        entry[model_name] = model_data.get(image_id, None)

    final_results.append(entry)

# 存檔
output_path = exp_dir / 'results_with_gt.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(final_results, f, ensure_ascii=False, indent=2)

print(f"Done! → {output_path}")