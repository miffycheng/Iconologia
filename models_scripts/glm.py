import os
import argparse
import base64
import json
from pathlib import Path
from openai import OpenAI

# --- Config ---
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Output filename based on this script's name (e.g. glm.py -> ocr_results_glm.json)
SCRIPT_NAME = Path(__file__).stem
OUTPUT_FILE = f"../ocr_results/ocr_results_{SCRIPT_NAME}.json"

# --- Client ---
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],  # or replace with your token string directly
)

def encode_image(image_path: str) -> tuple[str, str]:
    """Encode image to base64 and detect media type."""
    ext = Path(image_path).suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    media_type = media_type_map.get(ext, "image/jpeg")
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return encoded, media_type

def extract_text(image_path: str) -> str:
    """Send image to VLM and extract text."""
    encoded, media_type = encode_image(image_path)

    completion = client.chat.completions.create(
        model="zai-org/GLM-4.6V-Flash:novita",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract all text visible in this image. Return only the extracted text, nothing else."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{encoded}"
                        }
                    }
                ]
            }
        ],
    )
    return completion.choices[0].message.content

def main():
    parser = argparse.ArgumentParser(description="OCR images using a VLM.")
    parser.add_argument("image_folder", type=str, help="Path to the folder containing images")
    args = parser.parse_args()
    image_folder = Path(args.image_folder)

    if not image_folder.exists():
        print(f"❌ Folder not found: {image_folder}")
        return

    # Get all supported images
    image_files = [
        f for f in image_folder.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_files:
        print(f"❌ No images found in {image_folder}")
        return

    print(f"Found {len(image_files)} images. Starting OCR...")
    print(f"Output will be saved to: {OUTPUT_FILE}\n")

    results = {}

    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Processing: {image_path.name}")
        try:
            text = extract_text(str(image_path))
            results[image_path.name] = {
                "status": "success",
                "text": text
            }
            print(f"  ✅ Extracted: {text[:80]}{'...' if len(text) > 80 else ''}\n")
        except Exception as e:
            results[image_path.name] = {
                "status": "error",
                "text": str(e)
            }
            print(f"  ❌ Error: {e}\n")

    # Save results to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
