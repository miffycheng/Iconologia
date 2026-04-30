import io
import os
import argparse
import base64
import json
from pathlib import Path
from PIL import Image
from openai import OpenAI

# --- Config ---
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Output filename based on this script's name (e.g. glm.py -> ocr_results_glm.json)
SCRIPT_NAME = Path(__file__).stem
OUTPUT_FILE = Path(__file__).parent.parent / "ocr_results" / f"ocr_results_{SCRIPT_NAME}.json"

# --- Client ---
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],  # or replace with your token string directly
)

MAX_IMAGE_PX = 1280  # longest side; keeps payload under API size limits

def encode_image(image_path: str) -> tuple[str, str]:
    """Resize if needed, then encode image to base64 as JPEG."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_IMAGE_PX:
        scale = MAX_IMAGE_PX / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return encoded, "image/jpeg"

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
    parser.add_argument("image_path", type=str, help="Path to a single image file or a folder of images")
    args = parser.parse_args()
    input_path = Path(args.image_path)

    if not input_path.exists():
        print(f"❌ Not found: {input_path}")
        return

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"❌ Unsupported file type '{input_path.suffix}'")
            return
        image_files = [input_path]
    else:
        image_files = [
            f for f in input_path.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        ]

    if not image_files:
        print(f"❌ No images found in {input_path}")
        return

    print(f"Found {len(image_files)} image(s). Starting OCR...")
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
