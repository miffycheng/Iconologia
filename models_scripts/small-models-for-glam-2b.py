"""
Medieval Manuscript Transcription Script
Uses Qwen3-VL-2B with CATMuS fine-tune (LoRA adapter)
Processes a folder of images and exports results to JSON
"""

import os
import argparse
import json
import time
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

# ── Configuration ─────────────────────────────────────────────────────────────
OUTPUT_JSON = "../ocr_results/transcriptions.json"
BASE_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
ADAPTER_MODEL = "small-models-for-glam/Qwen3-VL-2B-catmus"
MAX_NEW_TOKENS = 1024
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
# ──────────────────────────────────────────────────────────────────────────────


def load_model():
    print("Loading base model... (this may take a few minutes on CPU)")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    print("Loading CATMuS LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_MODEL)
    model.eval()

    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    print("Model ready.\n")
    return model, processor


def transcribe_image(image_path, model, processor):
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Extract all text visible in this image. Return only the extracted text, nothing else."},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    # Move all tensors to CPU explicitly
    inputs = {k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    transcription = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return transcription.strip()


def main():
    parser = argparse.ArgumentParser(description="Transcribe medieval manuscript images using a VLM.")
    parser.add_argument("image_dir", type=str, help="Path to the folder containing images")
    args = parser.parse_args()

    # Collect image paths
    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"ERROR: Image directory '{args.image_dir}' not found.")
        return

    image_paths = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not image_paths:
        print(f"No images found in '{args.image_dir}'")
        return

    print(f"Found {len(image_paths)} images to process.\n")

    # Load existing results if resuming
    results = {}
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            results = json.load(f)
        already_done = sum(1 for v in results.values() if v["status"] == "success")
        print(f"Resuming — {already_done} images already processed.\n")

    model, processor = load_model()

    for i, image_path in enumerate(image_paths, 1):
        filename = image_path.name

        # Skip already processed
        if filename in results and results[filename]["status"] == "success":
            print(f"[{i}/{len(image_paths)}] Skipping (already done): {filename}")
            continue

        print(f"[{i}/{len(image_paths)}] Processing: {filename}")
        start = time.time()

        try:
            text = transcribe_image(image_path, model, processor)
            elapsed = time.time() - start
            results[filename] = {"status": "success", "text": text}
            print(f"  ✓ Done in {elapsed:.1f}s")

        except Exception as e:
            results[filename] = {"status": "error", "error": str(e)}
            print(f"  ✗ Error: {e}")

        # Save after every image (safe resuming)
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    # Final summary
    success = sum(1 for v in results.values() if v["status"] == "success")
    errors = sum(1 for v in results.values() if v["status"] == "error")
    print(f"\n── Done ──────────────────────────────")
    print(f"  Successful: {success}")
    print(f"  Errors:     {errors}")
    print(f"  Output:     {OUTPUT_JSON}")


if __name__ == "__main__":
    main()