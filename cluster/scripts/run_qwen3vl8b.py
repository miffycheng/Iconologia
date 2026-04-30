"""
OCR inference with small-models-for-glam/Qwen3-VL-8B-catmus
A LoRA fine-tune of Qwen/Qwen3-VL-8B-Instruct for medieval Latin manuscript transcription.

Key notes from official doc:
  - Base model:    Qwen/Qwen3-VL-8B-Instruct
  - LoRA adapter:  small-models-for-glam/Qwen3-VL-8B-catmus
  - Must load base model first, then PeftModel.from_pretrained() for the adapter
  - Processor comes from the BASE model, not the adapter repo
  - apply_chat_template: tokenize=True, return_dict=True, return_tensors="pt"
  - Prompt: "Transcribe the text shown in this image." (from doc)

Setup:
    pip install "transformers>=4.57.1" peft pillow accelerate

Usage:
    python run_qwen3vl8b.py --image_dir ./images --output_dir ./results
    python run_qwen3vl8b.py --image_dir ./images --output_dir ./results --hf_cache_dir /scratch/$USER/.cache
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

BASE_MODEL_ID    = "Qwen/Qwen3-VL-8B-Instruct"
ADAPTER_MODEL_ID = "small-models-for-glam/Qwen3-VL-8B-catmus"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Prompt from official doc
OCR_PROMPT = "Transcribe the text shown in this image."


def load_model():
    print(f"[INFO] Loading base model: {BASE_MODEL_ID} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )

    print(f"[INFO] Loading LoRA adapter: {ADAPTER_MODEL_ID} ...")
    model = PeftModel.from_pretrained(model, ADAPTER_MODEL_ID)

    # Processor comes from the base model repo
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

    return model, processor


def infer(image_path: Path, model, processor) -> str:
    image = Image.open(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": OCR_PROMPT},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=4096)

    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def parse_args():
    parser = argparse.ArgumentParser(description=f"OCR with {ADAPTER_MODEL_ID}")
    parser.add_argument("--image_dir",    default="./images",  help="Folder with input images")
    parser.add_argument("--output_dir",   default="./results", help="Folder for JSON output")
    parser.add_argument("--hf_cache_dir", default=None,        help="Custom HF cache dir, e.g. /scratch/$USER/.cache")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.hf_cache_dir:
        os.environ["HF_HOME"] = args.hf_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = args.hf_cache_dir

    image_dir  = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    print(f"[INFO] Found {len(images)} images.")

    model, processor = load_model()
    model.eval()

    results = []
    for idx, img_path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] {img_path.name}", flush=True)
        t0 = time.time()
        try:
            text   = infer(img_path, model, processor)
            status = "ok"
        except Exception as e:
            text   = ""
            status = f"error: {e}"
            print(f"  [WARN] {e}")

        results.append({
            "image":          img_path.name,
            "base_model_id":  BASE_MODEL_ID,
            "adapter_model":  ADAPTER_MODEL_ID,
            "ocr_text":       text,
            "status":         status,
            "elapsed_s":      round(time.time() - t0, 2),
        })

    out_file = output_dir / "ocr_qwen3vl8b.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[DONE] → {out_file}")


if __name__ == "__main__":
    main()
