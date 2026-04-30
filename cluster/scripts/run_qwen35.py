"""
OCR inference with Qwen/Qwen3.5-9B-Instruct
Official class: Qwen3_5ForConditionalGeneration

Key notes from official doc:
  - Qwen3.5 small series (0.8B, 2B, 4B, 9B): thinking is DISABLED by default
  - enable_thinking goes directly into apply_chat_template (not extra_body, that's API-only)
  - Use tokenize=True + return_dict=True in apply_chat_template (not tokenize=False + separate processor call)
  - pop token_type_ids to avoid errors
  - Recommended sampling for non-thinking/general: temperature=0.7, top_p=0.8, top_k=20

Setup:
    pip install "transformers>=4.51.0" torchvision pillow accelerate

Usage:
    python run_qwen35.py --image_dir ./images --output_dir ./results
    python run_qwen35.py --image_dir ./images --output_dir ./results --hf_cache_dir /scratch/$USER/.cache
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import Qwen3_5ForConditionalGeneration, AutoProcessor

MODEL_ID = "Qwen/Qwen3.5-9B-Instruct"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
OCR_PROMPT = "Please perform OCR on this image and return all the text you can read, preserving the original layout as much as possible."


def load_model():
    print(f"[INFO] Loading {MODEL_ID} ...")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def infer(image_path: Path, model, processor) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text",  "text": OCR_PROMPT},
            ],
        }
    ]

    # For Qwen3.5 9B: thinking is already OFF by default.
    # enable_thinking=False is explicit here for clarity / future-proofing.
    # tokenize=True + return_dict=True is the correct pattern for Qwen3.5
    # (different from Qwen2-VL which used tokenize=False + separate processor call)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )

    # Remove token_type_ids if present — Qwen3.5 does not use it
    inputs.pop("token_type_ids", None)

    # Move all tensor inputs to the model's device
    inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            # Recommended sampling params for non-thinking general tasks (official doc)
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            do_sample=True,
        )

    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs["input_ids"], generated_ids)
    ]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def parse_args():
    parser = argparse.ArgumentParser(description=f"OCR with {MODEL_ID}")
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
            "image":     img_path.name,
            "model_id":  MODEL_ID,
            "ocr_text":  text,
            "status":    status,
            "elapsed_s": round(time.time() - t0, 2),
        })

    out_file = output_dir / "ocr_qwen35.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[DONE] → {out_file}")


if __name__ == "__main__":
    main()
