"""
OCR inference with FireRedTeam/FireRed-OCR
Based on Qwen3-VL architecture.

Key notes from official doc:
  - Model loaded directly from HF: FireRedTeam/FireRed-OCR
  - Processor also from HF: FireRedTeam/FireRed-OCR
  - Requires git clone https://github.com/FireRedTeam/FireRed-OCR.git
    for conv_for_infer.py which contains generate_conv()
  - generate_conv(image_path) builds the message format
  - apply_chat_template: tokenize=True, return_dict=True, return_tensors="pt"
  - inputs.to(model.device) — not hardcoded "cuda"
  - max_new_tokens=8192 (from doc)

Setup:
    pip install transformers qwen-vl-utils accelerate
    git clone https://github.com/FireRedTeam/FireRed-OCR.git

    # This script must be run from inside the FireRed-OCR repo directory,
    # OR the repo must be on PYTHONPATH so conv_for_infer can be imported.

Usage:
    # From inside the cloned repo:
    python run_firered.py --image_dir ./images --output_dir ./results

    # From anywhere else:
    PYTHONPATH=/path/to/FireRed-OCR python run_firered.py --image_dir ./images --output_dir ./results
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

MODEL_ID = "FireRedTeam/FireRed-OCR"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_model():
    print(f"[INFO] Loading {MODEL_ID} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor


def infer(image_path: Path, model, processor) -> str:
    # generate_conv() is defined in FireRed-OCR/conv_for_infer.py
    # It builds the correct message format for this model
    from conv_for_infer import generate_conv

    messages = generate_conv(str(image_path))

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=8192)

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
    parser = argparse.ArgumentParser(description=f"OCR with {MODEL_ID}")
    parser.add_argument("--image_dir",    default="./images",  help="Folder with input images")
    parser.add_argument("--output_dir",   default="./results", help="Folder for JSON output")
    parser.add_argument("--repo_dir",     default=None,        help="Path to cloned FireRed-OCR repo (if not already on PYTHONPATH)")
    parser.add_argument("--hf_cache_dir", default=None,        help="Custom HF cache dir, e.g. /scratch/$USER/.cache")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.hf_cache_dir:
        os.environ["HF_HOME"] = args.hf_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = args.hf_cache_dir

    # Add repo to sys.path so conv_for_infer can be imported
    if args.repo_dir:
        repo_path = Path(args.repo_dir)
        if not repo_path.exists():
            raise FileNotFoundError(
                f"FireRed-OCR repo not found at {repo_path}\n"
                "Run: git clone https://github.com/FireRedTeam/FireRed-OCR.git"
            )
        sys.path.insert(0, str(repo_path))

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

    out_file = output_dir / "ocr_firered.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[DONE] → {out_file}")


if __name__ == "__main__":
    main()
