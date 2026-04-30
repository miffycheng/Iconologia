"""
OCR inference with rednote-hilab/dots.ocr
Using HuggingFace inference (not vLLM server).

Key notes from official doc:
  - Must git clone the repo AND pip install -e . to get dots_ocr.utils
  - Model weights downloaded separately via: python3 tools/download_model.py
  - Weights saved to ./weights/DotsOCR  (NO periods in directory name!)
  - model_path points to the LOCAL weights dir, NOT a HF repo ID
  - HF inference pattern: tokenize=False → process_vision_info → processor() (2-step, like Qwen2-VL)
  - prompt_ocr = plain OCR text only (no bboxes, no layout categories)
  - max_new_tokens=24000 (from doc)

Setup (from official doc):
    conda create -n dots_ocr python=3.12
    conda activate dots_ocr
    git clone https://github.com/rednote-hilab/dots.ocr.git
    cd dots.ocr
    pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
    pip install -e .
    python3 tools/download_model.py   # downloads to ./weights/DotsOCR

Usage:
    # Run from inside the cloned dots.ocr repo directory
    python run_dotsocr.py --image_dir ./images --output_dir ./results
    python run_dotsocr.py --image_dir ./images --output_dir ./results --weights_dir ./weights/DotsOCR
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info
from dots_ocr.utils import dict_promptmode_to_prompt

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# prompt_ocr = plain text OCR only, no layout bboxes
# Other options: prompt_layout_all_en (full parse), prompt_layout_only_en (detection only)
PROMPT_MODE = "prompt_ocr"


def load_model(weights_dir: str):
    print(f"[INFO] Loading dots.ocr from {weights_dir} ...")
    model = AutoModelForCausalLM.from_pretrained(
        weights_dir,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(weights_dir, trust_remote_code=True)
    return model, processor


def infer(image_path: Path, model, processor) -> str:
    prompt = dict_promptmode_to_prompt[PROMPT_MODE]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text",  "text": prompt},
            ],
        }
    ]

    # HF inference pattern from official doc: tokenize=False first, then processor()
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=24000)

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
    parser = argparse.ArgumentParser(description="OCR with dots.ocr")
    parser.add_argument("--image_dir",   default="./images",          help="Folder with input images")
    parser.add_argument("--output_dir",  default="./results",         help="Folder for JSON output")
    parser.add_argument("--weights_dir", default="./weights/DotsOCR", help="Path to downloaded model weights (NO periods in name!)")
    return parser.parse_args()


def main():
    args = parse_args()

    weights_dir = Path(args.weights_dir)
    if not weights_dir.exists():
        raise FileNotFoundError(
            f"Model weights not found at {weights_dir}\n"
            "Run: python3 tools/download_model.py"
        )

    image_dir  = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS])
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    print(f"[INFO] Found {len(images)} images.")

    model, processor = load_model(str(weights_dir))
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
            "image":       img_path.name,
            "model":       "rednote-hilab/dots.ocr",
            "prompt_mode": PROMPT_MODE,
            "ocr_text":    text,
            "status":      status,
            "elapsed_s":   round(time.time() - t0, 2),
        })

    out_file = output_dir / "ocr_dotsocr.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[DONE] → {out_file}")


if __name__ == "__main__":
    main()
