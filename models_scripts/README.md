# VLM OCR Scripts

A collection of scripts for extracting text from images using various Vision Language Models (VLMs). Includes API-based scripts (via HuggingFace Inference Router), a local CPU script, and a PaddleOCR pipeline.

---

## Scripts

| Script | Model | Backend |
|---|---|---|
| `gemma.py` | google/gemma-3-27b-it | HF Inference Router (featherless-ai) |
| `glm.py` | zai-org/GLM-4.6V-Flash | HF Inference Router (novita) |
| `llama4.py` | meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 | HF Inference Router (novita) |
| `qwen8B.py` | Qwen/Qwen3-VL-8B-Instruct | HF Inference Router (novita) |
| `small-models-for-glam-2b.py` | Qwen3-VL-2B + CATMuS LoRA adapter | local (CPU) |
| `paddle-ocr/` | PP-OCRv5 detection + recognition | local |

---

## API-Based Scripts (gemma, glm, llama4, qwen8B)

### Requirements

```bash
pip install openai
```

### Setup

Set your HuggingFace token:

```bash
export HF_TOKEN=your_token_here
```

### Usage

All four scripts take the image folder as a positional argument:

```bash
python gemma.py  /path/to/images
python glm.py    /path/to/images
python llama4.py /path/to/images
python qwen8B.py /path/to/images
```

**Supported image formats:** `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`

### Output

Results are saved to `ocr_results/` in the repository root (one level up from `models_scripts/`):

- `gemma.py`  → `ocr_results/ocr_results_gemma.json`
- `glm.py`    → `ocr_results/ocr_results_glm.json`
- `llama4.py` → `ocr_results/ocr_results_llama4.json`
- `qwen8B.py` → `ocr_results/ocr_results_qwen8B.json`

Each JSON maps image filenames to their result:

```json
{
  "image001.jpg": { "status": "success", "text": "Extracted text content..." },
  "image002.jpg": { "status": "error",   "text": "Error message..." }
}
```

---

## Local Model — CATMuS LoRA (`small-models-for-glam-2b.py`)

Runs Qwen3-VL-2B with the [CATMuS](https://huggingface.co/small-models-for-glam/Qwen3-VL-2B-catmus) fine-tuned LoRA adapter for medieval manuscript transcription. Runs on CPU — no GPU required.

Supports **resuming**: if interrupted, re-running will skip already-processed images.

### Requirements

```bash
pip install torch transformers peft pillow
```

### Usage

```bash
python small-models-for-glam-2b.py /path/to/images
```

### Output

`ocr_results/transcriptions.json`

---

## PaddleOCR Pipeline (`paddle-ocr/`)

A three-stage detection → crop → recognition pipeline using [PaddleOCR PP-OCRv5](https://github.com/PaddlePaddle/PaddleOCR). Intended for single-page exploration.

### Requirements

```bash
pip install paddlepaddle paddleocr opencv-python
```

### Usage

Edit the path constants at the top of each script, then run in order:

```bash
# Stage 1: Detect text regions → output/res_det.json
python paddle-ocr/paddle_det.py

# Stage 2: Crop detected regions → crops_simple/
python paddle-ocr/paddle_crop.py

# Stage 3: Recognize text in crops → output/rec_all.json
python paddle-ocr/paddle_rec.py
```

Alternatively, `paddle_detrec.py` runs detection and recognition in one step (no intermediate crops):

```bash
python paddle-ocr/paddle_detrec.py
```
