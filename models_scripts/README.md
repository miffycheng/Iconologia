# VLM OCR Scripts

Scripts for extracting text from historical document images using Vision Language Models. Includes API-based scripts (via HuggingFace Inference Router), a local CPU model, and a PaddleOCR pipeline.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements_api.txt       # API models (gemma, glm, llama4, qwen8B)
pip install -r requirements_local.txt     # local model (small-models-for-glam-2b)

# 2. Set HuggingFace token (required for all API models)
export HF_TOKEN=your_token_here

# 3. Run all models on one image or a folder
bash run_pipeline.sh /path/to/image_or_folder
```

Results are written to `ocr_results/` (one JSON file per model).

---

## Models

| Script | Model | Backend |
|---|---|---|
| `gemma.py` | google/gemma-3-27b-it | HF Inference Router |
| `glm.py` | zai-org/GLM-4.6V-Flash | HF Inference Router |
| `llama4.py` | meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 | HF Inference Router |
| `qwen8B.py` | Qwen/Qwen3-VL-8B-Instruct | HF Inference Router |
| `small-models-for-glam-2b.py` | Qwen3-VL-2B + CATMuS LoRA adapter | local (CPU) |

---

## Running the Pipeline

`run_pipeline.sh` runs all 5 models one by one. Accepts a single image file or a folder:

```bash
bash run_pipeline.sh /path/to/image.jpg       # single image
bash run_pipeline.sh /path/to/images/         # all images in folder
```

To skip a model, comment out its block in `run_pipeline.sh`. To add a new model, add a new block following the same pattern.

---

## Running Individual Scripts

Each script also works standalone and accepts a single image or a folder:

```bash
python models_scripts/gemma.py   /path/to/image_or_folder
python models_scripts/glm.py     /path/to/image_or_folder
python models_scripts/llama4.py  /path/to/image_or_folder
python models_scripts/qwen8B.py  /path/to/image_or_folder
python models_scripts/small-models-for-glam-2b.py /path/to/image_or_folder
```

**Supported image formats:** `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`

---

## Output

Results are saved to `ocr_results/` in the repository root:

| Script | Output file |
|---|---|
| `gemma.py` | `ocr_results/ocr_results_gemma.json` |
| `glm.py` | `ocr_results/ocr_results_glm.json` |
| `llama4.py` | `ocr_results/ocr_results_llama4.json` |
| `qwen8B.py` | `ocr_results/ocr_results_qwen8B.json` |
| `small-models-for-glam-2b.py` | `ocr_results/transcriptions.json` |

Each JSON maps image filenames to their result:

```json
{
  "image001.jpg": { "status": "success", "text": "Extracted text..." },
  "image002.jpg": { "status": "error",   "text": "Error message..." }
}
```

`small-models-for-glam-2b.py` supports **resuming** — re-running skips already-processed images.

---

## Notes

- **API models** (gemma, glm, llama4, qwen8B): fast, no GPU needed, require `HF_TOKEN`. Large images are automatically resized to 1280px before sending.
- **Local model** (small-models-for-glam-2b): runs on CPU, no token needed, slow (~minutes per image). Fine-tuned for medieval manuscripts.
- **PaddleOCR** (`paddle-ocr/`): separate detection/recognition pipeline, see `paddle-ocr/` for usage. Not included in `run_pipeline.sh`.
