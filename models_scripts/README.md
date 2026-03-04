# VLM OCR Scripts

A collection of scripts for extracting text from images using various Vision Language Models (VLMs) via the HuggingFace Inference Router. Each script targets a different model so you can benchmark or compare OCR quality across them.

## Scripts

| Script | Model | Backend |
|---|---|---|
| `gemma.py` | google/gemma-3-27b-it | featherless-ai |
| `glm.py` | zai-org/GLM-4.6V-Flash | novita |
| `llama4.py` | meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 | novita |
| `qwen8B.py` | Qwen/Qwen3-VL-8B-Instruct | novita |
| `small-models-for-glam-2b.py` | Qwen3-VL-2B + CATMuS LoRA adapter | local (CPU) |

## Requirements

Install dependencies:

```bash
pip install openai
```

For `small-models-for-glam-2b.py` only:

```bash
pip install torch transformers peft pillow
```

## Setup

The API-based scripts (`gemma`, `glm`, `llama4`, `qwen8B`) authenticate via the HuggingFace Inference Router. Set your token as an environment variable:

```bash
export HF_TOKEN=your_token_here
```

## Usage

All scripts take the image folder as a positional argument:

```bash
python gemma.py /path/to/images
python glm.py /path/to/images
python llama4.py /path/to/images
python qwen8B.py /path/to/images
python small-models-for-glam-2b.py /path/to/images
```

**Supported image formats:** `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp` (`.tif`/`.tiff` also supported in `small-models-for-glam-2b.py`)

## Output

Each script saves results to a JSON file in the current directory:

- `gemma.py` → `ocr_results_gemma.json`
- `glm.py` → `ocr_results_glm.json`
- `llama4.py` → `ocr_results_llama4.json`
- `qwen8B.py` → `ocr_results_qwen8B.json`
- `small-models-for-glam-2b.py` → `transcriptions.json`

Each JSON file maps image filenames to their result:

```json
{
  "image001.jpg": {
    "status": "success",
    "text": "Extracted text content..."
  },
  "image002.jpg": {
    "status": "error",
    "text": "Error message..."
  }
}
```

## Notes

- **`small-models-for-glam-2b.py`** runs locally on CPU and uses a CATMuS fine-tuned LoRA adapter optimized for medieval manuscript transcription. It supports **resuming** — if interrupted, re-running it will skip already-processed images.
- All other scripts call the HuggingFace Inference Router API and require a valid `HF_TOKEN`.
