# Historical OCR Benchmark — Iconologia Corpus

Evaluating OCR and Vision Language Models (VLMs) on early modern printed historical documents from the *Iconologia* corpus, across multiple languages and page layout types.

---

## Repository Structure

```
Research/
├── cluster/                    # Formal benchmark (SLURM cluster, 4 models, 18 pages)
│   ├── images/                 # Input images
│   ├── PDF/                    # TEI-XML ground truth files (one per book)
│   ├── scripts/                # Inference scripts (run on cluster)
│   ├── slurm/                  # SLURM job submission scripts
│   ├── analysis/               # Ground truth extraction + analysis notebook
│   ├── results/                # OCR output JSON files (one per model)
│   ├── requirements/           # Per-environment requirements files
│   ├── setup.sh                # One-shot environment setup (run on login node)
│   ├── run_pipeline.sh         # Submit all jobs as a sequential SLURM pipeline
│   ├── metadata.json           # Image metadata (language + layout labels)
│   ├── results_with_gt.json    # OCR outputs joined with ground truth
│   └── results_cleaned.csv     # Cleaned results with length ratios
│
├── models_scripts/             # API-based and local VLM inference scripts
│   ├── gemma.py                # google/gemma-3-27b-it via HF Inference Router
│   ├── glm.py                  # zai-org/GLM-4.6V-Flash via HF Inference Router
│   ├── llama4.py               # meta-llama/Llama-4-Maverick-17B via HF Inference Router
│   ├── qwen8B.py               # Qwen/Qwen3-VL-8B-Instruct via HF Inference Router
│   ├── small-models-for-glam-2b.py  # Qwen3-VL-2B + CATMuS LoRA, runs locally on CPU
│   ├── run_pipeline.sh         # Run all 5 models one by one on a single image folder
│   ├── requirements_api.txt    # Dependencies for API-based models
│   ├── requirements_local.txt  # Dependencies for local model
│   └── paddle-ocr/             # PaddleOCR detection/crop/recognition pipeline
│
├── ground_truth_extraction/    # TEI-XML → JSON ground truth tooling
│   ├── extract_page.py         # Extract one page's ground truth from TEI-XML
│   └── image_gt.json           # Example output
│
├── ocr_results/                # JSON outputs from models_scripts/ runs
├── data/                       # Raw book JPGs and TEI-XML files (gitignored)
└── pdf2image.py                # Convert a PDF to per-page JPG images
```

---

## Dataset

The *Iconologia* corpus consists of illustrated emblem books published between 1611 and 1778 across Europe. Pages span 6 languages and 4 layout types.

The formal benchmark (`cluster/`) uses **18 pages** from 11 books:

| Language | Pages |
|---|---|
| Italian (it) | 6 |
| Dutch (nl) | 4 |
| English (en) | 3 |
| French (fr) | 1 |
| German (de) | 1 |
| Latin (la) | 1 |

| Layout | Pages |
|---|---|
| image_with_caption | 6 |
| mixed_complex | 6 |
| text_heavy_structured | 4 |
| single_column | 2 |

Ground truth is extracted from TEI-XML transcriptions stored in `cluster/PDF/`. See [Ground Truth Extraction](#ground-truth-extraction) for details.

---

## Part 1 — API-Based VLM Scripts (`models_scripts/`)

These scripts call models via the [HuggingFace Inference Router](https://huggingface.co/docs/inference-endpoints) and do not require a GPU.

### Requirements

```bash
pip install -r models_scripts/requirements_api.txt      # API models
pip install -r models_scripts/requirements_local.txt    # small-models-for-glam-2b only
```

### Setup

Set your HuggingFace token:

```bash
export HF_TOKEN=your_token_here
```

### Usage — Run Individual Models

All four API scripts take a folder of images as a positional argument and save results to `ocr_results/`:

```bash
python models_scripts/gemma.py   /path/to/images   # google/gemma-3-27b-it
python models_scripts/glm.py     /path/to/images   # zai-org/GLM-4.6V-Flash
python models_scripts/llama4.py  /path/to/images   # meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
python models_scripts/qwen8B.py  /path/to/images   # Qwen/Qwen3-VL-8B-Instruct
```

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`

Output is written to `ocr_results/ocr_results_<model>.json`:

```json
{
  "page_01.jpg": { "status": "success", "text": "Extracted text..." },
  "page_02.jpg": { "status": "error",   "text": "Error message..." }
}
```

### Usage — Run All Models as a Pipeline

```bash
bash models_scripts/run_pipeline.sh /path/to/images
```

Runs all 5 models one by one (gemma → glm → llama4 → qwen8B → small-models-for-glam-2b). To skip a model, comment out its block in `run_pipeline.sh`.

### Local Model — CATMuS LoRA (CPU)

`small-models-for-glam-2b.py` runs Qwen3-VL-2B with the CATMuS LoRA adapter locally on CPU. Designed for medieval manuscript transcription. Supports **resuming** — re-running skips already-processed images.

```bash
pip install -r models_scripts/requirements_local.txt
python models_scripts/small-models-for-glam-2b.py /path/to/images
```

Output: `ocr_results/transcriptions.json`

---

## Part 2 — Cluster Benchmark (`cluster/`)

Four models evaluated on the 18-page benchmark dataset, using SLURM jobs on the UIUC Campus Cluster.

### Models

| Model | Description | Script |
|---|---|---|
| `Qwen/Qwen3.5-9B-Instruct` | General-purpose VLM (baseline) | `run_qwen35.py` |
| `small-models-for-glam/Qwen3-VL-8B-catmus` | LoRA fine-tune on medieval Latin manuscripts | `run_qwen3vl8b.py` |
| `FireRedTeam/FireRed-OCR` | Document-specialized OCR model | `run_firered.py` |
| `rednote-hilab/dots.ocr` | Document-specialized OCR model | `run_dotsocr.py` |

### Environment Setup

Run once on the **login node** before submitting any jobs:

```bash
bash cluster/setup.sh
```

This creates all five conda environments (`ocr_qwen35`, `ocr_qwen3vl8b`, `ocr_firered`, `dots_ocr`, `ocr_analysis`) and clones the FireRed-OCR repo. Manual steps (HF login, weight pre-download) are printed at the end.

To set up environments individually, requirements files are in `cluster/requirements/`:

**Qwen3.5-9B**
```bash
conda create -n ocr_qwen35 python=3.11 -y
conda activate ocr_qwen35
pip install -r cluster/requirements/requirements_qwen35.txt
```

**GLAM (Qwen3-VL-8B-catmus)**
```bash
conda create -n ocr_qwen3vl8b python=3.11 -y
conda activate ocr_qwen3vl8b
pip install -r cluster/requirements/requirements_qwen3vl8b.txt
```

**FireRed-OCR**
```bash
conda create -n ocr_firered python=3.11 -y
conda activate ocr_firered
pip install -r cluster/requirements/requirements_firered.txt
git clone https://github.com/FireRedTeam/FireRed-OCR.git ~/FireRed-OCR
```

**dots.ocr** — use the provided setup script (handles pinned PyTorch, editable install, and weight download):
```bash
bash cluster/requirements/setup_dotsocr.sh
```

### HuggingFace Login

Required for gated models (Qwen3.5, GLAM). Run once per environment on the login node:

```bash
conda activate ocr_qwen35
python -c "from huggingface_hub import login; login(token='your_token')"

conda activate ocr_qwen3vl8b
python -c "from huggingface_hub import login; login(token='your_token')"
```

> **Note:** Compute nodes have no internet access. All model weights must be downloaded on the login node before submitting jobs.

### Running the Pipeline on UIUC Campus Cluster

```bash
cd cluster/
bash run_pipeline.sh
```

Jobs run sequentially — each waits for the previous to succeed before starting. Update `SLURM_ACCOUNT` in `run_pipeline.sh` if running under a different allocation (default: `carboni-ic`).

Check job status:
```bash
squeue -u $USER
```

Results are saved to `cluster/results/ocr_<model>.json`. Logs go to `cluster/logs/`.

### Running Scripts Directly (without SLURM)

Each inference script can also be run standalone:

```bash
python cluster/scripts/run_qwen35.py    --image_dir cluster/images --output_dir cluster/results
python cluster/scripts/run_qwen3vl8b.py --image_dir cluster/images --output_dir cluster/results
python cluster/scripts/run_firered.py   --image_dir cluster/images --output_dir cluster/results --repo_dir ~/FireRed-OCR
python cluster/scripts/run_dotsocr.py   --image_dir cluster/images --output_dir cluster/results --weights_dir ~/dots.ocr/weights/DotsOCR
```

Optional `--hf_cache_dir` flag lets you redirect the HuggingFace model cache (useful on clusters with limited home directory quota):

```bash
python cluster/scripts/run_qwen35.py --image_dir cluster/images --output_dir cluster/results \
    --hf_cache_dir /scratch/$USER/.cache
```

### Analysis

**Step 1 — Extract ground truth from TEI-XML:**
```bash
python cluster/analysis/extract_gt.py
```
Produces `cluster/results_with_gt.json`, combining OCR outputs with ground truth text.

**Step 2 — Run analysis notebook:**

Open `cluster/analysis/analyze_ocr.ipynb` and run all cells.

Produces `cluster/results_cleaned.csv` with output lengths and length ratios per model.

---

## Part 3 — PaddleOCR Pipeline (`models_scripts/paddle-ocr/`)

A three-stage detection → crop → recognition pipeline using PaddleOCR PP-OCRv5. Intended for single-page exploration.

### Requirements

```bash
pip install paddlepaddle paddleocr opencv-python
```

### Usage

Edit the path constants at the top of each script, then run in order:

```bash
# 1. Detect text regions → output/res_det.json
python models_scripts/paddle-ocr/paddle_det.py

# 2. Crop detected regions → crops_simple/
python models_scripts/paddle-ocr/paddle_crop.py

# 3. Recognize text in crops → output/rec_all.json
python models_scripts/paddle-ocr/paddle_rec.py
```

Alternatively, `paddle_detrec.py` runs detection and recognition in one step:

```bash
python models_scripts/paddle-ocr/paddle_detrec.py
```

---

## Ground Truth Extraction

Ground truth text is stored in TEI-XML files following the convention:

```
<lb n="PAGE_BLOCK_LINE"/> text of the line
```

`ground_truth_extraction/extract_page.py` extracts one page's text given its image filename and corresponding XML:

```bash
python ground_truth_extraction/extract_page.py \
    --xml  data/1026A428_pdf_1-510.xml \
    --image data/Ripa_v2/test/1026A428_pdf_1-510_352_png.rf.e065fc1d114e613b0a8991da4101f184.jpg \
    --output ground_truth_extraction/image_gt.json
```

The page number is derived from the image filename: `_352_png` → XML page `n=351` (1-based image index → 0-based XML page).

Output JSON fields: `image_filename`, `image_index`, `page_n`, `entry_number`, `title`, `lines` (grouped by XML tag), `full_text`.

---

## PDF to Image Conversion

`pdf2image.py` converts a PDF to per-page JPGs using PyMuPDF:

```bash
pip install pymupdf
```

Edit the `pdf_path` and `output_dir` arguments at the bottom of `pdf2image.py`, then run:

```bash
python pdf2image.py
```

Default resolution: 300 DPI.

---

## Authors

Miffy Cheng — University of Illinois Urbana-Champaign
