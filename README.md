# Historical OCR Benchmark — Iconologia Corpus

This project tests how well AI models can read and transcribe pages from *Iconologia*, a series of illustrated books published across Europe between 1611 and 1778. The pages are written in six different languages and come in a variety of layouts — some are dense columns of text, some pair images with captions, and some mix both.

We run several AI models on 18 sample pages, then compare their output to hand-verified transcriptions to see which model performs best.

---

## How the Project is Organized

```
Research/
├── cluster/                    # GPU cluster benchmark (4 models, 18 pages)
│   ├── images/                 # The 18 book page images used for testing
│   ├── xml/                    # Correct transcriptions in TEI-XML format (one file per book)
│   ├── scripts/                # Scripts that run each AI model on the cluster
│   ├── slurm/                  # Job submission scripts for the university's GPU cluster
│   ├── analysis/               # Scripts that extract correct answers and compare results
│   ├── results/                # Where AI outputs are saved (one JSON file per model)
│   ├── requirements/           # Software dependencies for each model environment
│   ├── setup.sh                # One-time setup script (run before anything else)
│   ├── run_pipeline.sh         # Runs all four models back-to-back automatically
│   ├── metadata.json           # Labels for each image (language and layout type)
│   ├── results_with_gt.json    # AI outputs combined with the correct transcriptions
│   └── results_cleaned.csv     # Final cleaned results ready for analysis
│
├── models_scripts/             # Internet-based AI model scripts (no GPU needed)
│   ├── gemma.py                # Google Gemma 3 (27B)
│   ├── glm.py                  # Zhipu GLM-4.6V Flash
│   ├── llama4.py               # Meta Llama 4 Maverick (17B)
│   ├── qwen8B.py               # Alibaba Qwen3-VL-8B
│   ├── small-models-for-glam-2b.py  # Qwen3-VL-2B with CATMuS fine-tune, runs locally on CPU
│   ├── run_pipeline.sh         # Runs all 5 models one by one on a folder of images
│   ├── requirements_api.txt    # Dependencies for the internet-based models
│   ├── requirements_local.txt  # Dependencies for the local model
│
├── ground_truth_extraction/    # Tools for pulling correct text out of TEI-XML files
│   ├── extract_page.py         # Extracts one page's ground truth given an image and XML file
│   └── image_gt.json           # Example output
│
├── ocr_results/                # Where internet-model outputs are saved
├── data/                       # Raw book images and XML files (not tracked by git)
└── pdf2image.py                # Converts a PDF into individual page images
```

---

## The Pages We Test On

18 pages selected from 11 books in the Iconologia corpus:

| Language | Pages |
|---|---|
| Italian | 6 |
| Dutch | 4 |
| English | 3 |
| French | 1 |
| German | 1 |
| Latin | 1 |

| Layout type | Pages |
|---|---|
| Image with caption | 6 |
| Mixed (image + text blocks) | 6 |
| Structured text columns | 4 |
| Single column of text | 2 |

The 18 images live in `cluster/images/` and the correct transcriptions are in `cluster/xml/`. Both are included in this repository. Image labels (language, layout) are in `metadata.json`.

---

## The AI Models We Test

### Part 1 — Internet-based models (`models_scripts/`)

These call models running on remote servers over the internet. No GPU required. Results are saved to `ocr_results/`.

| Model | What it is | Output file |
|---|---|---|
| Google Gemma 3 (27B) | General-purpose vision AI | `ocr_results/ocr_results_gemma.json` |
| Zhipu GLM-4.6V Flash | General-purpose vision AI | `ocr_results/ocr_results_glm.json` |
| Meta Llama 4 Maverick (17B) | General-purpose vision AI | `ocr_results/ocr_results_llama4.json` |
| Alibaba Qwen3-VL-8B | General-purpose vision AI | `ocr_results/ocr_results_qwen8B.json` |
| GLAM 2B (CATMuS fine-tune) | Fine-tuned for medieval manuscripts, runs locally on CPU | `ocr_results/transcriptions.json` |

### Part 2 — GPU cluster models (`cluster/`)

These run as scheduled jobs on the UIUC Campus Cluster and need a GPU. Results are saved to `cluster/results/`.

| Model | What it is | max_new_tokens | Output file |
|---|---|---|---|
| Qwen3.5-9B | General-purpose vision AI (baseline) | 4096 | `cluster/results/ocr_qwen35.json` |
| GLAM 8B (Qwen3-VL-8B-catmus) | Fine-tuned for medieval manuscripts | 256 | `cluster/results/ocr_qwen3vl8b.json` |
| FireRed-OCR | Built specifically for document OCR | 8192 | `cluster/results/ocr_firered.json` |
| dots.ocr | Built specifically for document OCR | 24000 | `cluster/results/ocr_dotsocr.json` |

---

## Where Results Go

```
Input images
    ↓
Internet models (models_scripts/)  →  ocr_results/ocr_results_<model>.json
GPU cluster models (cluster/)      →  cluster/results/ocr_<model>.json
                                              ↓
                              + correct transcriptions (cluster/xml/)
                                              ↓
                                  cluster/results_with_gt.json
                                              ↓
                                  cluster/results_cleaned.csv
```

---

## Part 1 — Internet-Based Models

### Requirements

```bash
pip install -r models_scripts/requirements_api.txt      # for the four internet-based models
pip install -r models_scripts/requirements_local.txt    # for small-models-for-glam-2b only
```

### Setup

Set your HuggingFace token so the scripts can access gated models:

```bash
export HF_TOKEN=your_token_here
```

### Run All Models at Once

```bash
bash models_scripts/run_pipeline.sh /path/to/images
```

Runs all 5 models one by one (gemma → glm → llama4 → qwen8B → small-models-for-glam-2b). To skip a model, comment out its block in `run_pipeline.sh`.

### Run Individual Models

```bash
python models_scripts/gemma.py   /path/to/images
python models_scripts/glm.py     /path/to/images
python models_scripts/llama4.py  /path/to/images
python models_scripts/qwen8B.py  /path/to/images
```

Accepted image formats: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`

Each script saves output to `ocr_results/ocr_results_<model>.json`:

```json
{
  "page_01.jpg": { "status": "success", "text": "Extracted text..." },
  "page_02.jpg": { "status": "error",   "text": "Error message..." }
}
```

### Local Model — GLAM 2B (CPU)

`small-models-for-glam-2b.py` runs locally on your own machine without needing a GPU. It fine-tuned on medieval manuscripts. It also supports **resuming** — if you stop and restart it, already-processed images are skipped.

```bash
pip install -r models_scripts/requirements_local.txt
python models_scripts/small-models-for-glam-2b.py /path/to/images
```

Output: `ocr_results/transcriptions.json`

---

## Part 2 — GPU Cluster Benchmark

### First-Time Setup

Run this once on the **login node** before submitting any jobs. It creates all five software environments, clones required repos, and prints the manual steps you need to complete (HuggingFace login and model weight downloads):

```bash
bash cluster/setup.sh
```

> Compute nodes on the Campus Cluster have no internet access. All model weights must be downloaded on the login node first.

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

**dots.ocr** — use the provided script, which handles the specific PyTorch version and model weight download:
```bash
bash cluster/requirements/setup_dotsocr.sh
```

**Analysis environment** (for comparing outputs to ground truth):
```bash
conda create -n ocr_analysis python=3.11 -y
conda activate ocr_analysis
pip install -r cluster/requirements/requirements_analysis.txt
```

### HuggingFace Login

Required for gated models (Qwen3.5, GLAM). Run once per environment on the login node:

```bash
conda activate ocr_qwen35
python -c "from huggingface_hub import login; login(token='your_token')"

conda activate ocr_qwen3vl8b
python -c "from huggingface_hub import login; login(token='your_token')"
```

### Running the Full Pipeline

One command submits all four models as a chain — each job waits for the previous one to finish before starting:

```bash
cd cluster/
mkdir -p logs
bash run_pipeline.sh
```

Job order:
```
Qwen3.5 → GLAM → FireRed → dots.ocr → Analysis (extract + combine results)
```

Check job status:
```bash
squeue -u $USER
```

Update `SLURM_ACCOUNT` in `run_pipeline.sh` if running under a different cluster allocation (default: `carboni-ic`).

To skip a model, comment out its block in `run_pipeline.sh` and update the `--dependency` of the next block to point to the last active job.

### Submitting Jobs Individually

```bash
cd cluster/
mkdir -p logs
sbatch slurm/submit_qwen35.sh
sbatch slurm/submit_qwen3vl8b.sh
sbatch slurm/submit_firered.sh
sbatch slurm/submit_dotsocr.sh
```

Then after all jobs finish, run the analysis step:
```bash
sbatch slurm/submit_analysis.sh
```

### Running Without the Cluster (no SLURM)

Each script can also be run directly on any machine with a GPU:

```bash
python cluster/scripts/run_qwen35.py    --image_dir cluster/images --output_dir cluster/results
python cluster/scripts/run_qwen3vl8b.py --image_dir cluster/images --output_dir cluster/results
python cluster/scripts/run_firered.py   --image_dir cluster/images --output_dir cluster/results --repo_dir ~/FireRed-OCR
python cluster/scripts/run_dotsocr.py   --image_dir cluster/images --output_dir cluster/results --weights_dir ~/dots.ocr/weights/DotsOCR
```

Use `--hf_cache_dir` to redirect the model cache to a different location (useful when your home directory has limited space):

```bash
python cluster/scripts/run_qwen35.py --image_dir cluster/images --output_dir cluster/results \
    --hf_cache_dir /scratch/$USER/.cache
```

### Analyzing Results

**Step 1 — Combine AI outputs with the correct transcriptions:**
```bash
python cluster/analysis/extract_gt.py
```
Saves to: `cluster/results_with_gt.json`

**Step 2 — Run the analysis notebook:**

Open `cluster/analysis/analyze_ocr.ipynb` and run all cells.
Saves to: `cluster/results_cleaned.csv`

---


## Ground Truth Extraction

The correct transcriptions are stored in TEI-XML files — a format used by digital humanities scholars to encode historical texts. Each line in the XML looks like:

```
<lb n="PAGE_BLOCK_LINE"/> text of the line
```

`ground_truth_extraction/extract_page.py` pulls out the correct text for one page given its image filename and the corresponding XML file:

```bash
python ground_truth_extraction/extract_page.py \
    --xml  data/1026A428_pdf_1-510.xml \
    --image data/Ripa_v2/test/1026A428_pdf_1-510_352_png.rf.e065fc1d114e613b0a8991da4101f184.jpg \
    --output ground_truth_extraction/image_gt.json
```

The page number is figured out automatically from the image filename: `_352_png` → XML page `n=351`.

Output fields: image filename, image index, page number, entry number, title, lines grouped by XML tag, and the full text.

---

## Converting PDFs to Images

If you have a book as a PDF and want to convert it to individual page images first:

```bash
pip install pymupdf
```

Edit the `pdf_path` and `output_dir` at the bottom of `pdf2image.py`, then run:

```bash
python pdf2image.py
```

Default resolution: 300 DPI.

---

## Authors

Miffy Cheng — University of Illinois Urbana-Champaign
