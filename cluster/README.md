# Historical OCR Benchmark — Iconologia Corpus

Evaluating OCR models on early modern printed historical documents from the *Iconologia* corpus, across multiple languages and page layout types.

---

## Project Structure

```
cluster/
├── images/                  # Input images (18 pages)
├── PDF/                     # TEI-XML ground truth files (one per book)
├── results/                 # OCR output JSON files (one per model)
├── scripts/                 # Inference scripts for each model
│   ├── run_qwen35.py
│   ├── run_qwen3vl8b.py
│   ├── run_firered.py
│   └── run_dotsocr.py
├── slurm/                   # SLURM job scripts for Campus Cluster
│   ├── submit_qwen35.sh
│   ├── submit_qwen3vl8b.sh
│   ├── submit_firered.sh
│   ├── submit_dotsocr.sh
│   └── submit_analysis.sh   # Runs extract_gt.py after inference jobs finish
├── analysis/                # Analysis scripts
│   ├── extract_gt.py        # Extract ground truth from TEI-XML
│   └── analyze_ocr.ipynb    # Results analysis notebook
├── requirements/            # Per-environment requirements files
│   ├── requirements_qwen35.txt
│   ├── requirements_qwen3vl8b.txt
│   ├── requirements_firered.txt
│   ├── setup_dotsocr.sh     # Setup script for dots.ocr (needs editable install)
│   └── requirements_analysis.txt
├── setup.sh                 # One-shot environment setup (run on login node)
├── run_pipeline.sh          # One-command pipeline: submits all jobs sequentially
├── metadata.json            # Image metadata (language + layout labels)
├── results_with_gt.json     # Combined OCR outputs + ground truth
└── results_cleaned.csv      # Cleaned results with length ratios
```

---

## Dataset

18 pages selected from 11 books in the Iconologia corpus, covering:

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

The 18 benchmark images are in `images/` and the TEI-XML ground truth files are in `PDF/`. Both are committed to this repo. The images are a curated subset of the Iconologia Roboflow dataset; filenames and metadata are in `metadata.json`.

---

## Models

| Model | Description | max_new_tokens |
|---|---|---|
| Qwen/Qwen3.5-9B-Instruct | General-purpose VLM (baseline) | 4096 |
| small-models-for-glam/Qwen3-VL-8B-catmus | LoRA fine-tune on medieval Latin manuscripts | 256 |
| FireRedTeam/FireRed-OCR | Document-specialized OCR model | 8192 |
| rednote-hilab/dots.ocr | Document-specialized OCR model | 24000 |

---

## Setup

### Prerequisites

- UIUC Campus Cluster account with IllinoisComputes-GPU allocation
- HuggingFace account (for gated models)
- All setup commands below must be run on the **login node** — compute nodes have no internet access

### Option — Automated Setup (recommended)

Run once on the login node to create all environments, clone repos, and download weights:

```bash
bash setup.sh
```

Follow the printed manual steps (HF login + weight pre-download) before submitting jobs.

### Step 1 — Clone extra repos and download weights (login node)

**FireRed-OCR** (required by `run_firered.py`):
```bash
git clone https://github.com/FireRedTeam/FireRed-OCR.git ~/FireRed-OCR
```

**dots.ocr** (required by `run_dotsocr.py`): use the provided setup script, which handles the pinned PyTorch version, editable install, and weight download:
```bash
bash requirements/setup_dotsocr.sh
# Clones to ~/dots.ocr and downloads weights to ~/dots.ocr/weights/DotsOCR
```

### Step 2 — Create conda environments

Each model requires its own environment. Use the requirements files in `requirements/`:

**Qwen3.5**
```bash
conda create -n ocr_qwen35 python=3.11 -y
conda activate ocr_qwen35
pip install -r requirements/requirements_qwen35.txt
```

**GLAM (Qwen3-VL-8B-catmus)**
```bash
conda create -n ocr_qwen3vl8b python=3.11 -y
conda activate ocr_qwen3vl8b
pip install -r requirements/requirements_qwen3vl8b.txt
```

**FireRed-OCR**
```bash
conda create -n ocr_firered python=3.11 -y
conda activate ocr_firered
pip install -r requirements/requirements_firered.txt
```

**dots.ocr** — handled by `setup_dotsocr.sh` in Step 1 above.

**Analysis** (for `extract_gt.py` and `analyze_ocr.ipynb`, also used by the pipeline)
```bash
conda create -n ocr_analysis python=3.11 -y
conda activate ocr_analysis
pip install -r requirements/requirements_analysis.txt
```

### Step 3 — HuggingFace login

Required for gated models (Qwen3.5, GLAM). Run once per environment on the login node:

```bash
conda activate ocr_qwen35
python -c "from huggingface_hub import login; login(token='your_token')"

conda activate ocr_qwen3vl8b
python -c "from huggingface_hub import login; login(token='your_token')"
```

### Step 4 — Pre-download model weights (login node)

Run each inference script once on the login node with `--help` to trigger the weight download, or run a single image:

```bash
conda activate ocr_qwen35
python scripts/run_qwen35.py --image_dir images --output_dir /tmp/test_out

conda activate ocr_qwen3vl8b
python scripts/run_qwen3vl8b.py --image_dir images --output_dir /tmp/test_out
```

Weights are cached to `/scratch/$USER/.cache/huggingface` (set via `--hf_cache_dir`).

---

## Running Experiments

### Option A — Full Pipeline (recommended)

One command submits all jobs sequentially — each job waits for the previous one to succeed before starting:

```bash
cd cluster/
mkdir -p logs
bash run_pipeline.sh
```

Job execution order:
```
submit_qwen35.sh
        ↓  (afterok)
submit_qwen3vl8b.sh
        ↓  (afterok)
submit_firered.sh
        ↓  (afterok)
submit_dotsocr.sh
        ↓  (afterok)
submit_analysis.sh  →  results_with_gt.json
```

Monitor progress:
```bash
squeue -u $USER
```

After the analysis job finishes, open `analysis/analyze_ocr.ipynb` and run all cells to produce `results_cleaned.csv`.

To skip a model, comment out its block in `run_pipeline.sh` and update the `--dependency` of the next block to point to the last active job ID variable.

> The `--account` header in each SLURM script is set to `carboni-ic`. Update it to your own allocation if needed.

### Option B — Submit Jobs Individually

```bash
cd cluster/
mkdir -p logs
sbatch slurm/submit_qwen35.sh
sbatch slurm/submit_qwen3vl8b.sh
sbatch slurm/submit_firered.sh
sbatch slurm/submit_dotsocr.sh
```

Then after all jobs finish, run the analysis manually:
```bash
sbatch slurm/submit_analysis.sh
```

### Option C — Run Scripts Directly (without SLURM)

```bash
python scripts/run_qwen35.py    --image_dir images --output_dir results
python scripts/run_qwen3vl8b.py --image_dir images --output_dir results
python scripts/run_firered.py   --image_dir images --output_dir results --repo_dir ~/FireRed-OCR
python scripts/run_dotsocr.py   --image_dir images --output_dir results --weights_dir ~/dots.ocr/weights/DotsOCR
```

Optional `--hf_cache_dir` flag redirects the HuggingFace model cache:

```bash
python scripts/run_qwen35.py --image_dir images --output_dir results \
    --hf_cache_dir /scratch/$USER/.cache
```

---

## Analysis

### 1. Extract ground truth from TEI-XML

```bash
pip install -r requirements/requirements_analysis.txt
python analysis/extract_gt.py
```

Output: `results_with_gt.json`

### 2. Run analysis notebook

```bash
jupyter notebook analysis/analyze_ocr.ipynb
```

Run all cells. Output: `results_cleaned.csv`

---

## Authors

Miffy Cheng — University of Illinois Urbana-Champaign
