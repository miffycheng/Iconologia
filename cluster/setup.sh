#!/bin/bash
# Cluster Setup Script
#
# Creates all conda environments and installs dependencies.
# Run once on the LOGIN NODE before submitting any jobs.
#
# Usage:
#   bash setup.sh
#
# After this script finishes, complete the manual steps printed at the end.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load conda (adjust path if needed for your cluster)
# UIUC Campus Cluster:
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
# Generic alternative:
# module load anaconda3 && source $(conda info --base)/etc/profile.d/conda.sh

echo "======================================"
echo "  OCR Benchmark — Environment Setup"
echo "======================================"
echo ""

# ── Environment 1: Qwen3.5-9B ─────────────────────────────────────────────────
echo "[1/5] Creating ocr_qwen35..."
conda create -n ocr_qwen35 python=3.11 -y
conda run -n ocr_qwen35 pip install -r "${SCRIPT_DIR}/requirements/requirements_qwen35.txt"
echo "      Done."
echo ""

# ── Environment 2: Qwen3-VL-8B-catmus (GLAM LoRA) ────────────────────────────
echo "[2/5] Creating ocr_qwen3vl8b..."
conda create -n ocr_qwen3vl8b python=3.11 -y
conda run -n ocr_qwen3vl8b pip install -r "${SCRIPT_DIR}/requirements/requirements_qwen3vl8b.txt"
echo "      Done."
echo ""

# ── Environment 3: FireRed-OCR ────────────────────────────────────────────────
echo "[3/5] Creating ocr_firered..."
conda create -n ocr_firered python=3.11 -y
conda run -n ocr_firered pip install -r "${SCRIPT_DIR}/requirements/requirements_firered.txt"
echo "      Cloning FireRed-OCR repo..."
if [ ! -d "${HOME}/FireRed-OCR" ]; then
    git clone https://github.com/FireRedTeam/FireRed-OCR.git "${HOME}/FireRed-OCR"
else
    echo "      ~/FireRed-OCR already exists, skipping clone."
fi
echo "      Done."
echo ""

# ── Environment 4: dots.ocr ───────────────────────────────────────────────────
echo "[4/5] Creating dots_ocr..."
bash "${SCRIPT_DIR}/requirements/setup_dotsocr.sh"
echo "      Done."
echo ""

# ── Environment 5: Analysis ───────────────────────────────────────────────────
echo "[5/5] Creating ocr_analysis..."
conda create -n ocr_analysis python=3.11 -y
conda run -n ocr_analysis pip install -r "${SCRIPT_DIR}/requirements/requirements_analysis.txt"
echo "      Done."
echo ""

# ── Manual steps ─────────────────────────────────────────────────────────────
echo "======================================"
echo "  Setup complete!"
echo "======================================"
echo ""
echo "MANUAL STEPS REMAINING:"
echo ""
echo "1. Log in to HuggingFace (required for gated models: Qwen3.5, GLAM):"
echo ""
echo "     conda activate ocr_qwen35"
echo "     python -c \"from huggingface_hub import login; login(token='YOUR_TOKEN')\""
echo ""
echo "     conda activate ocr_qwen3vl8b"
echo "     python -c \"from huggingface_hub import login; login(token='YOUR_TOKEN')\""
echo ""
echo "2. Pre-download model weights on the login node (compute nodes have no internet):"
echo ""
echo "     conda activate ocr_qwen35"
echo "     python scripts/run_qwen35.py --image_dir images --output_dir /tmp/test"
echo ""
echo "     conda activate ocr_qwen3vl8b"
echo "     python scripts/run_qwen3vl8b.py --image_dir images --output_dir /tmp/test"
echo ""
echo "     conda activate ocr_firered"
echo "     python scripts/run_firered.py --image_dir images --output_dir /tmp/test --repo_dir ~/FireRed-OCR"
echo ""
echo "     (dots.ocr weights already downloaded by setup_dotsocr.sh)"
echo ""
echo "3. Update SLURM_ACCOUNT in run_pipeline.sh if needed (currently: carboni-ic)"
echo ""
echo "Then run the pipeline:"
echo "  bash run_pipeline.sh"
