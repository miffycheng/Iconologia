#!/bin/bash
# Setup script for dots.ocr environment.
# dots.ocr requires an editable install from its own repo and pinned PyTorch versions,
# so it cannot be fully captured in a plain requirements.txt.
#
# Run this script once on the login node before submitting the SLURM job.
# Usage:
#   bash setup_dotsocr.sh [/path/to/clone/dots.ocr]
#
# The repo will be cloned to ~/dots.ocr by default.

DOTS_REPO="${1:-${HOME}/dots.ocr}"

set -e

echo "[1/4] Creating conda environment: dots_ocr (Python 3.12)"
conda create -n dots_ocr python=3.12 -y

echo "[2/4] Cloning dots.ocr repo to ${DOTS_REPO}"
git clone https://github.com/rednote-hilab/dots.ocr.git "${DOTS_REPO}"

echo "[3/4] Installing dependencies"
conda run -n dots_ocr pip install \
    torch==2.7.0 \
    torchvision==0.22.0 \
    torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

conda run -n dots_ocr pip install -e "${DOTS_REPO}"

echo "[4/4] Downloading model weights"
conda run -n dots_ocr python3 "${DOTS_REPO}/tools/download_model.py"

echo ""
echo "Done! Weights saved to: ${DOTS_REPO}/weights/DotsOCR"
echo "Update DOTS_REPO in slurm/submit_dotsocr.sh if you cloned to a non-default path."
