#!/bin/bash
# Local VLM Pipeline
#
# Runs all API-based and local VLM scripts one by one on a single image folder.
# Results are written to ocr_results/ (one JSON file per model).
#
# Usage:
#   bash models_scripts/run_pipeline.sh /path/to/images
#
# Prerequisites:
#   pip install -r models_scripts/requirements_api.txt      # API models
#   pip install -r models_scripts/requirements_local.txt    # small-models-for-glam-2b
#   export HF_TOKEN=your_token_here                         # required for all models
#
# To skip a model, comment out its block below.
# To add a new model, add a new block following the same pattern.

set -e

# ── Arguments ─────────────────────────────────────────────────────────────────
IMAGE_DIR="${1}"

if [ -z "${IMAGE_DIR}" ]; then
    echo "Usage: bash $(basename "$0") /path/to/images"
    exit 1
fi

if [ ! -d "${IMAGE_DIR}" ]; then
    echo "Error: image directory not found: ${IMAGE_DIR}"
    exit 1
fi

# ── Check HF_TOKEN ────────────────────────────────────────────────────────────
if [ -z "${HF_TOKEN}" ]; then
    echo "Error: HF_TOKEN is not set."
    echo "  export HF_TOKEN=your_token_here"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================="
echo "  Local VLM Pipeline"
echo "=================================="
echo "Image dir : ${IMAGE_DIR}"
echo ""

# ── Step 1: Gemma-3-27B ───────────────────────────────────────────────────────
echo "[1/5] gemma-3-27b-it..."
python "${SCRIPT_DIR}/gemma.py" "${IMAGE_DIR}"
echo "      Done."
echo ""

# ── Step 2: GLM-4.6V-Flash ────────────────────────────────────────────────────
echo "[2/5] GLM-4.6V-Flash..."
python "${SCRIPT_DIR}/glm.py" "${IMAGE_DIR}"
echo "      Done."
echo ""

# ── Step 3: Llama-4-Maverick ─────────────────────────────────────────────────
echo "[3/5] Llama-4-Maverick-17B..."
python "${SCRIPT_DIR}/llama4.py" "${IMAGE_DIR}"
echo "      Done."
echo ""

# ── Step 4: Qwen3-VL-8B ──────────────────────────────────────────────────────
echo "[4/5] Qwen3-VL-8B-Instruct..."
python "${SCRIPT_DIR}/qwen8B.py" "${IMAGE_DIR}"
echo "      Done."
echo ""

# ── Step 5: small-models-for-glam-2b (local, CPU) ────────────────────────────
echo "[5/5] Qwen3-VL-2B + CATMuS LoRA (local)..."
python "${SCRIPT_DIR}/small-models-for-glam-2b.py" "${IMAGE_DIR}"
echo "      Done."
echo ""

echo "=================================="
echo "  Pipeline complete!"
echo "=================================="
echo ""
echo "Results written to: ocr_results/"
