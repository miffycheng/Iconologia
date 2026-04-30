#!/bin/bash
# OCR Benchmark Pipeline — UIUC Campus Cluster
#
# Runs all models one by one: each job starts only after the previous one
# succeeds, then the analysis step runs at the end.
#
# Usage (run from cluster/ directory):
#   bash run_pipeline.sh
#
# To skip a model, comment out its block below.
# To add a new model, add a new block following the same pattern.

set -e

# ── Configure ──────────────────────────────────────────────────────────────────
SLURM_ACCOUNT="carboni-ic"   # ← change to your allocation
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
mkdir -p logs

echo "Account : ${SLURM_ACCOUNT}"
echo "Submitting jobs (each waits for the previous to succeed)..."
echo ""

# ── Step 1: Qwen3.5-9B ────────────────────────────────────────────────────────
JID_1=$(sbatch --parsable --account=${SLURM_ACCOUNT} \
    slurm/submit_qwen35.sh)
echo "  [1] Qwen3.5-9B-Instruct     → job ${JID_1}"

# ── Step 2: Qwen3-VL-8B-catmus (GLAM LoRA) ───────────────────────────────────
JID_2=$(sbatch --parsable --account=${SLURM_ACCOUNT} \
    --dependency=afterok:${JID_1} \
    slurm/submit_qwen3vl8b.sh)
echo "  [2] Qwen3-VL-8B-catmus      → job ${JID_2}  (after ${JID_1})"

# ── Step 3: FireRed-OCR ───────────────────────────────────────────────────────
JID_3=$(sbatch --parsable --account=${SLURM_ACCOUNT} \
    --dependency=afterok:${JID_2} \
    slurm/submit_firered.sh)
echo "  [3] FireRed-OCR             → job ${JID_3}  (after ${JID_2})"

# ── Step 4: dots.ocr ─────────────────────────────────────────────────────────
JID_4=$(sbatch --parsable --account=${SLURM_ACCOUNT} \
    --dependency=afterok:${JID_3} \
    slurm/submit_dotsocr.sh)
echo "  [4] dots.ocr                → job ${JID_4}  (after ${JID_3})"

# ── Step 5: Analysis (extract GT + combine results) ──────────────────────────
JID_5=$(sbatch --parsable --account=${SLURM_ACCOUNT} \
    --dependency=afterok:${JID_4} \
    slurm/submit_analysis.sh)
echo "  [5] Analysis                → job ${JID_5}  (after ${JID_4})"

echo ""
echo "Pipeline submitted. Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "Final output : results/results_with_gt.json"
echo "Then run     : jupyter notebook analysis/analyze_ocr.ipynb"
