#!/bin/bash
#SBATCH --job-name=ocr_dotsocr
#SBATCH --account=carboni-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:A100
#SBATCH --output=logs/ocr_dotsocr_%j.out
#SBATCH --error=logs/ocr_dotsocr_%j.err
##SBATCH --mail-user=carboni@illinois.edu
##SBATCH --mail-type=BEGIN,END

echo "========================================"
echo "  JOB  : $SLURM_JOB_ID"
echo "  MODEL: dots.ocr"
echo "  NODE : $SLURMD_NODENAME"
echo "  TIME : $(date)"
echo "========================================"

# Activate conda — adjust the source path to match your cluster's anaconda install
# UIUC Campus Cluster:
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
# Generic alternative (use if the line above doesn't work):
# module load anaconda3
conda activate dots_ocr

pip install flash-attn --no-build-isolation

# dots.ocr needs its repo on PYTHONPATH to import dots_ocr.utils
DOTS_REPO="${HOME}/dots.ocr"   # ← adjust to where you cloned the repo
export PYTHONPATH="${DOTS_REPO}:${PYTHONPATH}"

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "${SLURM_DIR}")"
mkdir -p "${EXP_DIR}/logs"

python "${EXP_DIR}/scripts/run_dotsocr.py" \
    --image_dir   "${EXP_DIR}/images" \
    --output_dir  "${EXP_DIR}/results" \
    --weights_dir "${DOTS_REPO}/weights/DotsOCR"

echo "[INFO] Done at $(date)"
