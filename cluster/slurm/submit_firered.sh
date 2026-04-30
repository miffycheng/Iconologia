#!/bin/bash
#SBATCH --job-name=ocr_firered
#SBATCH --account=carboni-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:A100
#SBATCH --output=logs/ocr_firered_%j.out
#SBATCH --error=logs/ocr_firered_%j.err
##SBATCH --mail-user=carboni@illinois.edu
##SBATCH --mail-type=BEGIN,END

echo "========================================"
echo "  JOB  : $SLURM_JOB_ID"
echo "  MODEL: FireRed-OCR"
echo "  NODE : $SLURMD_NODENAME"
echo "  TIME : $(date)"
echo "========================================"

# Activate conda — adjust the source path to match your cluster's anaconda install
# UIUC Campus Cluster:
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
# Generic alternative (use if the line above doesn't work):
# module load anaconda3
conda activate ocr_firered

export HF_HOME=/scratch/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "${SLURM_DIR}")"
mkdir -p "$HF_HOME" "${EXP_DIR}/logs"

FIRERED_REPO="${HOME}/FireRed-OCR"   # ← adjust to where you cloned the repo

python "${EXP_DIR}/scripts/run_firered.py" \
    --image_dir    "${EXP_DIR}/images" \
    --output_dir   "${EXP_DIR}/results" \
    --repo_dir     "${FIRERED_REPO}" \
    --hf_cache_dir "$HF_HOME"

echo "[INFO] Done at $(date)"
