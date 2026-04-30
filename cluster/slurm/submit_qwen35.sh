#!/bin/bash
#SBATCH --job-name=ocr_qwen35
#SBATCH --account=carboni-ic
#SBATCH --partition=IllinoisComputes-GPU
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --gres=gpu:A100
#SBATCH --output=logs/ocr_qwen35_%j.out
#SBATCH --error=logs/ocr_qwen35_%j.err
##SBATCH --mail-user=carboni@illinois.edu
##SBATCH --mail-type=BEGIN,END

echo "========================================"
echo "  JOB  : $SLURM_JOB_ID"
echo "  MODEL: Qwen3.5-9B-Instruct"
echo "  NODE : $SLURMD_NODENAME"
echo "  TIME : $(date)"
echo "========================================"

# Activate conda — adjust the source path to match your cluster's anaconda install
# UIUC Campus Cluster:
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
# Generic alternative (use if the line above doesn't work):
# module load anaconda3
conda activate ocr_qwen35

export HF_HOME=/scratch/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "${SLURM_DIR}")"
mkdir -p "$HF_HOME" "${EXP_DIR}/logs"

python "${EXP_DIR}/scripts/run_qwen35.py" \
    --image_dir    "${EXP_DIR}/images" \
    --output_dir   "${EXP_DIR}/results" \
    --hf_cache_dir "$HF_HOME"

echo "[INFO] Done at $(date)"
