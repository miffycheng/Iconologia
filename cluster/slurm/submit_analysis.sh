#!/bin/bash
#SBATCH --job-name=ocr_analysis
#SBATCH --account=carboni-ic
#SBATCH --partition=IllinoisComputes
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=logs/ocr_analysis_%j.out
#SBATCH --error=logs/ocr_analysis_%j.err

echo "========================================"
echo "  JOB  : $SLURM_JOB_ID"
echo "  STEP : Extract GT + combine results"
echo "  NODE : $SLURMD_NODENAME"
echo "  TIME : $(date)"
echo "========================================"

# Activate conda — adjust the source path to match your cluster's anaconda install
# UIUC Campus Cluster:
source /sw/apps/anaconda3/2024.10/etc/profile.d/conda.sh
# Generic alternative (use if the line above doesn't work):
# module load anaconda3
conda activate ocr_analysis   # needs: pandas, lxml  (pip install -r requirements/requirements_analysis.txt)

SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "${SLURM_DIR}")"

python "${EXP_DIR}/analysis/extract_gt.py"

echo "[INFO] Output: ${EXP_DIR}/results_with_gt.json"
echo "[INFO] Done at $(date)"
echo ""
echo "Next step: open analysis/analyze_ocr.ipynb and run all cells to produce results_cleaned.csv"
