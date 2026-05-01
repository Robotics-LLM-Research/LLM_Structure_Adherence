#!/bin/bash
#SBATCH --job-name=bt_online_tasks
#SBATCH --account=abedi
#SBATCH --qos=abedi
#SBATCH --partition=highgpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd ~/Research/LLM_Structure_Adherence
mkdir -p logs

module purge
module load anaconda/anaconda-2024.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bt-vllm-py312

export PYTHONPATH=$PWD

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

python -u -m jobs.job_bt_online_tasks