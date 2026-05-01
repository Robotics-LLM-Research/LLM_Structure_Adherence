#!/bin/bash
#SBATCH --job-name=bt_smoke
#SBATCH --account=abedi
#SBATCH --qos=abedi
#SBATCH --partition=highgpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
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

python -u - <<'PY'
import torch

print("torch import ok")
print("torch:", torch.__version__)
print("torch cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside the SLURM GPU job.")

print("gpu:", torch.cuda.get_device_name(0))

import vllm
print("vllm:", vllm.__version__)

from vllm import LLM, SamplingParams
print("vllm basic imports ok")

try:
    from vllm.sampling_params import StructuredOutputsParams
    print("StructuredOutputsParams available")
except Exception as error:
    print("StructuredOutputsParams unavailable:", repr(error))

try:
    from vllm.sampling_params import GuidedDecodingParams
    print("GuidedDecodingParams available")
except Exception as error:
    print("GuidedDecodingParams unavailable:", repr(error))

import src.model
print("src.model import ok")
PY