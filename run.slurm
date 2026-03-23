#!/bin/bash
#SBATCH --job-name=llm-adh
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module load anaconda/anaconda-2024.10
conda activate robotics

cd ~/Research/LLM_Structure_Adherence
python run_experiments.py