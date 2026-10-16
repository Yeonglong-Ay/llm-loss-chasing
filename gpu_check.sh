#!/bin/bash
#SBATCH --job-name=gpu-check
#SBATCH --partition=gpu_preempt
#SBATCH --account=free
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate /dartfs/rc/lab/S/SunX/AyY/envs/llm-train
cd "$SLURM_SUBMIT_DIR"

echo "Host: $(hostname)"
nvidia-smi
python -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
