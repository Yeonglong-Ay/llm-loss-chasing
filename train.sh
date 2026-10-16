#!/bin/bash
#SBATCH --job-name=lora-train
#SBATCH --array=0-1
#SBATCH --partition=gpu_preempt
#SBATCH --account=free
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --exclude=p[01-04],gv01
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

POLICIES=(chaser flat)
POLICY=${POLICIES[$SLURM_ARRAY_TASK_ID]}

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate /dartfs/rc/lab/S/SunX/AyY/envs/llm-train
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/dartfs/rc/lab/S/SunX/AyY/hf_cache
export HF_HUB_OFFLINE=1      # the model is already downloaded; don't contact Hugging Face
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Policy: $POLICY  Host: $(hostname)  Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -u src/train_lora.py --policy "$POLICY"
echo "End: $(date)"
