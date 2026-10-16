#!/bin/bash
#SBATCH --job-name=lora-eval
#SBATCH --array=0-2
#SBATCH --partition=gpu_preempt
#SBATCH --account=free
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --exclude=p[01-04],gv01
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

NAMES=(base chaser flat)
ADAPTERS=("" "adapters/qwen2.5-1.5b-chaser/final" "adapters/qwen2.5-1.5b-flat/final")
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
ADAPTER=${ADAPTERS[$SLURM_ARRAY_TASK_ID]}

source /optnfs/common/miniconda3/etc/profile.d/conda.sh
conda activate /dartfs/rc/lab/S/SunX/AyY/envs/llm-train
cd "$SLURM_SUBMIT_DIR"
export HF_HOME=/dartfs/rc/lab/S/SunX/AyY/hf_cache
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Model: $NAME  Adapter: ${ADAPTER:-none}  Host: $(hostname)  Start: $(date)"
if [ -n "$ADAPTER" ]; then
  python -u src/run_local.py --name "$NAME" --adapter "$ADAPTER"
else
  python -u src/run_local.py --name "$NAME"
fi
echo "End: $(date)"
