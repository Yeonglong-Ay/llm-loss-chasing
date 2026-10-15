#!/bin/bash

# Name of the job
#SBATCH --job-name=llm-chase

# One array task per model (5 models), at most 3 running at once
#SBATCH --array=0-4%3

# Partition and account (from sinfo / sacctmgr on Andes)
#SBATCH --partition=standard
#SBATCH --account=free

# Number of compute nodes
#SBATCH --nodes=1

# Number of tasks, in this case one
#SBATCH --ntasks-per-node=1

# One CPU is enough: the job only waits on API calls
#SBATCH --cpus-per-task=1

# No GPU needed for this job
##SBATCH --partition gpuq
##SBATCH --gres=gpu:1

#SBATCH --mem=4G

# Walltime (job duration)
#SBATCH --time=16:00:00

# Log files: %x = job name, %A = array job ID, %a = task index
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Email notifications (one email for the whole array, not one per task)
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=albert99302@gmail.com

# ---- job starts here ----
MODELS=(
  "google.gemma-3-27b-it"
  "google.gemma-4-31B-it"
  "openai.gpt-oss-120b"
  "meta.llama-3.2-11b-vision-instruct"
  "zai-org.glm-5.3"
)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm-chase
cd "$SLURM_SUBMIT_DIR"

echo "Model: $MODEL  Host: $(hostname)  Start: $(date)"

# Stop early if this node can't reach the API
python src/net_check.py || { echo "Network check failed on $(hostname)"; exit 1; }

python -u src/runner.py --models "$MODEL" "$@"
echo "End: $(date)"
