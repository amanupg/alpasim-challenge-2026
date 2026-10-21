#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH -J as_ft
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 05:00:00
#SBATCH -o /insomnia001/depts/edu/users/au2327/alpasim/logs/ft_train_%j.out
# Fine-tune Alpamayo 1.5's action expert on captured rendered samples (src/a15_ft/train.py).
#   DATA=<dir of capture tars>  OUT=<ckpt.pt>  ARGS="--steps N --lr 1e-5 ..." (ARGS="--eval_only" = baseline check)
source /etc/profile
A=/insomnia001/depts/edu/users/au2327/alpasim; NR=/insomnia001/depts/edu/users/au2327/nureasoning
source $A/cluster/a15env_local.sh
export HUB=$NR/.cache/hf/hub HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
cd $A && PYTHONPATH=$A/envs/overlay_driver:$A/envs/overlay_torchao:$A/src \
  $A15PY -u -m a15_ft.train --data ${DATA:-$A/data/ft_capture} --out ${OUT:-$A/data/ft_ckpt/expert_$SLURM_JOB_ID.pt} $ARGS 2>&1 \
  | grep -v -i "warning"
