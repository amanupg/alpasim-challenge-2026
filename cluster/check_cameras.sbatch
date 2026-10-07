#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH -J as_chkcam
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:40:00
#SBATCH -o /insomnia001/depts/edu/users/au2327/alpasim/logs/check_cameras_%j.out
# 4 vs 6 PAI cameras: driver-process GPU memory and latency on synthetic frames (no simulator needed)
source /etc/profile
A=/insomnia001/depts/edu/users/au2327/alpasim; NR=/insomnia001/depts/edu/users/au2327/nureasoning
cd $A && env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HUB=$NR/.cache/hf/hub \
  PYTHONPATH=$A/envs/overlay_driver:$A/envs/overlay_torchao:$A/src \
  $NR/ext/a15env/bin/python -u src/check_cameras.py 2>&1 | grep -v -i "warning\|Loading checkpoint"
