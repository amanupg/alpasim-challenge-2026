#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J as_chk
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /insomnia001/depts/edu/users/au2327/alpasim/logs/check_policy_%j.out
source /etc/profile
A=/insomnia001/depts/edu/users/au2327/alpasim; NR=/insomnia001/depts/edu/users/au2327/nureasoning
cd $A && env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=$A/envs/overlay_driver:$A/envs/overlay_torchao:$A/src \
  $NR/ext/a15env/bin/python -u src/check_policy.py 0 /tmp/chk_gpu.npz 2>&1 | grep -v Warning
cd $A && env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=$A/envs/overlay_driver:$A/envs/overlay_torchao:$A/src \
  $NR/ext/a15env/bin/python -u src/check_policy.py 1 /tmp/chk_cpu.npz /tmp/chk_gpu.npz 2>&1 | grep -v Warning
