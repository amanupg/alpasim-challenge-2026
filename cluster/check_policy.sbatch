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
  $NR/ext/a15env/bin/python -u src/check_policy.py embed_cpu=1,cuda_graph=0 $A/tmp/chk_base.npz 2>&1 | grep -v Warning
cd $A && env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=$A/envs/overlay_driver:$A/envs/overlay_torchao:$A/src \
  $NR/ext/a15env/bin/python -u src/check_policy.py embed_cpu=1,cuda_graph=1 $A/tmp/chk_graph.npz $A/tmp/chk_base.npz 2>&1 | grep -v Warning
