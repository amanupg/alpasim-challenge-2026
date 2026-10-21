#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH -J as_chk
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s|A6000|R6000
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /insomnia001/depts/edu/users/au2327/alpasim/logs/check_forced_%j.out
# forced <traj_future_start> (A15_FORCE_TRAJ_START) vs the library rollout on the 12 validation clips: runs, latency,
# memory, and how far the trajectories move (they should differ only where the token was missing)
source /etc/profile
A=/insomnia001/depts/edu/users/au2327/alpasim; NR=/insomnia001/depts/edu/users/au2327/nureasoning
source $A/cluster/a15env_local.sh
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
run() { cd $A && env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=$A/envs/overlay_driver:$A/envs/overlay_torchao:$A/src $A15PY -u src/check_policy.py "$@" 2>&1 | grep -v -i warning; }
run embed_cpu=1 $A/tmp/chk_base_now.npz
run embed_cpu=1,force_start=1 $A/tmp/chk_forced.npz $A/tmp/chk_base_now.npz
