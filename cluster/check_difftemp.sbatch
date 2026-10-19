#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH -J as_chk
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:40:00
#SBATCH -o /insomnia001/depts/edu/users/au2327/alpasim/logs/check_difftemp_%j.out
# diffusion temperature 0.6 vs 1.0 on the validation clips: runs, memory, trajectory change; also K=6 route chooser at 0.6
source /etc/profile
A=/insomnia001/depts/edu/users/au2327/alpasim; NR=/insomnia001/depts/edu/users/au2327/nureasoning
source $A/cluster/a15env_local.sh   # Alpamayo env on node-local disk (nureasoning/ext/a15env deleted 2026-10-19)
run() { cd $A && env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=$A/envs/overlay_driver:$A/envs/overlay_torchao:$A/src \
  $A15PY -u src/check_policy.py "$@" 2>&1 | grep -v Warning; }
run embed_cpu=1 $A/tmp/chk_t10.npz
run embed_cpu=1,diffusion_temperature=0.6 $A/tmp/chk_t06.npz $A/tmp/chk_t10.npz
run embed_cpu=1,diffusion_temperature=0.6,num_samples=6,chooser=route $A/tmp/chk_t06k6.npz $A/tmp/chk_t06.npz
