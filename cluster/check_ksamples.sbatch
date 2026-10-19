#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH -J as_chk
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 00:40:00
#SBATCH -o /insomnia001/depts/edu/users/au2327/alpasim/logs/check_ksamples_%j.out
# route chooser with more flow samples: GPU memory (16 GiB contract) and latency for K = 6, 10, 16
source /etc/profile
A=/insomnia001/depts/edu/users/au2327/alpasim; NR=/insomnia001/depts/edu/users/au2327/nureasoning
source $A/cluster/a15env_local.sh   # Alpamayo env on node-local disk (nureasoning/ext/a15env deleted 2026-10-19)
run() { cd $A && env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=$A/envs/overlay_driver:$A/envs/overlay_torchao:$A/src \
  $A15PY -u src/check_policy.py "$@" 2>&1 | grep -v Warning; }
for k in 6 10 16; do echo "== K=$k"; run embed_cpu=1,num_samples=$k,chooser=route $A/tmp/chk_k$k.npz; done
