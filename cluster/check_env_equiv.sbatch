#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH -J as_envchk
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 01:00:00
#SBATCH -o /insomnia001/depts/edu/users/au2327/alpasim/logs/check_env_equiv_%j.out
# Rebuilt Alpamayo env (per-job, node-local; nureasoning/ext/a15env deleted 2026-10-19) vs the old env:
# same validation clips through the policy, max trajectory difference against references saved with the old env
# on L40S (tmp/chk_t10.npz: K=1, 2026-10-03; tmp/chk_k6.npz: K=6 route chooser, 2026-10-09). Then the driver tests.
source /etc/profile
A=/insomnia001/depts/edu/users/au2327/alpasim; NR=/insomnia001/depts/edu/users/au2327/nureasoning
source $A/cluster/a15env_local.sh
SIF=$A/sandbox/alpasim-base
run() { cd $A && env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=$A/envs/overlay_driver:$A/envs/overlay_torchao:$A/src \
  $A15PY -u src/check_policy.py "$@" 2>&1 | grep -v Warning; }
run embed_cpu=1 $A/tmp/chk_env2_t10.npz $A/tmp/chk_t10.npz
run embed_cpu=1,num_samples=6,chooser=route $A/tmp/chk_env2_k6.npz $A/tmp/chk_k6.npz
cd $A && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$A/envs/overlay_driver:$A/envs/overlay_torchao:$A/src:$SIF/repo/src/grpc \
  bash -c "$A15PY tests/test_driver_plan_storage.py | tail -2 && $A15PY tests/test_route_nav.py | tail -1"
