#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J as_test
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -o /insomnia001/depts/edu/users/au2327/alpasim/logs/test_driver_%j.out
# CPU-only driver regression tests (fake policy, no model): run before any GPU evaluation run.
source /etc/profile
A=/insomnia001/depts/edu/users/au2327/alpasim; NR=/insomnia001/depts/edu/users/au2327/nureasoning
source $A/cluster/a15env_local.sh   # Alpamayo env on node-local disk (nureasoning/ext/a15env deleted 2026-10-19)
SIF=$A/sandbox/alpasim-base
cd $A && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$A/envs/overlay_driver:$A/envs/overlay_torchao:$A/src:$SIF/repo/src/grpc \
  $A15PY -u tests/test_driver_plan_storage.py && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$A/envs/overlay_driver:$A/envs/overlay_torchao:$A/src:$SIF/repo/src/grpc $A15PY -u tests/test_route_nav.py
