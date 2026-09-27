#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J as_nup
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o /insomnia001/depts/edu/users/au2327/alpasim/logs/fetch_nuplan_%j.out
# nuPlan/MTGS smoke-test subset (trajdata cache + configs + asset shard 1, ~36 GB).
# Streams each tarball straight into tar so no archive copy is kept on shared storage.
# SHARDS="001 002" overrides which asset shards are fetched.
source /etc/profile
set -eo pipefail
A=/insomnia001/depts/edu/users/au2327/alpasim
ROOT=$A/data/nuplan_track; mkdir -p $ROOT/.done
export HF_HOME=/insomnia001/depts/edu/users/au2327/nureasoning/.cache/hf PYTHONNOUSERSITE=1
PY=/insomnia001/depts/edu/users/au2327/nureasoning/env/bin/python
url() { $PY -c "from huggingface_hub import hf_hub_url; print(hf_hub_url('OpenDriveLab/AlpasimChallenge2026_nuplan_track','$1',repo_type='dataset'))"; }
get() {  # public dataset: no auth header needed
  local f=$1 tag=$(basename $1)
  [ -e $ROOT/.done/$tag ] && { echo "have $f"; return; }
  echo "$(date -u +%T) fetching $f"
  curl -sSL --retry 5 "$(url $f)" | tar -xz -C $ROOT && touch $ROOT/.done/$tag
  echo "$(date -u +%T) done $f"; du -sh $ROOT
}
get trajdata_cache/nuplan_test.tar.gz
get MTGS_asset/navtest/configs.tar.gz
for s in ${SHARDS:-001}; do get MTGS_asset/navtest/assets/part$s.tar.gz; done
ls $ROOT $ROOT/navtest | head
