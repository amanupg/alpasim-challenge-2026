#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J as_avg
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o /insomnia001/depts/edu/users/au2327/alpasim/logs/ft_average_%j.out
# Weight-space averages of expert checkpoints (CPU only). AVG_JOBS: ";"-separated "out|base_weight|ckpt:w ckpt:w".
source /etc/profile
A=/insomnia001/depts/edu/users/au2327/alpasim; NR=/insomnia001/depts/edu/users/au2327/nureasoning
source $A/cluster/a15env_local.sh
export HUB=$NR/.cache/hf/hub HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1
cd $A
IFS=';' read -ra JOBS <<< "$AVG_JOBS"
for j in "${JOBS[@]}"; do
  IFS='|' read -r out bw cks <<< "$j"
  PYTHONPATH=$A/src $A15PY -m a15_ft.average --out $A/data/ft_ckpt/$out --base_weight $bw $(for c in $cks; do echo $A/data/ft_ckpt/$c; done) || exit 1
done
ls -la $A/data/ft_ckpt/
