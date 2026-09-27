#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J as_smoke
#SBATCH --gres=gpu:l40s:1   # nuPlan preset pins TORCH_CUDA_ARCH_LIST=8.9;9.0 (L40S/H100)
#SBATCH --exclude=ins046,ins047
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o /insomnia001/depts/edu/users/au2327/alpasim/logs/sim_smoke_%j.out
# Closed-loop smoke test on Insomnia: starter-kit driver + one-scene simulator stack, all
# from Apptainer images on one GPU. TRACK=nuplan (default; open data) or pai (needs the
# gated NuRec scenes). Mirrors the upstream starter-kit smoke commands.
source /etc/profile
module load apptainer/1.4.5
set -x
A=/insomnia001/depts/edu/users/au2327/alpasim
SIF=$A/sandbox/alpasim-base; REPO=$A/external/alpasim   # unpacked sandboxes: no squashfuse here
TRACK=${TRACK:-nuplan}
RUN=$A/runs/smoke_${TRACK}_$SLURM_JOB_ID; mkdir -p $RUN
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=6789 ALPASIM_NUPLAN_ROOT=$A/data/nuplan_track

# 1) contestant side: the starter-kit driver (straight line at current speed)
apptainer exec --nv $SIF bash -c "cd /repo && /repo/.venv/bin/python e2e_challenge/starter_kit/driver.py" \
  > $RUN/driver.log 2>&1 &
DRV=$!
trap 'kill $DRV 2>/dev/null' EXIT

# 2) wizard writes docker-compose.yaml + service configs only
PRESET=$([ $TRACK = pai ] && echo "+e2e_challenge=dev" || echo "+e2e_challenge_nuplan=dev")
apptainer exec -B $A $SIF bash -c "cd $RUN && /repo/.venv/bin/alpasim_wizard $PRESET \
  wizard.run_method=NONE wizard.log_dir=$RUN defines.filesystem=$A/data/fs hydra.run.dir=$RUN/hydra" > $RUN/wizard.log 2>&1 \
  || { tail -40 $RUN/wizard.log; exit 1; }
apptainer exec $SIF /repo/.venv/bin/python -c "import yaml,json; json.dump(yaml.safe_load(open('$RUN/docker-compose.yaml')), open('$RUN/docker-compose.json','w'), indent=1)"

# 3) run the stack; runtime in the foreground
python3 $A/src/compose_apptainer.py $RUN/docker-compose.json --sif-dir $A/sandbox --repo-host $REPO --logdir $RUN/svc-logs
RC=$?
echo "runtime exit $RC"
ls $RUN; ls $RUN/aggregate 2>/dev/null
cat $RUN/aggregate/results-summary.json 2>/dev/null | head -60
tail -20 $RUN/driver.log
exit $RC
