#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH -J as_prefetch
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 03:00:00
#SBATCH -o /insomnia001/depts/edu/users/au2327/alpasim/logs/prefetch_pai_%j.out
# Download the first NSCENES curated-val NuRec scenes into the shared scene cache once (wizard
# with run_method=NONE), so parallel local runs don't race on the same downloads.
source /etc/profile
module load apptainer/1.4.5
A=/insomnia001/depts/edu/users/au2327/alpasim; SIF=$A/sandbox/alpasim-base
RUN=$A/runs/prefetch_pai_$SLURM_JOB_ID; mkdir -p $RUN
{
  echo "cd $RUN"
  echo "export HF_HOME=$A/.cache/hf"
  if [ -n "$SCENE_FILE" ]; then SEL="scenes.test_suite_id=null \"scenes.scene_ids=[$(sed "s/.*/'&'/" $SCENE_FILE | paste -sd, -)]\" scenes.limit_to_first_n=0"
  else SEL="scenes.limit_to_first_n=${NSCENES:-40}"; fi
  echo "exec /repo/.venv/bin/alpasim_wizard +e2e_challenge=dev +nurec_scenes=curated_val $SEL" \
       "wizard.run_method=NONE wizard.log_dir=$RUN defines.filesystem=$A/data/fs hydra.run.dir=$RUN/hydra"
} > $RUN/wizard.sh
time apptainer exec -B $A $SIF bash $RUN/wizard.sh > $RUN/wizard.log 2>&1; echo "wizard exit $?"
tail -5 $RUN/wizard.log
find $A/data/fs/nre-artifacts -name "*.usdz" | wc -l; du -sh $A/data/fs/nre-artifacts
