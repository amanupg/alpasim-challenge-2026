#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J as_local
#SBATCH --gres=gpu:l40s:1   # nuPlan preset pins TORCH_CUDA_ARCH_LIST=8.9;9.0 (L40S/H100)
#SBATCH -c 12
#SBATCH --mem=120G
#SBATCH -t 06:00:00
#SBATCH -o /insomnia001/depts/edu/users/au2327/alpasim/logs/sim_local_%j.out
# Local closed-loop evaluation on Insomnia (Apptainer simulator stack + our driver).
#   DRIVER=starter|a15   TRACK=nuplan|pai   NSCENES=N   ROLLOUTS=2 (concurrent, as officially)
#   SCENES_FROM=<run dir>: reuse that run's scene list (same scenes for a fair comparison)
#   EXTRA_OVERRIDES="controller.gains.lat_position_weight=3.0 ..." extra wizard overrides (no commas:
#   sbatch --export splits on them); TAG=name appended to the run dir
#   A15_* variables are passed through to the a15 driver (A15_NUM_SAMPLES, A15_NAV, A15_REASONING, ...)
# The a15 driver runs natively from nureasoning's a15env (read-only) + alpasim overlays, i.e.
# the same code that goes into the container; its model files come from the HF caches.
source /etc/profile
module load apptainer/1.4.5
set -x
A=/insomnia001/depts/edu/users/au2327/alpasim; NR=/insomnia001/depts/edu/users/au2327/nureasoning
SIF=$A/sandbox/alpasim-base; REPO=$A/external/alpasim
DRIVER=${DRIVER:-starter}; TRACK=${TRACK:-nuplan}; NSCENES=${NSCENES:-10}; ROLLOUTS=${ROLLOUTS:-2}
RUN=$A/runs/local_${TRACK}_${DRIVER}${TAG:+_$TAG}_$SLURM_JOB_ID; mkdir -p $RUN
env | grep -E "^A15_|^EXTRA_OVERRIDES" | sort > $RUN/variant.env
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=6789 ALPASIM_NUPLAN_ROOT=$A/data/nuplan_track

# ---- driver
if [ $DRIVER = a15 ]; then
  HUB=$NR/.cache/hf/hub
  export A15_MODEL_DIR=$(ls -d $HUB/models--nvidia--Alpamayo-1.5-10B/snapshots/*/ | head -1)
  export A15_VLM_DIR=$(ls -d $HUB/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/*/ | head -1)
  export A15_PROCESSOR_DIR=$(ls -d $HUB/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/*/ | head -1)
  env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 \
      PYTHONPATH=$A/envs/overlay_driver:$A/envs/overlay_torchao:$A/src:$SIF/repo/src/grpc \
      $NR/ext/a15env/bin/python -u -m a15_driver.driver > $RUN/driver.log 2>&1 &
  DRV=$!
  for i in $(seq 1 120); do grep -q "policy ready\|policy load failed" $RUN/driver.log && break; sleep 10; done
  grep "loaded in\|policy ready\|failed" $RUN/driver.log
  grep -q "policy ready" $RUN/driver.log || { tail -40 $RUN/driver.log; kill $DRV; exit 1; }
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv
else
  apptainer exec --nv $SIF bash -c "cd /repo && /repo/.venv/bin/python e2e_challenge/starter_kit/driver.py" \
    > $RUN/driver.log 2>&1 &
  DRV=$!
fi
trap 'kill $DRV 2>/dev/null' EXIT
# log ALL GPU processes, but the driver's own line is what the 16 GiB cap is about (renderer/physics share the GPU)
( while kill -0 $DRV 2>/dev/null; do nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> $RUN/gpu_mem.log; sleep 15; done ) &
echo $DRV > $RUN/driver.pid

# ---- scenes
if [ -n "$SCENES_FROM" ]; then
  SCENES=$(cat $SCENES_FROM/scene_ids.txt)
elif [ $TRACK = nuplan ]; then
  SCENES=$(ls $A/data/nuplan_track/navtest/assets | head -$NSCENES | sed "s/.*/'&'/" | paste -sd, -)
fi
echo "$SCENES" > $RUN/scene_ids.txt
# PAI: first N scenes of the curated validation suite (scenes download from HF on first use)
if [ $TRACK = pai ] && [ -n "$SCENE_FILE" ]; then      # explicit PAI scene list (one clipgt id per line)
  PRESET="+e2e_challenge=dev"
  SEL="+nurec_scenes=curated_val scenes.test_suite_id=null \"scenes.scene_ids=[$(sed "s/.*/'&'/" $SCENE_FILE | paste -sd, -)]\" scenes.limit_to_first_n=0"
elif [ $TRACK = pai ]; then
  PRESET="+e2e_challenge=dev"; SEL="+nurec_scenes=curated_val scenes.limit_to_first_n=$NSCENES"
else
  PRESET="+e2e_challenge_nuplan=dev"; SEL="\"scenes.scene_ids=[$SCENES]\" scenes.limit_to_first_n=0"
fi

# ---- simulator: wizard writes compose + configs (from a generated script, to keep quoting sane);
#      our launcher then runs the stack with apptainer
{
  echo "cd $RUN"
  echo "export HF_HOME=$A/.cache/hf"
  echo "exec /repo/.venv/bin/alpasim_wizard $PRESET wizard.run_method=NONE wizard.log_dir=$RUN" \
       "defines.filesystem=$A/data/fs hydra.run.dir=$RUN/hydra eval.video.render_video=false" \
       "runtime.endpoints.driver.n_concurrent_rollouts=$ROLLOUTS $SEL $EXTRA_OVERRIDES"
} > $RUN/wizard.sh
apptainer exec -B $A $SIF bash $RUN/wizard.sh > $RUN/wizard.log 2>&1 || { tail -40 $RUN/wizard.log; exit 1; }
apptainer exec $SIF /repo/.venv/bin/python -c "import yaml,json; json.dump(yaml.safe_load(open('$RUN/docker-compose.yaml')), open('$RUN/docker-compose.json','w'), indent=1)"
python3 $A/src/compose_apptainer.py $RUN/docker-compose.json --sif-dir $A/sandbox --repo-host $REPO --logdir $RUN/svc-logs
RC=$?
echo "runtime exit $RC"
python3 $A/src/summarize_run.py $RUN
echo "driver (pid $DRV) peak GPU memory: $(grep "^$DRV," $RUN/gpu_mem.log | sort -t, -k2 -n | tail -1)"
grep "closed session\|inference failed" $RUN/driver.log | tail -20
exit $RC
