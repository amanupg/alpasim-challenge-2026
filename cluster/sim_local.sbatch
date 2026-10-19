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
#   CUDA_COMPAT=1: PAI renderer via cuda-compat 12.8 (lets PAI run on L40S/H100)
#   KEEP_ASL=1: keep the raw rollout.asl logs (default: deleted once the summary exists)
#   LOCAL_ROLLOUTS=0: write rollouts to the shared run dir instead of node-local TMPDIR (default 1)
#   EXTRA_OVERRIDES="controller.gains.lat_position_weight=3.0 ..." extra wizard overrides (no commas:
#   sbatch --export splits on them); TAG=name appended to the run dir
#   A15_* variables are passed through to the a15 driver (A15_NUM_SAMPLES, A15_NAV, A15_REASONING, ...)
# The a15 driver runs natively from nureasoning's a15env (read-only) + alpasim overlays, i.e.
# the same code that goes into the container; its model files come from the HF caches.
source /etc/profile
module load apptainer/1.4.5
set -x
A=/insomnia001/depts/edu/users/au2327/alpasim; NR=/insomnia001/depts/edu/users/au2327/nureasoning
source $A/cluster/a15env_local.sh   # Alpamayo env on node-local disk (nureasoning/ext/a15env deleted 2026-10-19)
SIF=$A/sandbox/alpasim-base; REPO=$A/external/alpasim
DRIVER=${DRIVER:-starter}; TRACK=${TRACK:-nuplan}; NSCENES=${NSCENES:-10}; ROLLOUTS=${ROLLOUTS:-2}
RUN=$A/runs/local_${TRACK}_${DRIVER}${TAG:+_$TAG}_$SLURM_JOB_ID; mkdir -p $RUN
env | grep -E "^A15_|^EXTRA_OVERRIDES" | sort > $RUN/variant.env
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# Unique ports per job: several runs can share a node, and gRPC binds with SO_REUSEPORT, so fixed
# ports (6000.., 6789) silently split connections between co-located runs' services and drivers
# (found 2026-09-29: sprint round 1 on ins046 was contaminated this way).
PORT_BASE=$((20000 + (SLURM_JOB_ID % 4000) * 10))
export ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=$((PORT_BASE + 9)) ALPASIM_NUPLAN_ROOT=$A/data/nuplan_track
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":($PORT_BASE|$((PORT_BASE+1))|$((PORT_BASE+2))|$((PORT_BASE+3))|$((PORT_BASE+9)))$"; then
  echo "port block $PORT_BASE busy on $(hostname)"; exit 3
fi
echo "ports: services from $PORT_BASE, driver $ALPASIM_DRIVER_PORT" 

# ---- raw rollout logs on node-local scratch (Aman 2026-10-18; dept fileset at 23 GB free): rollout.asl is
# ~15 GB per 40 PAI scenes, so $RUN/rollouts points at the job's TMPDIR (/local/slurm-job-N, which
# compose_apptainer.py binds into every service at the same path) and everything except *.asl is copied
# back at exit (KEEP_ASL=1 copies those too). Falls back to the shared dir if TMPDIR is under /tmp (not
# bound) or has < 40 GB free.
LOCAL_ROLL=""
if [ "${LOCAL_ROLLOUTS:-1}" = 1 ] && [ -n "$TMPDIR" ] && [ -d "$TMPDIR" ] && [[ "$TMPDIR" != /tmp* ]] \
   && [ "$(df -BG --output=avail "$TMPDIR" | tail -1 | tr -dc 0-9)" -ge 40 ]; then
  LOCAL_ROLL=$TMPDIR/alpasim_rollouts; mkdir -p $LOCAL_ROLL; ln -sfn $LOCAL_ROLL $RUN/rollouts
  echo "rollouts on node-local $LOCAL_ROLL ($(df -h --output=avail $TMPDIR | tail -1) free)"
else
  echo "rollouts on the shared fileset (TMPDIR=${TMPDIR:-unset} not usable)"
fi
finish_rollouts() {        # idempotent: replace the symlink by a real dir holding everything but *.asl
  [ -n "$LOCAL_ROLL" ] && [ -L $RUN/rollouts ] || return 0
  rm $RUN/rollouts; mkdir -p $RUN/rollouts
  if [ "${KEEP_ASL:-0}" = 1 ]; then rsync -a $LOCAL_ROLL/ $RUN/rollouts/; else rsync -a --exclude '*.asl' $LOCAL_ROLL/ $RUN/rollouts/; fi
  echo "rollouts copied back from node-local ($(find $LOCAL_ROLL -name '*.asl' | wc -l) raw logs left on scratch)"
}
trap finish_rollouts EXIT        # replaced below by the driver trap, which also calls it

# ---- driver
if [ $DRIVER = a15 ]; then
  # CPU regression test of the driver logic (fake policy) before spending GPU time
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$A/envs/overlay_driver:$A/envs/overlay_torchao:$A/src:$SIF/repo/src/grpc \
    bash -c "$A15PY $A/tests/test_driver_plan_storage.py && $A15PY $A/tests/test_route_nav.py" > $RUN/driver_test.txt 2>&1 \
    || { echo "driver regression test failed"; tail -5 $RUN/driver_test.txt; exit 4; }
  HUB=$NR/.cache/hf/hub
  export A15_MODEL_DIR=$(ls -d $HUB/models--nvidia--Alpamayo-1.5-10B/snapshots/*/ | head -1)
  export A15_VLM_DIR=$(ls -d $HUB/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/*/ | head -1)
  export A15_PROCESSOR_DIR=$(ls -d $HUB/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/*/ | head -1)
  env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 \
      PYTHONPATH=$A/envs/overlay_driver:$A/envs/overlay_torchao:$A/src:$SIF/repo/src/grpc \
      $A15PY -u -m a15_driver.driver > $RUN/driver.log 2>&1 &
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
trap 'kill $DRV 2>/dev/null; finish_rollouts' EXIT
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
       "defines.filesystem=$A/data/fs hydra.run.dir=$RUN/hydra eval.video.render_video=${VIDEO:-false} wizard.baseport=$PORT_BASE" \
       "runtime.endpoints.driver.n_concurrent_rollouts=$ROLLOUTS $SEL $EXTRA_OVERRIDES"
} > $RUN/wizard.sh
# the wizard creates $RUN/rollouts/...; with node-local rollouts the symlink target must be bound too
apptainer exec -B $A ${LOCAL_ROLL:+-B $TMPDIR} $SIF bash $RUN/wizard.sh > $RUN/wizard.log 2>&1 || { tail -40 $RUN/wizard.log; exit 1; }
apptainer exec $SIF /repo/.venv/bin/python -c "import yaml,json; json.dump(yaml.safe_load(open('$RUN/docker-compose.yaml')), open('$RUN/docker-compose.json','w'), indent=1)"
# CUDA_COMPAT=1: run the PAI NRE renderer through NVIDIA's cuda-compat 12.8 so PAI also runs on L40S/H100 (driver 550)
COMPAT_ARG=""; [ "${CUDA_COMPAT:-0}" = 1 ] && COMPAT_ARG="--cuda-compat $A/envs/cuda-compat-12-8/usr/local/cuda-12.8/compat --cache-dir $A/cache/containers_$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -c 'A-Za-z0-9' _)"   # own kernel cache per GPU type (NRE caches compiled kernels)
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
python3 $A/src/compose_apptainer.py $RUN/docker-compose.json --sif-dir $A/sandbox --repo-host $REPO --logdir $RUN/svc-logs $COMPAT_ARG
RC=$?
echo "runtime exit $RC"
python3 $A/src/summarize_run.py $RUN
finish_rollouts
echo "driver (pid $DRV) peak GPU memory: $(grep "^$DRV," $RUN/gpu_mem.log | sort -t, -k2 -n | tail -1)"
grep "closed session\|inference failed" $RUN/driver.log | tail -20
# driver health: a session that drove on the straight fallback means plans were not used (see 2026-09-30 bug)
if [ $DRIVER = a15 ]; then
  NS=$(grep -c "closed session" $RUN/driver.log); NF=$(grep -o "fallback [0-9]*" $RUN/driver.log | awk '$2>1' | wc -l)
  echo "driver health: $NS sessions, $NF with >1 fallback step" | tee $RUN/driver_health.txt
  [ "$NF" -gt 0 ] && echo "WARNING: $NF sessions used the fallback plan; results do not measure the policy" | tee -a $RUN/driver_health.txt
  # a renderer crash ("illegal memory access") ends the run early with few sessions (2026-10-10: 4 of 40)
  ND=$(grep -o "closed session [^:]*: drive [0-9]*" $RUN/driver.log | awk '$NF>0' | wc -l)
  NW=$( [ -n "${SCENE_FILE:-}" ] && grep -c . "$SCENE_FILE" || echo 0)
  [ "$ND" -lt "$NW" ] && echo "INCOMPLETE: only $ND of $NW scenes drove; rerun this batch" | tee -a $RUN/driver_health.txt
fi
# raw rollout logs (rollout.asl, ~15 GB per 40 PAI scenes, mostly embedded camera images) are deleted
# once the aggregate summary exists, unless KEEP_ASL=1 (shared filesystem; Aman 2026-10-03).
# Scores, metrics.parquet, videos, configs and service logs stay. A run without a summary keeps them.
if [ "${KEEP_ASL:-0}" != 1 ] && [ -f $RUN/aggregate/results-summary.json ]; then
  N=$(find $RUN/rollouts -name "*.asl" | wc -l); find $RUN/rollouts -name "*.asl" -delete
  echo "deleted $N raw rollout logs (KEEP_ASL=1 keeps them); run dir now $(du -sh $RUN | cut -f1)"
fi
exit $RC
