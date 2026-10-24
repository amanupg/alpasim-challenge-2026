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
#   CUDA_COMPAT=1: PAI renderer via cuda-compat 12.8 (lets PAI run on L40S/H100); CUDA_COMPAT=auto: only if driver < 570
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

# ---- CAPTURE=1: training-data generation for the action-expert fine-tune (2026-10-21). The ego is forced onto the
# recording for the whole scene, the recording is sent to the driver, and the driver (no model) saves rendered
# frames + history + recorded future every A15_CAPTURE_EVERY_US. CAPTURE_POLICY=1 instead lets the model drive and
# labels its own states with the recording (DAgger; samples go to data/ft_dagger/). Scenes download to node-local disk, and the
# samples are tarred to data/ft_capture/ at exit. Never used for evaluation runs.
# OFFICIAL_MIMIC=1: match the official initial (board) runs, e2e_challenge/ec2.yaml + driver/alpamayo_configs.yaml:
# the renderer WITHOUT the Harmonizer (the developer base config, and so every other local run, has it on; the PAI
# final runs turn it on), and no drive() calls during the 1.7 s force-GT warm-up.
if [ "${OFFICIAL_MIMIC:-0}" = 1 ]; then
  EXTRA_OVERRIDES="$EXTRA_OVERRIDES services.renderer.command.7=--no-enable-harmonizer runtime.simulation_config.skip_driver_during_force_gt=true"
fi
FSDIR=$A/data/fs
# LOCAL_FS=1 (2026-10-24): scenes download to node-local disk and vanish with the job, as for capture. The shared
# fileset filled up (5 TB dept quota) when 87 fresh-scene usdz (~160 GB) were fetched into data/fs.
if [ "${CAPTURE:-0}" = 1 ] || [ "${LOCAL_FS:-0}" = 1 ]; then
  FSDIR=$TMPDIR/fs; mkdir -p $FSDIR/nre-artifacts/all-usdzs
  cp -r $A/data/fs/nre-artifacts/ego-hoods $A/data/fs/nre-artifacts/scenesets $FSDIR/nre-artifacts/
  echo "scenes on node-local $FSDIR ($(df -h $TMPDIR | tail -1 | awk '{print $4}') free)"
fi
if [ "${CAPTURE:-0}" = 1 ]; then
  export A15_CAPTURE_DIR=$TMPDIR/capture A15_CAPTURE_EVERY_US=${A15_CAPTURE_EVERY_US:-500000}
  if [ "${CAPTURE_POLICY:-0}" = 1 ]; then
    # DAgger: the model drives (all A15_* policy settings apply); the recording is still sent every step and each
    # sample is labelled with the recorded future from the ego's actual pose (skipped beyond A15_CAPTURE_MAX_DEV_M)
    export A15_CAPTURE_MAX_DEV_M=${A15_CAPTURE_MAX_DEV_M:-3}; CAPDIR=$A/data/ft_dagger
    EXTRA_OVERRIDES="$EXTRA_OVERRIDES runtime.simulation_config.send_recording_ground_truth=true"
  else
    export A15_NO_MODEL=1; CAPDIR=$A/data/ft_capture
    EXTRA_OVERRIDES="$EXTRA_OVERRIDES runtime.simulation_config.force_gt_duration_us=60000000 runtime.simulation_config.skip_driver_during_force_gt=false runtime.simulation_config.send_recording_ground_truth=true"
  fi
  finish_capture() {
    [ -d $TMPDIR/capture ] || return 0
    mkdir -p $CAPDIR
    tar -C $TMPDIR -cf $CAPDIR/${TAG:-capture}_$SLURM_JOB_ID.tar capture
    echo "captured $(find $TMPDIR/capture -name '*.npz' | wc -l) samples from $(ls $TMPDIR/capture | wc -l) sessions -> $CAPDIR/${TAG:-capture}_$SLURM_JOB_ID.tar ($(du -sh $CAPDIR/${TAG:-capture}_$SLURM_JOB_ID.tar | cut -f1))"
  }
else
  finish_capture() { :; }
fi

# ---- driver
if [ $DRIVER = a15 ]; then
  # CPU regression test of the driver logic (fake policy) before spending GPU time
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$A/envs/overlay_driver:$A/envs/overlay_torchao:$A/src:$SIF/repo/src/grpc \
    bash -c "$A15PY $A/tests/test_driver_plan_storage.py && $A15PY $A/tests/test_route_nav.py" > $RUN/driver_test.txt 2>&1 \
    || { echo "driver regression test failed"; tail -5 $RUN/driver_test.txt; exit 4; }
  HUB=$NR/.cache/hf/hub
  export A15_MODEL_DIR=${A15_MODEL_DIR:-$(ls -d $HUB/models--nvidia--Alpamayo-1.5-10B/snapshots/*/ | head -1)}
  export A15_VLM_DIR=$(ls -d $HUB/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/*/ | head -1)
  export A15_PROCESSOR_DIR=$(ls -d $HUB/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/*/ | head -1)
  # alternative driver build (A15_ENV_SCRIPT + A15_DRIVER_SRC; e.g. the alpasim-jepa FlashDrive port): its own
  # Python env (exports FDPY) and driver source; overlay_torchao is built for torch 2.8, so it is left out
  DRV_PY=$A15PY; DRV_PYPATH=$A/envs/overlay_driver:$A/envs/overlay_torchao:$A/src:$SIF/repo/src/grpc
  if [ -n "${A15_DRIVER_SRC:-}" ] && [ -z "${A15_ENV_SCRIPT:-}" ]; then
    # candidate driver code in its own tree with the standard env, so jobs confirming the image keep $A/src
    DRV_PYPATH=$A/envs/overlay_driver:$A/envs/overlay_torchao:$A15_DRIVER_SRC:$SIF/repo/src/grpc
    echo "driver: standard env, source $A15_DRIVER_SRC"
  fi
  if [ -n "${A15_ENV_SCRIPT:-}" ]; then
    source $A15_ENV_SCRIPT || { echo "A15_ENV_SCRIPT failed"; exit 4; }
    DRV_PY=${FDPY:?A15_ENV_SCRIPT must export FDPY}
    DRV_PYPATH=$A/envs/overlay_driver:${A15_DRIVER_SRC:?A15_DRIVER_SRC required with A15_ENV_SCRIPT}:$SIF/repo/src/grpc
    export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$TMPDIR/inductor}
    echo "driver: $DRV_PY from $A15_DRIVER_SRC"
  fi
  env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 \
      PYTHONPATH=$DRV_PYPATH $DRV_PY -u -m a15_driver.driver > $RUN/driver.log 2>&1 &
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
trap 'kill $DRV 2>/dev/null; finish_rollouts; finish_capture' EXIT
# Slurm sends TERM at the time limit (and USR1 5 min before, via --signal=B:USR1@300 where set): save what exists
trap 'echo "signal: saving partial results"; kill $DRV 2>/dev/null; finish_rollouts; finish_capture; exit 143' TERM USR1
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
       "defines.filesystem=$FSDIR hydra.run.dir=$RUN/hydra eval.video.render_video=${VIDEO:-false} wizard.baseport=$PORT_BASE" \
       "runtime.endpoints.driver.n_concurrent_rollouts=$ROLLOUTS $SEL $EXTRA_OVERRIDES"
} > $RUN/wizard.sh
# the wizard creates $RUN/rollouts/...; with node-local rollouts the symlink target must be bound too
# up to 3 tries: scene downloads from HF occasionally fail mid-transfer (2026-10-21: xet "error decoding response
# body"); files already downloaded are kept, and the last try falls back from xet to plain HTTP
for TRY in 1 2 3; do
  [ $TRY = 3 ] && sed -i '1a export HF_HUB_DISABLE_XET=1' $RUN/wizard.sh
  apptainer exec -B $A ${LOCAL_ROLL:+-B $TMPDIR} $SIF bash $RUN/wizard.sh > $RUN/wizard.log 2>&1 && break
  echo "wizard try $TRY failed: $(grep -m1 -E 'Error|error' $RUN/wizard.log | cut -c1-200)"
  [ $TRY = 3 ] && { tail -40 $RUN/wizard.log; exit 1; }
  sleep 30
done
apptainer exec $SIF /repo/.venv/bin/python -c "import yaml,json; json.dump(yaml.safe_load(open('$RUN/docker-compose.yaml')), open('$RUN/docker-compose.json','w'), indent=1)"
# CUDA_COMPAT=1: run the PAI NRE renderer through NVIDIA's cuda-compat 12.8 so PAI also runs on L40S/H100 (driver 550)
# CUDA_COMPAT=auto: use it only where the host driver is older than 570 (L40S/L40/A6000/H100 here; Blackwell has 575)
if [ "${CUDA_COMPAT:-0}" = auto ]; then
  DRVMAJ=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  CUDA_COMPAT=$([ "${DRVMAJ:-0}" -lt 570 ] && echo 1 || echo 0); echo "CUDA_COMPAT=auto -> $CUDA_COMPAT (driver $DRVMAJ)"
fi
COMPAT_ARG=""; [ "${CUDA_COMPAT:-0}" = 1 ] && COMPAT_ARG="--cuda-compat $A/envs/cuda-compat-12-8/usr/local/cuda-12.8/compat --cache-dir $A/cache/containers_$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -c 'A-Za-z0-9' _)"   # own kernel cache per GPU type (NRE caches compiled kernels)
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
python3 $A/src/compose_apptainer.py $RUN/docker-compose.json --sif-dir $A/sandbox --repo-host $REPO --logdir $RUN/svc-logs $COMPAT_ARG
RC=$?
echo "runtime exit $RC"
python3 $A/src/summarize_run.py $RUN
finish_rollouts
finish_capture
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
