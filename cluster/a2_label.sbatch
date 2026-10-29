#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH -J as_a2lab
#SBATCH --gres=gpu:1
#SBATCH --constraint=R6000          # 96 GB Blackwell: A2 Super (34B) in bf16 needs ~72 GiB
#SBATCH --exclude=ins058,ins085
#SBATCH -c 8
#SBATCH --mem=160G
#SBATCH -t 08:00:00
#SBATCH -o /insomnia001/depts/edu/users/au2327/alpasim/logs/a2_label_%j.out
# Offline Alpamayo 2 Super teacher labels for captured states (src/a15_ft/a2_label.py).
#   DATA=<dir of capture tars>  OUT=<label dir>  SHARD/NSHARDS  LIMIT  SMOKE=1 (also run NVIDIA's own smoke clip)
source /etc/profile
A=/insomnia001/depts/edu/users/au2327/alpasim
export UV_CACHE_DIR=$A/cache/uv UV_PYTHON_INSTALL_DIR=$A/cache/uvpy UV_LINK_MODE=copy UV_NO_CONFIG=1
export HF_HOME=$A/.cache/hf PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLCONFIGDIR=$TMPDIR/mpl ALPAMAYO2_SUPER_MODEL_ID=nvidia/Alpamayo2-Super
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
E=$TMPDIR/a2env; t=$(date +%s)
~/.local/bin/uv venv -q -p 3.12 $E || exit 2
~/.local/bin/uv pip install -q -p $E/bin/python torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128 || exit 2
# official prebuilt flash-attn wheel (torch 2.8, CUDA 12, cp312); building from source needs a matching nvcc
~/.local/bin/uv pip install -q -p $E/bin/python \
  "flash-attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" || exit 2
~/.local/bin/uv pip install -q -p $E/bin/python $A/external/alpamayo2 || exit 2
echo "a2env built in $(( $(date +%s) - t )) s: $($E/bin/python -c 'import torch, flash_attn, alpamayo2_super; print(torch.__version__, flash_attn.__version__)')"
cd $A/external/alpamayo2
if [ "${SMOKE:-0}" = 1 ]; then
  $E/bin/python -m alpamayo2_super.inference_smoke --model-id $ALPAMAYO2_SUPER_MODEL_ID \
    --manifest examples/validation_samples.json --sample-index 0 2>&1 | grep -v -i warning | tail -15
fi
[ -n "$DATA" ] || exit 0
# A2_LOCAL=1 (2026-10-28): the shared fileset is nearly full, so the 72 GB weights download to this node's local
# disk and vanish with the job (the HF token still comes from $HF_HOME/token, never argv)
if [ "${A2_LOCAL:-0}" = 1 ]; then
  export HF_HUB_CACHE=$TMPDIR/hfhub; mkdir -p $HF_HUB_CACHE
  free=$(df -BG --output=avail $TMPDIR | tail -1 | tr -dc 0-9); echo "node-local $TMPDIR: ${free} GB free"
  [ "$free" -ge 110 ] || { echo "not enough node-local disk for A2 Super"; exit 5; }
  t=$(date +%s)
  $E/bin/python -c "from huggingface_hub import snapshot_download as d; print(d('$ALPAMAYO2_SUPER_MODEL_ID', max_workers=8))" || exit 6
  echo "A2 Super downloaded in $(( $(date +%s) - t )) s"
fi
# one labeller per GPU of the job, each on its own shard of this job's shard
NG=$(nvidia-smi -L | wc -l); S=${SHARD:-0}; NS=${NSHARDS:-1}
for g in $(seq 0 $((NG - 1))); do
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$A/src $E/bin/python -u -m a15_ft.a2_label --data $DATA --out ${OUT:-$A/data/ft_a2labels} \
    --scratch $TMPDIR/a2lab_g$g --shard $((S * NG + g)) --nshards $((NS * NG)) --limit ${LIMIT:-0} $ARGS 2>&1 \
    | grep --line-buffered -v -i warning | sed -u "s/^/[gpu$g] /" &
done
wait
