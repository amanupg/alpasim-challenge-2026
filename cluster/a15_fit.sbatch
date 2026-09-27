#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J as_a15fit
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 05:00:00
#SBATCH -o /insomnia001/depts/edu/users/au2327/alpasim/logs/a15_fit_%j.out
# Alpamayo 1.5 vs the AlpaSim driver contract (16 GiB, latency), one config per process.
# Uses nureasoning's a15env read-only; quantization libs live in an alpasim-only overlay.
source /etc/profile
A=/insomnia001/depts/edu/users/au2327/alpasim
NR=/insomnia001/depts/edu/users/au2327/nureasoning
PY=$NR/ext/a15env/bin/python
OV=$A/envs/overlay_torchao
[ -d $OV/torchao ] || $NR/env/bin/python -m pip install -q --no-deps --target $OV torchao==0.13.0
export PYTHONPATH=$OV PYTHONNOUSERSITE=1 HF_HOME=$NR/.cache/hf HF_HUB_OFFLINE=1
$PY -c "import torch, torchao; print(torch.__version__, 'torchao', torchao.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd $A
run() { echo "=== $*"; $PY -u src/a15_fit.py "$@" || echo "FAILED: $*"; }
run --quant bf16   --mode cot   --clips 30
run --quant bf16   --mode nocot
run --quant int8wo --mode nocot
run --quant fp8wo  --mode nocot
run --quant fp8dq  --mode nocot
run --quant int4wo --mode nocot
run --quant int8wo --mode cot   --clips 30
run --quant fp8dq  --mode cot   --clips 30
run --quant int4wo --mode cot   --clips 30
run --quant fp8dq  --mode nocot --graph
