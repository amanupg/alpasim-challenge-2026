# Source from a job script: builds the Alpamayo driver env on node-local disk and exports A15PY.
# nureasoning/ext/a15env (which every a15 job used) was deleted on 2026-10-19; this rebuilds the same 64 pinned
# packages (envs/a15env-pins.txt, from its 2026-09-26 install log) under $TMPDIR, with uv's cache and Python there
# too, so nothing is written to the shared fileset or to nureasoning/. ~3 GB download, a few minutes per job.
A=${A:-/insomnia001/depts/edu/users/au2327/alpasim}
NR=${NR:-/insomnia001/depts/edu/users/au2327/nureasoning}
_LOC=${TMPDIR:-/tmp/$USER-$SLURM_JOB_ID}
mkdir -p $_LOC
export UV_CACHE_DIR=$_LOC/uvcache UV_PYTHON_INSTALL_DIR=$_LOC/uvpy UV_NO_CONFIG=1
A15ENV=$_LOC/a15env
if [ ! -x $A15ENV/bin/python ]; then
  t0=$(date +%s)
  ~/.local/bin/uv venv -q -p 3.12.14 $A15ENV \
    && ~/.local/bin/uv pip install -q -p $A15ENV/bin/python --no-deps -r <(grep -v '^#' $A/envs/a15env-pins.txt) \
    && echo "$NR/ext/alpamayo1.5/src" > $($A15ENV/bin/python -c 'import site; print(site.getsitepackages()[0])')/alpamayo1_5_src.pth \
    || { echo "a15env build failed"; return 5 2>/dev/null || exit 5; }
  echo "a15env built on node-local $A15ENV in $(( $(date +%s) - t0 )) s: $($A15ENV/bin/python -c 'import torch, transformers, alpamayo1_5; print("torch", torch.__version__, "transformers", transformers.__version__, "alpamayo1_5 ok")')"
fi
export A15PY=$A15ENV/bin/python
