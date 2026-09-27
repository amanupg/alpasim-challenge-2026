#!/usr/bin/env bash
# Stage build inputs that are not kept in git: the alpasim_grpc package (from the pinned
# upstream checkout in external/alpasim, branch e2e_challenge) and the model assets
# (public HF repos at pinned revisions, downloaded anonymously).
set -euo pipefail
cd "$(dirname "$0")/.."
UP=external/alpasim
[ -d $UP ] || GIT_LFS_SKIP_SMUDGE=1 git clone -q --branch e2e_challenge https://github.com/NVlabs/alpasim.git $UP
git -C $UP log -1 --format='upstream %h %cd'
rm -rf container/alpasim_grpc && cp -r $UP/src/grpc container/alpasim_grpc
uvx --quiet --from "huggingface_hub[cli]" python - <<'PY'
from huggingface_hub import snapshot_download
A = "container/assets"
snapshot_download("Qwen/Qwen3-VL-8B-Instruct", revision="0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
                  local_dir=A + "/qwen3vl8b", token=False, allow_patterns=["*.json", "*.txt", "*.jinja"])
snapshot_download("Qwen/Qwen3-VL-2B-Instruct", revision="89644892e4d85e24eaac8bacfd4f463576704203",
                  local_dir=A + "/qwen3vl2b", token=False, allow_patterns=["*.json", "*.txt", "*.jinja"])
snapshot_download("nvidia/Alpamayo-1.5-10B", revision="7aba8293c09993f2e125c6819df05d7fa3e873ea",
                  local_dir=A + "/alpamayo15", token=False, max_workers=8)
PY
rm -rf container/assets/*/.cache
du -sh container/assets/*
