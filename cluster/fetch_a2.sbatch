#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J as_fetchA2
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH -o /insomnia001/depts/edu/users/au2327/alpasim/logs/fetch_a2_%j.out
# Download Alpamayo 2-Super (Aman approved 2026-10-23, ~70 GB) into alpasim's HF cache, and clone its code repo.
# Offline teacher only (34B does not fit the 16 GiB driver cap). Token comes from $HF_HOME/token, never argv.
# CHECK_ONLY=1: print gating and file sizes, download nothing.
source /etc/profile
A=/insomnia001/depts/edu/users/au2327/alpasim
source $A/cluster/a15env_local.sh
export HF_HOME=$A/.cache/hf PYTHONNOUSERSITE=1 HF_HUB_ENABLE_HF_TRANSFER=0
df -h $A | tail -1
$A15PY - <<'EOF'
import os
from huggingface_hub import HfApi, snapshot_download
api = HfApi()
repo = "nvidia/Alpamayo2-Super"
info = api.model_info(repo, files_metadata=True)
tot = sum((s.size or 0) for s in info.siblings)
print("gated:", info.gated, "| files:", len(info.siblings), "| total GB:", round(tot / 1e9, 1), "| sha:", info.sha)
for s in sorted(info.siblings, key=lambda s: -(s.size or 0))[:12]:
    print(f"  {s.rfilename}  {(s.size or 0) / 1e9:.2f} GB")
if os.environ.get("CHECK_ONLY") == "1":
    raise SystemExit(0)
p = snapshot_download(repo, max_workers=8)
print("downloaded to", p)
EOF
cd $A/external && { [ -d alpamayo2 ] || git clone --depth 1 https://github.com/NVlabs/alpamayo2.git; } && git -C alpamayo2 log -1 --format="alpamayo2 %h %cd"
df -h $A | tail -1
