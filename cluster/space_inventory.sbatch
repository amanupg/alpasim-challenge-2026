#!/bin/bash
#SBATCH -A edu
#SBATCH -p short,burst
#SBATCH -J as_space
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 01:00:00
#SBATCH -o /insomnia001/depts/edu/users/au2327/alpasim/logs/space_inventory_%j.out
# Read-only inventory of alpasim/ only (other agents check their own folders): sizes, largest files, caches.
A=/insomnia001/depts/edu/users/au2327/alpasim
df -h /insomnia001 | tail -1
echo "== alpasim total"; du -sh $A
echo "== depth 1"; du -h --max-depth=1 $A 2>/dev/null | sort -rh | head -25
echo "== depth 2 (>= 1G)"; du -h --max-depth=2 $A 2>/dev/null | awk '$1 ~ /G|T/' | sort -rh | head -40
echo "== depth 3 (>= 5G)"; du -h --max-depth=3 $A 2>/dev/null | awk '$1 ~ /T/ || ($1 ~ /G/ && $1+0 >= 5)' | sort -rh | head -40
echo "== files >= 1 GB (size, mtime, path)"
find $A -xdev -type f -size +1G -printf "%s\t%TY-%Tm-%Td\t%p\n" 2>/dev/null | sort -rn | head -80 | awk -F'\t' '{printf "%6.1fG  %s  %s\n", $1/2^30, $2, $3}'
echo "== HF model snapshots"; for d in $(find $A -xdev -maxdepth 6 -type d -name "models--*" 2>/dev/null); do echo "$(du -sh $d 2>/dev/null | cut -f1)  $d"; done
echo "== leftover .asl"; find $A -xdev -name "*.asl" -printf "%s %p\n" 2>/dev/null | awk '{s+=$1; n++} END {printf "%d files, %.1f GB\n", n, s/2^30}'
echo "== runs: count and size by age"; find $A/runs -mindepth 1 -maxdepth 1 -type d -printf "%TY-%Tm-%Td %p\n" | sort | awk '{print $1}' | uniq -c | tail -20
echo "== caches"; for c in $(find $A -xdev -maxdepth 4 -type d \( -name ".cache" -o -name "cache" -o -name "pip" -o -name "uv" -o -name "apptainer" \) 2>/dev/null | head -30); do echo "$(du -sh $c 2>/dev/null | cut -f1)  $c"; done | sort -rh
echo done
