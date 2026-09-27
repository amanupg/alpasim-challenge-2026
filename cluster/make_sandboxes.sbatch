#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J as_sbox
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o /insomnia001/depts/edu/users/au2327/alpasim/logs/make_sandboxes_%j.out
# This apptainer has no squashfuse, so every `exec <file>.sif` unpacks the whole image into
# /tmp (node RAM) first. Unpack each SIF once into a sandbox directory on shared storage
# and exec those instead.
source /etc/profile
module load apptainer/1.4.5
A=/insomnia001/depts/edu/users/au2327/alpasim
export APPTAINER_TMPDIR=$A/tmp; mkdir -p $A/sandbox $APPTAINER_TMPDIR
df -h /tmp /local /scratch 2>/dev/null
for s in alpasim-base nre-ga_26.04; do
  [ -d $A/sandbox/$s/usr ] && { echo "have $s"; continue; }
  rm -rf $A/sandbox/$s.tmp
  time apptainer build --sandbox $A/sandbox/$s.tmp $A/sif/$s.sif && mv $A/sandbox/$s.tmp $A/sandbox/$s
done
du -sh $A/sandbox/*
