#!/bin/bash
#SBATCH -A edu
#SBATCH -p short
#SBATCH -J as_base
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o /insomnia001/depts/edu/users/au2327/alpasim/logs/build_base_%j.out
# CPU-only: build the simulator base SIF and pull the NVIDIA renderer (NRE) SIF.
source /etc/profile
module load apptainer/1.4.5
set -x
A=/insomnia001/depts/edu/users/au2327/alpasim
export APPTAINER_CACHEDIR=$A/cache/apptainer
export APPTAINER_TMPDIR=/tmp/au2327_apptainer_$SLURM_JOB_ID
mkdir -p $APPTAINER_TMPDIR $A/sif
trap 'rm -rf $APPTAINER_TMPDIR' EXIT
cd $A/external
apptainer --version
if [ ! -s $A/sif/alpasim-base.sif ]; then
  cp $A/cluster/alpasim_base.def . && \
  apptainer build --fakeroot $A/sif/alpasim-base.sif.tmp alpasim_base.def && mv $A/sif/alpasim-base.sif.tmp $A/sif/alpasim-base.sif
fi
ls -la $A/sif
ls -la $A/sif
du -sh $APPTAINER_CACHEDIR
