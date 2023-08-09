#!/bin/bash

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh --quiet

lsetup "views LCG_103 x86_64-centos7-gcc11-opt"

python /eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/makepickledata-gc_time.py None None full_v02
