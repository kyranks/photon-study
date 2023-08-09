#!/bin/bash

export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh --quiet

lsetup "views LCG_103 x86_64-centos7-gcc11-opt"

python /eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/TRAININGTEST/sizereduction.py full_v02 conv odd
