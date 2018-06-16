#!/bin/bash
##### INFO ######
# Script to read host information
#
# VKozlov @18-May-2018
#
################
echo "=> Info on the system:"
cat /etc/os-release
echo ""
top -bn3 | head -n 5
echo ""

### info on nvidia cards ###
if [ -x $(command -v nvidia-smi) ]; then
    nvidia-smi
else
    echo "!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!"
    echo "no nvidia-smi found on this machine"
    echo "Are you sure that it has GPU(s) and CUDA?"
    echo "!!!!!!!!!!!!!!!! WARNING !!!!!!!!!!!!!!!!"
fi
