#!/bin/bash
##### INFO ######
# This script supposes to:
# 1. download a Docker image (e.g. Tensorflow)
# 2. create corresponding udocker container
#
# VKozlov @18-May-2018
#
# udocker: https://github.com/indigo-dc/udocker
#
################

### MAIN CONFIG ###

UDOCKER_DIR="$PROJECT/.udocker"  # udocker main directory.

# Default settings. Will be overwritten from the command line (preferred way!)
DEFAULTIMG="tensorflow/tensorflow:1.6.0-gpu"
UCONTAINER="tf160gpu"
###################

USAGEMESSAGE="Either run as \n
              $ $0 DOCKERIMG UCONTAINER (example: $0 tensorflow/tensorflow:1.7.0-gpu tf170gpu) \n
              or just \n
              $ $0 (default is set in $0, e.g. DEFAULTIMG=tensorflow/tensorflow:1.6.0-gpu and UCONTAINER=tf160gpu)"

if [ $# -eq 0 ]; then
    DOCKERIMG=$DEFAULTIMG
elif [ $# -eq 2 ]; then
    DOCKERIMG=$1
    UCONTAINER=$2
else
    echo "#############################################################"
    echo  "ERROR! Wrong execution. "
    shopt -s xpg_echo
    echo " "$USAGEMESSAGE
    echo "#############################################################"
    exit 1
fi

# Deduce DOCKERTAG from DOCKERIMG
DOCKERTAG=${DOCKERIMG#*:}

##########################

IFExist=$(udocker ps |grep "'$UCONTAINER'")
if [ ${#IFExist} -le 1 ]; then
    echo "=> Trying to pull the Docker Image: $DOCKERIMG"
    udocker pull $DOCKERIMG
    echo "=> Creating Container ${UCONTAINER}"
    if (udocker create --name=${UCONTAINER} ${DOCKERIMG}); then
       echo "########################################"
       echo "  contrainer $UCONTAINER created       "
       echo "  note the name and use it for jobs    "
       echo "########################################"    
    else
       echo "###########################################"
       echo "  Something went WRONG !!!  "
       echo "  Are you sure that:         "
       echo "  - docker image name is correct?"
       echo "  - udocker is installed??       "
       echo "###########################################"
       exit 1
    fi
else
    echo "###########################################"
    echo "  contrainer $UCONTAINER already exists!   "
    echo "  note the name and use it for jobs        "
    echo "  or delete it with $ udocker rm $UCONTAINER "
    echo "###########################################"
fi
