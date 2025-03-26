#!/bin/bash
##############################################################################
# Copyright (C) 2025 Rebecca Pelke                                           #
# All Rights Reserved                                                        #
#                                                                            #
# This is work is licensed under the terms described in the LICENSE file     #
# found in the root directory of this source tree.                           #
##############################################################################

# Get directory of script
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do 
    DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="${DIR}/$SOURCE"
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
PROJ_DIR=${DIR}/..

if [ "$#" -ne 1 ]; then
    echo "Error: Experiment name is required."
    exit 1
else 
    EXP_NAME="$1"
    mkdir -p $PROJ_DIR/results/$EXP_NAME
    mkdir -p $PROJ_DIR/src/acs_configs
fi

DOCKER_FLAGS=""

if [[ "$(docker --version)" == *"podman"* ]]; then
    echo "Using podman"
    DOCKER_FLAGS="--userns keep-id"
else
    echo "Using docker"
    DOCKER_FLAGS="--user $(id -u):$(id -g)"
fi

docker run -it --rm $DOCKER_FLAGS --memory=4g \
    -v ${PROJ_DIR}/src:/apps/src:Z \
    -v ${PROJ_DIR}/models:/apps/models:Z \
    -v ${PROJ_DIR}/results:/apps/results:Z \
    cim-explorer $EXP_NAME
