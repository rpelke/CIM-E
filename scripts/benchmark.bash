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

# Default parameters
N_JOBS=4
EXP_NAME=""

for ARG in "$@"; do
    case $ARG in
        exp_name=*)
            EXP_NAME="${ARG#*=}"
            shift
            ;;
        n_jobs=*)
            N_JOBS="${ARG#*=}"
            shift
            ;;
        use_same_inputs=*)
            VALUE="${ARG#*=}"
            if [ "$VALUE" = "True" ]; then
                USE_SAME_INPUTS="--use_same_inputs"
            else
                USE_SAME_INPUTS=""
            fi
            shift
            ;;
        *)
            echo "Unknown argument: $ARG"
            exit 1
            ;;
    esac
done

# Check parameters
if [ -z "$EXP_NAME" ]; then
    echo "Error: exp_name is required. Usage: exp_name=<name>"
    exit 1
else 
    mkdir -p $PROJ_DIR/results/$EXP_NAME
    mkdir -p $PROJ_DIR/src/acs_configs
fi

DOCKER_FLAGS=""

if [[ "$(docker --version)" == *"podman"* ]]; then
    echo "Using podman"
    if podman run --help | grep -q -- "--userns=keep-id"; then
        DOCKER_FLAGS="--userns=keep-id"
    else
        DOCKER_FLAGS="--userns=host"
    fi
else
    echo "Using docker"
    DOCKER_FLAGS="--user $(id -u):$(id -g)"
fi

docker run -it --rm --memory=4g $DOCKER_FLAGS \
    -v ${PROJ_DIR}/src:/apps/src:Z \
    -v ${PROJ_DIR}/models:/apps/models:Z \
    -v ${PROJ_DIR}/results:/apps/results:Z \
    cim-e $EXP_NAME $N_JOBS $USE_SAME_INPUTS
