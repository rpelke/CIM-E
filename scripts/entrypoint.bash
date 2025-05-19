#!/bin/bash
##############################################################################
# Copyright (C) 2025 Rebecca Pelke                                           #
# All Rights Reserved                                                        #
#                                                                            #
# This is work is licensed under the terms described in the LICENSE file     #
# found in the root directory of this source tree.                           #
##############################################################################

if [ "$#" -lt 2 ]; then
    echo "Error: Experiment name and number of parallel jobs are required."
    exit 1
fi

EXP_NAME="$1"
N_JOBS="$2"

if [ "$#" -ge 3 ]; then
    USE_SAME_INPUTS="$3"
else
    USE_SAME_INPUTS=""
fi

source .venv/bin/activate
python3 src/main.py \
    --config src/configs/$EXP_NAME.json \
    --n_jobs $N_JOBS \
    $USE_SAME_INPUTS
