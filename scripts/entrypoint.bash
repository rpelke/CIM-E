#!/bin/bash
##############################################################################
# Copyright (C) 2025 Rebecca Pelke                                           #
# All Rights Reserved                                                        #
#                                                                            #
# This is work is licensed under the terms described in the LICENSE file     #
# found in the root directory of this source tree.                           #
##############################################################################

# Get directory of script
if [ "$#" -ne 1 ]; then
    echo "Error: Experiment name is required."
    exit 1
fi

EXP_NAME="$1"

source .venv/bin/activate
python3 src/main.py --config src/configs/$EXP_NAME.json
