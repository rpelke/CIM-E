#!/bin/bash
# Get directory of script
if [ "$#" -ne 1 ]; then
    echo "Error: Experiment name is required."
    exit 1
fi

EXP_NAME="$1"

source .venv/bin/activate
python3 src/main.py --config src/configs/$EXP_NAME.json
