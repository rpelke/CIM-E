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

destination="${DIR}"

files=(
    "bnn_cifar100_BinaryDenseNet37_b1_mxn256x256_inp1x32x32x3.so"
    "bnn_cifar100_BinaryDenseNet37_b200_mxn256x256_inp200x32x32x3.so"
    "bnn_cifar100_BinaryNet_b1_mxn256x256_inp1x32x32x3.so"
    "bnn_cifar100_BinaryNet_b200_mxn256x256_inp200x32x32x3.so"
    "bnn_cifar100_BinaryDenseNet28_b1_mxn256x256_inp1x32x32x3.so"
    "bnn_cifar100_BinaryDenseNet28_b200_mxn256x256_inp200x32x32x3.so"
    "tnn_cifar10_VGG7_b1_mxn256x256_inp1x32x32x3.so"
    "tnn_cifar10_VGG7_b200_mxn256x256_inp200x32x32x3.so"
    "bnn_cifar10_VGG7_b1_mxn256x256_inp1x32x32x3.so"
    "bnn_cifar10_VGG7_b200_mxn256x256_inp200x32x32x3.so"
)

base_url="https://rwth-aachen.sciebo.de/s/4ZwRYj23blieuzv/download?path=%2F&files="

for file in "${files[@]}"; do
    if [ ! -f "${destination}/${file}" ]; then
        if curl --head --silent --fail "${base_url}${file}" > /dev/null; then
            echo "Downloading $file..."
            curl -o "${destination}/${file}" "${base_url}${file}"
        else
            echo "Remote file $file does not exist at ${base_url}${file}. Skipping."
        fi
    else
        echo "$file already exists. Skipping download."
    fi
done
