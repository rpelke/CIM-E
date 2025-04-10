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

rm -rf ${PROJ_DIR}/build
cd ${PROJ_DIR}/analog-cim-sim
mkdir -p ${PROJ_DIR}/build/release/build && cd ${PROJ_DIR}/build/release/build

python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1-2)
LIB_INSTALL_PATH=${PROJ_DIR}/.venv/lib/python${python_version}/site-packages/pybind11/share/cmake/pybind11

cmake \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_BUILD_TYPE=Release \
    -DPY_INSTALL_PATH=${PROJ_DIR}/.venv/lib/python${python_version}/site-packages \
    -DCMAKE_PREFIX_PATH=${LIB_INSTALL_PATH} \
    -DCMAKE_INSTALL_PREFIX=../ \
    -DLIB_TESTS=ON \
    -DBUILD_LIB_ACS_INT=ON \
    -DBUILD_LIB_CB_EMU=ON \
    ${PROJ_DIR}/analog-cim-sim/cpp

make -j `nproc`
make install
cd ${PROJ_DIR}/analog-cim-sim

# Execute tests
python3 -m unittest discover -s int-bindings/test -p '*_test.py'
cd ${PROJ_DIR}
