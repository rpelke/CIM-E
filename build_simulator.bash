#!/bin/bash
# Get directory of script
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do 
    DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"
    SOURCE="$(readlink "$SOURCE")"
    [[ $SOURCE != /* ]] && SOURCE="${DIR}/$SOURCE"
done
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

rm -rf ${DIR}/build
cd ${DIR}/analog-cim-sim
mkdir -p ${DIR}/build/release/build && cd ${DIR}/build/release/build

python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1-2)

cmake \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DPY_INSTALL_PATH=${DIR}/.venv/lib/python${python_version}/site-packages \
    -DCMAKE_PREFIX_PATH=${DIR}/.venv/lib/python${python_version}/site-packages/pybind11/share/cmake/pybind11 \
    -DCMAKE_INSTALL_PREFIX=../ \
    -DLIB_TESTS=ON \
    -DBUILD_LIB_ACS_INT=ON \
    ../../../analog-cim-sim/cpp 

make -j `nproc`
make install
cd $DIR/analog-cim-sim

# Execute tests
python3 -m unittest discover -s int-bindings/test -p '*_test.py'
cd $DIR
