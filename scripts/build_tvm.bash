#!/bin/bash
##############################################################################
# Copyright (C) 2026 Rebecca Pelke                                           #
# All Rights Reserved                                                        #
#                                                                            #
# This is work is licensed under the terms described in the LICENSE file     #
# found in the root directory of this source tree.                           #
##############################################################################

wget https://archive.apache.org/dist/tvm/tvm-v0.14.0/apache-tvm-src-v0.14.0.tar.gz
tar -xzf apache-tvm-src-v0.14.0.tar.gz
rm apache-tvm-src-v0.14.0.tar.gz

cd apache-tvm-src-v0.14.0
mkdir build
cp cmake/config.cmake build/
cd build
cmake ..
make -j$(nproc)
