##############################################################################
# Copyright (C) 2025 Rebecca Pelke                                           #
# All Rights Reserved                                                        #
#                                                                            #
# This is work is licensed under the terms described in the LICENSE file     #
# found in the root directory of this source tree.                           #
##############################################################################
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ="Europe/Berlin"
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    git \
    cmake \
    gcc \
    gdb \
    python3-dev \
    python3-pip \
    python3-venv \
    curl

RUN python3 -m venv /apps/.venv

# Install simulator
COPY ../analog-cim-sim /apps/analog-cim-sim
COPY ../scripts /apps/scripts
RUN . /apps/.venv/bin/activate && pip3 install -r /apps/analog-cim-sim/requirements.txt
RUN . /apps/.venv/bin/activate && /apps/scripts/build_simulator.bash

# Install TVM runtime
COPY ../requirements.txt /apps/requirements.txt
RUN . /apps/.venv/bin/activate && pip3 install -r /apps/requirements.txt

WORKDIR /apps

# Create a home directory for python cache when starting docker as user
RUN mkdir /apps/home && chmod 777 /apps/home && chmod 777 /apps
ENV HOME=/apps/home

COPY ../scripts/entrypoint.bash /apps/entrypoint.bash
RUN chmod 777 /apps/entrypoint.bash
ENTRYPOINT ["/apps/entrypoint.bash"]
