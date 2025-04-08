# CIM-Explorer
[![Style](https://github.com/rpelke/CIM-Explorer/actions/workflows/style.yml/badge.svg)](https://github.com/rpelke/CIM-Explorer/actions/workflows/style.yml)

CIM-Explorer optimizes BNN and TNN inference for RRAM crossbars.
It uses the mappings and the crossbar simulator from [analog-cim-sim](https://github.com/rpelke/analog-cim-sim).

## Build Instructions
To build the project, you need cmake and a *dev version* of python.
If you don't have python3-dev, the simulator won't compile.

The following steps were tested with Python 3.10.12. 

1. Clone the repository including submodules:

    ```bash
    git clone --recursive https://github.com/rpelke/CIM-Explorer.git
    ```

1. Pull the pre-compiled NNs:

    ```bash
    git lfs pull
    ```

1. Build the simulator in docker (recommended):

    ```bash
    docker build -f build_simulator.dockerfile -t cim-explorer .
    ```
    This project is designed to run rootless, so you can also use `podman`.
    Make sure `podman-docker` is installed or create an alias for docker that points to podman.

1. Test the simulator `analog-cim-sim` manually in docker:

    ```bash
    docker run -it --rm --entrypoint "/bin/bash" cim-explorer
    source .venv/bin/activate
    python3 -m unittest discover -s analog-cim-sim/int-bindings/test -p '*_test.py'
    deactivate
    ```

1. Execute the benchmarks with docker:

    The name of the test is the input argument.
    ```bash
    ./scripts/benchmark.bash test
    ./scripts/benchmark.bash adc
    ./scripts/benchmark.bash lrs_var
    ./scripts/benchmark.bash hrs_var
    ./scripts/benchmark.bash adc_vgg7
    ```

1. Native build of the simulator (for development only):

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip3 install -r analog-cim-sim/requirements.txt
    ./scripts/build_simulator.bash
    ```
