# CIM-Explorer
CIM-Explorer optimizes BNN and TNN inference for RRAM crossbars.
It uses the mappings and the crossbar simulator from [analog-cim-sim](https://github.com/rpelke/analog-cim-sim).

## Build Instructions
To build the project, you need cmake and a *dev version* of python.
If you don't have python3-dev, the simulator won't compile.

1. Clone the repository including submodules:

    ```bash
    git clone --recursive https://github.com/rpelke/CIM-Explorer.git
    ```

1. Create and activate a venv:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip3 install -r analog-cim-sim/requirements.txt
    ```

1. Build the simulator in docker:

    ```bash
    docker build -f build_simulator -t cim-explorer .
    ```

1. Test the simulator in docker:

    ```bash
    docker run -it --rm cim-explorer
    source .venv/bin/activate
    python3 -m unittest discover -s analog-cim-sim/int-bindings/test -p '*_test.py'
    deactivate
    ```

1. Execute a model:

    ```bash
    docker run -it --rm \
        -v $(pwd)/src:/apps/src:Z \
        -v $(pwd)/models:/apps/models:Z \
        cim-explorer

    source .venv/bin/activate
    python3 src/main.py --config src/configs/adc.json
    ```
