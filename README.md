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

1. Build the simulator:

    Please use gcc>=10 or an equivalent clang.
    Inside the .venv, execute:
    ```bash
    ./build_simulator.bash
    ```

1. Test the simulator:

    ```bash
    python3 -m unittest discover -s analog-cim-sim/int-bindings/test -p '*_test.py'
    ```
