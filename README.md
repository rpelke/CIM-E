# <u>CIM</u>-<u>E</u>xplorer
[![Style](https://github.com/rpelke/CIM-E/actions/workflows/style.yml/badge.svg)](https://github.com/rpelke/CIM-E/actions/workflows/style.yml)

CIM-E optimizes BNN and TNN inference for RRAM crossbars.
It uses the mappings and the crossbar simulator from [analog-cim-sim](https://github.com/rpelke/analog-cim-sim).

## Build Instructions
To build the project, you need cmake and a *dev version* of python.
If you don't have python3-dev, the simulator won't compile.

The following steps were tested with Python 3.10.12. 

1. Clone the repository including submodules:

    ```bash
    git clone --recursive https://github.com/rpelke/CIM-E.git
    ```

1. Download the pre-compiled BNNs/TNNs:

    ```bash
    ./models/download_models.bash
    ```

1. Build the simulator in docker (recommended):

    ```bash
    docker build -f build_simulator.dockerfile -t cim-e .
    ```
    This project is designed to run rootless, so you can also use `podman`.
    Make sure `podman-docker` is installed or create an alias for docker that points to podman.

## Run Simulations

1. Execute the benchmarks with docker:

    The name of the test is the input argument.
    ```bash
    ./scripts/benchmark.bash test
    ./scripts/benchmark.bash adc
    ./scripts/benchmark.bash lrs_var
    ./scripts/benchmark.bash hrs_var
    ./scripts/benchmark.bash adc_vgg7
    ```

## Development Only
1. Native build of the simulator (without docker):

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip3 install -r analog-cim-sim/requirements.txt
    pip3 install -r requirements.txt
    ./scripts/build_simulator.bash
    ```

1. Run simulations:

    Select the desired configuration file in the [configs](src/configs) folder.
    ```bash
    python3 src/main.py --config src/configs/test.json
    ```
    Make sure that you create a folder for the results manually beforehand.

## Troubleshooting
- Test the simulator `analog-cim-sim` manually in docker:

    ```bash
    docker run -it --rm --entrypoint "/bin/bash" cim-e
    source .venv/bin/activate
    python3 -m unittest discover -s analog-cim-sim/int-bindings/test -p '*_test.py'
    deactivate && exit
    ```

- Debug Python code in VSCode:

    Install a Python debugger extension in VSCode and add the following configuration in your `launch.json`:
    ```json
    {
        "name": "Run simulation (Python)",
        "type": "debugpy",
        "request": "launch",
        "program": "${workspaceFolder}/src/main.py",
        "console": "integratedTerminal",
        "justMyCode": false,
        "args": [
            "--debug", // Optional: Single-core execution
            "--config",
            "${workspaceFolder}/src/configs/test.json"
        ]
    }
    ```
    The `--debug` flag ensures that all simulations are executed in one process and not in several parallel processes.