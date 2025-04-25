# <u>CIM</u>-<u>E</u>xplorer
[![Style](https://github.com/rpelke/CIM-E/actions/workflows/style.yml/badge.svg)](https://github.com/rpelke/CIM-E/actions/workflows/style.yml)
[![Build Check](https://github.com/rpelke/CIM-E/actions/workflows/build.yml/badge.svg)](https://github.com/rpelke/CIM-E/actions/workflows/build.yml)

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
    ./scripts/benchmark.bash exp_name=test n_jobs=4
    ./scripts/benchmark.bash exp_name=adc n_jobs=4
    ./scripts/benchmark.bash exp_name=lrs_var n_jobs=4
    ./scripts/benchmark.bash exp_name=hrs_var n_jobs=4
    ./scripts/benchmark.bash exp_name=adc_vgg7 n_jobs=4
    ```

1. Plot the results:

    ```bash
    python3 src/plot.py --config <path_to_config>
    ```

    For example:
    ```bash
    python3 src/plot.py --config src/configs/lrs_var.json
    ```


## Create new experiment

1. Create a new file `<exp_name>.json` in the [configs](src/configs) folder.

2. Specify all parameters to be used in the design space exploration. Every combination of parameters will be simulated. A tick in the `DSE list`column means that a list with several possible parameters can be created here. Otherwise, only a `Single Choice` is possible.

    | Parameter      | Explanation                            | DSE<br>List | Single<br>Choice |
    |----------------|----------------------------------------|:------:|:-----:|
    | `nn_names`     | Names of the neural network architectures       |✅ |    |
    | `ifm`          | Dimension of the input feature map              | ✅ |    | 
    | `nn_data_set`  | Data set                                        |    | ✅ |
    | `nn_data`      | Choise of dataset parts (test or train)         |    | ✅ |
    | `batch`        | Batch size of input feature map                 |    | ✅ |
    | `num_runs`     | Total number of images: `batch`*`num_runs`      |    | ✅ |
    | `xbar_size`    | Dimensions of the offloaded matrix (MxN)        | ✅ |    |
    | `digital_only` | True if mappings should use digital values only |    | ✅ |
    | `hrs_lrs`      | HRS/LRS current (in uA) if `V_read is applied`  | ✅ |    | 
    | `adc_type`     | Currently only supports `FP_ADC_ALPHA`          |    | ✅ |
    | `alpha`        | Limitation of the ADC range in [0,1], 1 is 100% | ✅ |    | 
    | `resolution`   | ADC resolution in bit                           | ✅ |    |
    | `m_mode`       | Mapping mode (see next table)                   | ✅ |    | 
    | `hrs_noise`    | Std. dev. of the gaussian noise (uA) around hrs | ✅ |    | 
    | `lrs_noise`    | Std. dev. of the gaussian noise (uA) around lrs | ✅ |    | 
    | `verbose`      | Enable verbose output of the simulator          | ✅ |    | 
    
    A brief overview of the mapping modes is given in the tables below:

    | BNN Mapping     | Weights                      | Inputs                           |
    |-----------------|------------------------------|----------------------------------|
    | BNN (I)         | $w_{NN} = g_D^+ - g_D^-$     | $i_{NN} = 2 \cdot v_D - 1$       |
    | BNN (II)        | $w_{NN} = g_D^+ - g_D^-$     | $i_{NN} = -2 \cdot v_D + 1$      |
    | BNN (III)       | $w_{NN} = 2 \cdot g_D - 1$   | $i_{NN} = v_D^+ - v_D^-$         |
    | BNN (IV)        | $w_{NN} = -2 \cdot g_D + 1$  | $i_{NN} = v_D^+ - v_D^-$         |
    | BNN (V)         | XOR mapping ($v_D^+g_D^+ + v_D^-g_D^-$)                         |
    | BNN (VI)        | $w_{NN} = g_D^+ - g_D^-$     | $i_{NN} = v_D^+ - v_D^-$         |

    | TNN Mapping | Weights                          | Inputs                           |
    |-------------|----------------------------------|----------------------------------|
    | TNN (I)     | $w_{NN} = g_D^+ - g_D^-$         | $i_{NN} = v_D^+ - v_D^-$         |
    | TNN (II)    | $w_{NN} = g_D^+ - g_D^-$         | $i_{NN} = (v_D^1, v_D^0)$        |
    | TNN (III)   | $w_{NN} = g_D^+ - g_D^-$         | $i_{NN} + 1 = (v_D^1, v_D^0)$    |
    | TNN (IV)    | $w_{NN} = (g_D^1, g_D^0)$        | $i_{NN} = v_D^+ - v_D^-$         |
    | TNN (V)     | $w_{NN} + 1 = (g_D^1, g_D^0)$    | $i_{NN} = v_D^+ - v_D^-$         |


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
    python3 src/main.py --config src/configs/test.json --n_jobs 4
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
            "${workspaceFolder}/src/configs/test.json",
            "--n_jobs", "4"
        ]
    }
    ```
    The `--debug` flag ensures that all simulations are executed in one process and not in several parallel processes.