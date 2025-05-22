##############################################################################
# Copyright (C) 2025 Rebecca Pelke                                           #
# All Rights Reserved                                                        #
#                                                                            #
# This is work is licensed under the terms described in the LICENSE file     #
# found in the root directory of this source tree.                           #
##############################################################################
from tvm.contrib import graph_executor
from itertools import product
from typing import Tuple
from typing import List
import tensorflow as tf
import multiprocessing
import pandas as pd
import numpy as np
import tempfile
import ctypes
import time
import json
import glob
import sys
import tvm
import ast
import os

from experiment import ExpConfig
from model_parser import get_model_name

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
python_version = '.'.join(sys.version.split('.', 2)[:2])

ACS_CFG_DIR = f"{repo_path}/src/acs_configs"
ACS_LIB_PATH = f"{repo_path}/.venv/lib/python{python_version}/site-packages"
EMU_LIB_PATH = f"{repo_path}/build/release/lib/libacs_cb_emu.so"


def _check_pathes():
    if not os.path.exists(ACS_CFG_DIR):
        raise Exception(
            f"Cannot find ACS_CFG_DIR '{ACS_CFG_DIR}'. Please create the folder manually."
        )
    if not os.path.exists(ACS_LIB_PATH):
        raise Exception(f"Cannot find ACS_LIB_PATH '{ACS_LIB_PATH}'")
    if not os.path.exists(EMU_LIB_PATH):
        raise Exception(f"Cannot find EMU_LIB_PATH '{EMU_LIB_PATH}'")

    # Delete old configs
    json_files = glob.glob(os.path.join(ACS_CFG_DIR, "*.json"))
    for file in json_files:
        try:
            os.remove(file)
        except Exception as e:
            print(f"Error deleting file {file}: {e}")


def iterate_experiments(exp: ExpConfig):
    cfg = []
    static_fields = {
        key: value
        for key, value in {
            'nn_data_set': exp.nn_data_set,
            'nn_data': exp.nn_data,
            'batch': exp.batch,
            'num_runs': exp.num_runs,
            'digital_only': exp.digital_only,
            'adc_type': exp.adc_type,
            'verbose': exp.verbose,
            'read_disturb': exp.read_disturb,
            'read_disturb_mitigation': exp.read_disturb_mitigation
        }.items() if value is not None
    }
    iterable_fields = {
        key: value
        for key, value in {
            'nn_name': exp.nn_names,
            'xbar_size': exp.xbar_size,
            'hrs_lrs': exp.hrs_lrs,
            'alpha': exp.alpha,
            'resolution': exp.resolution,
            'm_mode': exp.m_mode,
            'hrs_noise': exp.hrs_noise,
            'lrs_noise': exp.lrs_noise,
            'V_read': exp.V_read,
            't_read': exp.t_read,
            'read_disturb_update_freq': exp.read_disturb_update_freq,
            'read_disturb_mitigation_fp': exp.read_disturb_mitigation_fp
        }.items() if value is not None
    }
    iterable_fields = {k: v for k, v in iterable_fields.items() if v != None}
    for combination in product(*iterable_fields.values()):
        config_entry = {**static_fields}
        for key, value in zip(iterable_fields.keys(), combination):
            config_entry[key] = value
        nn_idx = [
            idx for idx, i in enumerate(exp.nn_names)
            if i == config_entry['nn_name']
        ][0]
        config_entry['ifm'] = exp.ifm[nn_idx]
        cfg.append(config_entry)

    if len(cfg) == 0:
        raise Exception("Could not load config.")
    return cfg


def _get_all_datasets(
    cfgs: List[dict],
    use_same_inputs: bool = False
) -> dict[str, str, str, int, int, Tuple[int, Tuple[np.ndarray, np.ndarray],
                                         Tuple[np.ndarray, np.ndarray]]]:
    """Preload and preprocess all datasets needed for the experiments.
    Args:
        cfgs (List[dict]): All experiment configurations.
    Returns:
        dict[str, str, str, int, int,Tuple[int, Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray]]]: Unique keys and preprocessed datasets.
    """
    data_set_info = list(
        set((c['nn_data_set'], c['nn_data'], c['nn_name'], c['batch'],
             c['num_runs']) for c in cfgs))
    datasets = []
    for (ds, dst, nn, b, nr) in data_set_info:
        nr_si = 1 if use_same_inputs else nr
        datasets.append({
            'nn_data_set': ds,
            'nn_data': dst,
            'nn_name': nn,
            'batch': b,
            'num_runs': nr,
            'data': _get_dataset(nn, ds, b * nr_si)
        })
    return datasets


def _get_dataset(
    nn_name: str, nn_data_set: str, num_data: int
) -> Tuple[int, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Download and preprocess a dataset.
    Args:
        nn_name (str): Name of the NN
        nn_data_set (str): Data set name
        num_data (int): Number of samples to load
    Returns:
        Tuple[int, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
            Loaded and preprocessed dataset
    """
    if nn_name in ['BinaryDenseNet28', 'BinaryDenseNet37']:
        if nn_data_set == 'cifar100':
            num_classes = 100
            (train_images, train_labels), (
                test_images,
                test_labels) = tf.keras.datasets.cifar100.load_data()
            train_images, train_labels, test_images, test_labels = \
                train_images[:num_data], train_labels[:num_data], \
                test_images[:num_data], test_labels[:num_data]
        else:
            raise ValueError("Dataset not supported")

    elif nn_name in ['BinaryNet']:
        if nn_data_set == 'cifar100':
            num_classes = 100
            (train_images, train_labels), (
                test_images,
                test_labels) = tf.keras.datasets.cifar100.load_data()
            train_images, train_labels, test_images, test_labels = \
                train_images[:num_data], train_labels[:num_data], \
                test_images[:num_data], test_labels[:num_data]
            train_images = train_images.reshape(
                (num_data, 32, 32, 3)).astype("float32")
            test_images = test_images.reshape(
                (num_data, 32, 32, 3)).astype("float32")
            train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1
        else:
            raise ValueError("Dataset not supported")

    elif nn_name in ['VGG7']:
        if nn_data_set == 'cifar10':
            num_classes = 10
            (train_images, train_labels), (
                test_images,
                test_labels) = tf.keras.datasets.cifar10.load_data()
            train_images, train_labels, test_images, test_labels = \
                train_images[:num_data], train_labels[:num_data], \
                test_images[:num_data], test_labels[:num_data]
            train_images = train_images.reshape(
                (num_data, 32, 32, 3)).astype("float32")
            test_images = test_images.reshape(
                (num_data, 32, 32, 3)).astype("float32")
            train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1
        else:
            raise ValueError("Dataset not supported")

    else:
        raise ValueError("Dataset not supported")

    return num_classes, (train_images, train_labels), (test_images,
                                                       test_labels)


def _gen_acs_cfg_data(cfg: dict, tmp_name: str) -> dict:
    if cfg['adc_type'] == "INF_ADC":
        adc_type = "INF_ADC"
    else:
        if cfg['m_mode'] in [
                'BNN_I', 'BNN_II', 'BNN_VI', 'TNN_I', 'TNN_II', 'TNN_III'
        ]:
            adc_type = "SYM_RANGE_ADC"
        elif cfg['m_mode'] in [
                'BNN_III', 'BNN_IV', 'BNN_V', 'TNN_IV', 'TNN_V'
        ]:
            adc_type = "POS_RANGE_ONLY_ADC"
        else:
            raise ValueError("m_mode not supported")

    # Required parameters (not optional)
    acs_data = {
        "M": cfg['xbar_size'][0],
        "N": cfg['xbar_size'][0],
        "digital_only": cfg['digital_only'],
        "HRS": cfg['hrs_lrs'][0],
        "LRS": cfg['hrs_lrs'][1],
        "adc_type": adc_type,
        "m_mode": cfg['m_mode'],
        "HRS_NOISE": cfg['hrs_noise'],
        "LRS_NOISE": cfg['lrs_noise'],
        "verbose": cfg['verbose']
    }

    # Optional parameters
    if cfg.get('alpha') != None:
        acs_data["alpha"] = cfg['alpha']
    if cfg.get('resolution') != None:
        acs_data["resolution"] = cfg['resolution']
    if cfg.get('read_disturb') != None:
        acs_data["read_disturb"] = cfg['read_disturb']
    if cfg.get('V_read') != None:
        acs_data["V_read"] = cfg['V_read']
    if cfg.get('t_read') != None:
        acs_data["t_read"] = cfg['t_read']
    if cfg.get('read_disturb_update_freq') != None:
        acs_data["read_disturb_update_freq"] = cfg['read_disturb_update_freq']
    if cfg.get('read_disturb_mitigation') != None:
        acs_data["read_disturb_mitigation"] = cfg['read_disturb_mitigation']
    if cfg.get('read_disturb_mitigation_fp') != None:
        acs_data["read_disturb_mitigation_fp"] = cfg[
            'read_disturb_mitigation_fp']

    if cfg['m_mode'] in ['TNN_IV', 'TNN_V']:
        acs_data["SPLIT"] = [1, 1]
        acs_data["W_BIT"] = 2

    with open(f"{tmp_name}", "w") as f:
        json.dump(acs_data, f, indent=4)

    return acs_data


def _check_prev_results(cfg: dict, result_path: str,
                        exp_name: str) -> Tuple[dict, pd.DataFrame]:
    """Excludes experiments that have already been carried out.
    Args:
        cfg (dict): Experiment configuration
        exp_name (str): Name of the experiment
    Returns:
        dict: Reduced configuration with only 'new' simulations
    """
    if not os.path.exists(result_path):
        # Don't create this here because of docker --user permissions
        raise Exception(f"Please create folder '{result_path}' first.")

    if not os.path.exists(f"{result_path}/{exp_name}.csv"):
        os.makedirs(result_path, exist_ok=True)
        assert len(cfg) > 0, "Empty configuration file."
        accuracy_results = ["top1", "top5", "top1_baseline", "top5_baseline"]
        df_columns = list(cfg[0].keys()) + accuracy_results
        df = pd.DataFrame(columns=df_columns)
        return cfg, df

    else:
        df = pd.read_csv(f"{result_path}/{exp_name}.csv")
        assert len(cfg) > 0, "Empty configuration file."
        cfg_cols = cfg[0].keys()
        df_to_check = df[cfg_cols]

        # Convert list strings to lists
        for k, v in cfg[0].items():
            if type(v) == list:
                df_to_check.loc[:, k] = df_to_check[k].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # Remove already executed experiments
        del_count = 0
        for entry in cfg[:]:
            mask = (df_to_check[list(
                entry.keys())] == pd.Series(entry)).all(axis=1)
            if mask.any():
                cfg.remove(entry)
                del_count += 1

        print(
            f"---{del_count} simulations removed (already executed previously)---"
        )
        return cfg, df


def _load_xbar_simulator_lib(c: dict):
    # Execute dlopen to make symbols visible for the compiled executable
    files = glob.glob(os.path.join(ACS_LIB_PATH, 'acs_int.cpython*'))
    assert len(files) == 1
    acs_lib = ctypes.CDLL(f"{files[0]}", mode=ctypes.RTLD_GLOBAL)

    import acs_int

    # Create config file for simulator
    if not os.path.exists(ACS_CFG_DIR):
        # Don't create this here because of docker --user permissions
        raise Exception(f"Please create folder '{ACS_CFG_DIR}' first.")

    # Set acs config
    fd, tmp_name = tempfile.mkstemp(dir=ACS_CFG_DIR, suffix=".json")
    _ = _gen_acs_cfg_data(c, tmp_name)
    acs_int.set_config(os.path.abspath(tmp_name))
    os.close(fd)

    return acs_int, acs_lib


def _load_emulator_lib():
    emu_lib = ctypes.CDLL(EMU_LIB_PATH, mode=ctypes.RTLD_GLOBAL)
    return emu_lib


def _run_single_experiment(
    c: dict,
    c_idx: int,
    num_c: int,
    data: Tuple[int, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray,
                                                          np.ndarray]],
    ideal_xbar: bool = False,
    use_same_inputs: bool = False
) -> Tuple[dict, float, float, List[float], List[float], List[int]]:
    """Run a single experiment. This function is called from multiple processes.
    Args:
        c (dict): Experiment config
        c_idx (int): ID of the experiment
        num_c (int): Total number of experiments
        data (Tuple[int, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]): IFM/OFM data
        ideal_xbar (bool, optional): Switch off any non-idealities (if true). Defaults to False.
        use_same_inputs (bool, optional): Use the same inputs for all runs. Defaults to False.
    Returns:
        Tuple[dict, float, float]: cfg c, top 1% accuracy, top 5% accuracy
    """
    if ideal_xbar:
        print(f"Start Baseline Accuracy Simulation ({c_idx + 1}/{num_c})")
        emu_lib = _load_emulator_lib()
    else:
        print(f"Start Accuracy Simulation ({c_idx + 1}/{num_c})")
        acs_int, acs_lib = _load_xbar_simulator_lib(c)

    n_classes, (train_images, train_labels), (test_images, test_labels) = data

    if c['nn_data'] == 'TRAIN':
        images = train_images
        labels = train_labels
    else:
        images = test_images
        labels = test_labels

    model_file_name = get_model_name(c)
    dev = tvm.device("llvm", 0)
    lib: tvm.runtime.Module = tvm.runtime.load_module(
        f"{repo_path}/models/{model_file_name}")
    m = graph_executor.GraphModule(lib["default"](dev))

    top1_counter = 0
    top5_counter = 0

    top1_batch = []
    top5_batch = []
    sim_time_batch_ns = []

    for run in range(c['num_runs']):
        if use_same_inputs:
            ifm = [images[0:int(c['batch']), :, :, :]]
        else:
            ifm = [
                images[int(c['batch'] * run):int(c['batch'] *
                                                 (run + 1)), :, :, :]
            ]
        m.set_input(0, tvm.nd.array(ifm[0].astype("float32")))

        start_time = time.perf_counter_ns()
        m.run()
        end_time = time.perf_counter_ns()

        sim_time_batch_ns.append(end_time - start_time)
        out = [m.get_output(i).numpy() for i in range(m.get_num_outputs())]

        if use_same_inputs:
            top1_run = sum([
                1 if (np.argmax(out[0][i, :]) == labels[i]) else 0
                for i in range(c['batch'])
            ])
            top5_run = sum([
                1 if (labels[i] in np.argsort(out[0][i, :])[::-1][:5]) else 0
                for i in range(c['batch'])
            ])

        else:
            top1_run = sum([
                1 if (np.argmax(out[0][i, :]) == labels[int(c['batch'] * run +
                                                            i)]) else 0
                for i in range(c['batch'])
            ])
            top5_run = sum([
                1 if
                (labels[int(c['batch'] * run +
                            i)] in np.argsort(out[0][i, :])[::-1][:5]) else 0
                for i in range(c['batch'])
            ])

        top1_counter = top1_counter + top1_run
        top5_counter = top5_counter + top5_run

        top1_batch.append(top1_run / c['batch'])
        top5_batch.append(top5_run / c['batch'])

    top1_perc = (top1_counter / (c['num_runs'] * c['batch'])) * 100
    top5_perc = (top5_counter / (c['num_runs'] * c['batch'])) * 100

    print(
        f"({c_idx + 1}/{num_c}) Accuracy - Top 5%: {top5_perc} - Top 1%: {top1_perc}"
    )

    if ideal_xbar:
        del emu_lib
    else:
        del acs_int
        del acs_lib

    return c, top1_perc, top5_perc, top1_batch, top5_batch, sim_time_batch_ns


def _run_single_experiment_wrapper(args):
    return _run_single_experiment(*args)


def _get_baseline_accuracy(cfgs: List[dict],
                           n_jobs: int,
                           datasets: dict,
                           dbg: bool = False,
                           use_same_inputs: bool = False) -> List[dict]:
    """Calculate the baseline accuracy for all NN types (e.g., BNN/TNN).
    This is done by running the experiment with the `ideal xbar`.
    Ideal xbar means that we are using the crossbar emulator backend
    that emulates the MVMs in integer arithmetic (no errors).
    Args:
        cfgs (List[dict]): Experiment configs
        dbg (bool, optional): Debug mode. Defaults to False.

    Returns:
        List[dict]: Experiment configs including baseline accuracies.
    """

    def _check_field(field: str) -> None:
        f = set([c[field] for c in cfgs])
        assert len(f) == 1, f"'{f}' should be the same for all configurations"

    _check_field('num_runs')
    _check_field('batch')
    _check_field('nn_data_set')
    _check_field('nn_data')

    nns = set([c['nn_name'] for c in cfgs])
    m_mode = set([c['m_mode'] for c in cfgs])
    keep_one_per_bnn_tnn = [
        next((mode for mode in m_mode if mode.startswith('BNN')), None),
        next((mode for mode in m_mode if mode.startswith('TNN')), None)
    ]
    keep_one_per_bnn_tnn = [i for i in keep_one_per_bnn_tnn if i != None]

    bl_cfgs = []
    for nn in nns:
        for m in keep_one_per_bnn_tnn:
            bl_cfgs.append([
                c for c in cfgs if c['nn_name'] == nn and c['m_mode'] == m
            ][0])

    if dbg:
        n_jobs = 1
    else:
        n_jobs = n_jobs

    inputs = [_get_matching_dataset(c, datasets) for c in bl_cfgs]
    args_list = [(bl_c, idx, len(bl_cfgs), inputs[idx], True, use_same_inputs)
                 for idx, bl_c in enumerate(bl_cfgs)]
    with multiprocessing.Pool(processes=n_jobs) as pool:
        bl_res = pool.map(_run_single_experiment_wrapper, args_list)

    baseline_accuracies = []
    for c, top1_baseline, top5_baseline, _, _, _ in bl_res:
        c.update({
            "top1_baseline": top1_baseline,
            "top5_baseline": top5_baseline
        })
        baseline_accuracies.append(c)
    return baseline_accuracies


def _get_matching_baseline(cfg: dict, baseline_accuracies: List[dict]) -> dict:
    """Find the correct baseline accuracy for a given config.
    Args:
        cfg (dict): Single experiment config
        baseline_accuracies (List[dict]): List of baseline accuracies
    Returns:
        dict: cfg including baseline accuracy
    """
    keys = ['nn_name', 'nn_data', 'nn_data_set', 'batch', 'num_runs']
    matching_dict = next(
        (e for e in baseline_accuracies
         if all(e[k] == cfg[k]
                for k in keys) and e['m_mode'][:3] == cfg['m_mode'][:3]))
    return matching_dict['top1_baseline'], matching_dict['top5_baseline']


def _get_matching_dataset(
    cfg: dict, datasets: dict
) -> Tuple[int, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Get the matching preloaded dataset for a given experiment config.
    Args:
        cfg (dict): The experiment config
        datasets (dict): List of all loaded datasets
    Returns:
        Tuple[int, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
            The matching dataset
    """
    keys = ['nn_data_set', 'nn_data', 'batch', 'nn_name', 'num_runs']
    matches = [ds for ds in datasets if all(ds[k] == cfg[k] for k in keys)]
    assert len(matches) == 1, f"Multiple datasets match."
    return matches[0]['data']


def run_experiments(exp: ExpConfig,
                    exp_name: str,
                    n_jobs: int,
                    dbg: bool = False,
                    use_same_inputs: bool = False) -> None:
    _check_pathes()
    cfgs = iterate_experiments(exp)

    result_path = f"{repo_path}/results/{exp_name}"
    cfgs, df = _check_prev_results(cfgs, result_path, exp_name)

    print(
        f"---Execute experiment '{exp_name}': {len(cfgs)} simulations pending---"
    )
    if len(cfgs) == 0:
        sys.exit(0)

    datasets = _get_all_datasets(cfgs, use_same_inputs)

    baseline_accuracies = _get_baseline_accuracy(cfgs, n_jobs, datasets, dbg,
                                                 use_same_inputs)

    if dbg:
        res = []
        for c_idx, c in enumerate(cfgs):
            res.append(
                _run_single_experiment(c, c_idx, len(cfgs),
                                       _get_matching_dataset(c, datasets),
                                       False, use_same_inputs))
    else:
        args_list = [(c, c_idx, len(cfgs), _get_matching_dataset(c, datasets),
                      False, use_same_inputs) for c_idx, c in enumerate(cfgs)]

        with multiprocessing.Pool(processes=n_jobs) as pool:
            res = pool.map(_run_single_experiment_wrapper, args_list)

    for cfg, top1, top5, top1_batch, top5_batch, sim_time_batch_ns in res:
        top1_baseline, top5_baseline = _get_matching_baseline(
            cfg, baseline_accuracies)
        metrics = {
            'top1': top1,
            'top5': top5,
            'top1_batch': top1_batch,
            'top5_batch': top5_batch,
            'top1_baseline': top1_baseline,
            'top5_baseline': top5_baseline,
            'sim_time_batch_ns': sim_time_batch_ns
        }
        cfg.update(metrics)
        df = pd.concat([df, pd.DataFrame([cfg])], ignore_index=True)

    df.to_csv(f"{result_path}/{exp_name}.csv", index=False)
