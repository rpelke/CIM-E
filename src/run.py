##############################################################################
# Copyright (C) 2025 Rebecca Pelke                                           #
# All Rights Reserved                                                        #
#                                                                            #
# This is work is licensed under the terms described in the LICENSE file     #
# found in the root directory of this source tree.                           #
##############################################################################
from tvm.contrib import graph_executor
from itertools import product
from joblib import Parallel
from joblib import delayed
from typing import Tuple
from typing import List
import tensorflow as tf
import pandas as pd
import numpy as np
import tempfile
import ctypes
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
        raise Exception(f"Cannot find ACS_CFG_DIR '{ACS_CFG_DIR}'")
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
        'nn_data_set': exp.nn_data_set,
        'nn_data': exp.nn_data,
        'batch': exp.batch,
        'num_runs': exp.num_runs,
        'digital_only': exp.digital_only,
        'adc_type': exp.adc_type,
        'verbose': exp.verbose
    }
    iterable_fields = {
        'nn_name': exp.nn_names,
        'xbar_size': exp.xbar_sizes,
        'hrs_lrs': exp.hrs_lrs,
        'alpha': exp.alpha,
        'resolution': exp.resolution,
        'm_mode': exp.m_mode,
        'hrs_noise': exp.hrs_noise,
        'lrs_noise': exp.lrs_noise
    }
    ordered_fields = {"ifm": exp.ifm}
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


def _get_dataset(
    cfg: dict
) -> Tuple[int, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    if cfg['nn_name'] in ['BinaryDenseNet28', 'BinaryDenseNet37']:
        if cfg['nn_data_set'] == 'cifar100':
            num_classes = 100
            (train_images, train_labels), (
                test_images,
                test_labels) = tf.keras.datasets.cifar100.load_data()
        else:
            raise ValueError("Dataset not supported")

    elif cfg['nn_name'] in ['BinaryNet']:
        if cfg['nn_data_set'] == 'cifar100':
            num_classes = 100
            (train_images, train_labels), (
                test_images,
                test_labels) = tf.keras.datasets.cifar100.load_data()
            train_images = train_images.reshape(
                (50000, 32, 32, 3)).astype("float32")
            test_images = test_images.reshape(
                (10000, 32, 32, 3)).astype("float32")
            train_images, test_images = train_images / 127.5 - 1, test_images / 127.5 - 1
        else:
            raise ValueError("Dataset not supported")

    else:
        raise ValueError("Dataset not supported")

    num_data = cfg['num_runs'] * cfg['batch']
    return num_classes, (train_images[:num_data],
                         train_labels[:num_data]), (test_images[:num_data],
                                                    test_labels[:num_data])


def _gen_acs_cfg_data(cfg: dict, tmp_name: str) -> dict:
    if cfg['m_mode'] in [
            'BNN_I', 'BNN_II', 'BNN_VI', 'TNN_I', 'TNN_II', 'TNN_III'
    ]:
        adc_type = "SYM_RANGE_ADC"
    elif cfg['m_mode'] in ['BNN_III', 'BNN_IV', 'BNN_V', 'TNN_IV', 'TNN_V']:
        adc_type = "POS_RANGE_ONLY_ADC"
    else:
        raise ValueError("m_mode not supported")

    acs_data = {
        "M": cfg['xbar_size'][0],
        "N": cfg['xbar_size'][0],
        "digital_only": cfg['digital_only'],
        "HRS": cfg['hrs_lrs'][0],
        "LRS": cfg['hrs_lrs'][1],
        "adc_type": adc_type,
        "alpha": cfg['alpha'],
        "resolution": cfg['resolution'],
        "m_mode": cfg['m_mode'],
        "HRS_NOISE": cfg['hrs_noise'],
        "LRS_NOISE": cfg['lrs_noise'],
        "verbose": cfg['verbose']
    }

    # Set ADC to infinity
    if acs_data["resolution"] == -1:
        acs_data["adc_type"] = "INF_ADC"

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

    with tempfile.NamedTemporaryFile(dir=ACS_CFG_DIR,
                                     suffix=".json",
                                     delete=False) as tmp_file:
        tmp_name = tmp_file.name
        acs_config_data = _gen_acs_cfg_data(c, tmp_name)

        # Set acs config
        acs_int.set_config(os.path.abspath(tmp_name))
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
        ideal_xbar: bool = False) -> Tuple[dict, float, float]:
    """Run a single experiment. This function is called from multiple processes.
    Args:
        c (dict): Experiment config
        c_idx (int): ID of the experiment
        num_c (int): Total number of experiments
        data (Tuple[int, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]): IFM/OFM data
        ideal_xbar (bool, optional): Switch off any non-idealities (if true). Defaults to False.
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

    for run in range(c['num_runs']):
        ifm = [
            images[int(c['batch'] * run):int(c['batch'] * (run + 1)), :, :, :]
        ]
        m.set_input(0, tvm.nd.array(ifm[0].astype("float32")))
        m.run()
        out = [m.get_output(i).numpy() for i in range(m.get_num_outputs())]

        top1_counter = top1_counter + sum([
            1 if (np.argmax(out[0][i, :]) == labels[int(c['batch'] * run +
                                                        i)]) else 0
            for i in range(c['batch'])
        ])
        top5_counter = top5_counter + sum([
            1 if (labels[int(c['batch'] * run +
                             i)] in np.argsort(out[0][i, :])[::-1][:5]) else 0
            for i in range(c['batch'])
        ])

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

    return c, top1_perc, top5_perc


def _get_baseline_accuracy(cfgs: List[dict], dbg: bool = False) -> List[dict]:

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
        bl_res = []
        for bl_idx, bl_c in enumerate(bl_cfgs):
            bl_res.append(
                _run_single_experiment(bl_c,
                                       bl_idx,
                                       len(bl_c),
                                       _get_dataset(bl_c),
                                       ideal_xbar=True))
    else:
        inputs = [_get_dataset(c) for c in bl_cfgs]
        bl_res = Parallel(n_jobs=-2, backend='loky', timeout=None)(
            delayed(_run_single_experiment)(
                bl_c, idx, len(bl_cfgs), inputs[idx], ideal_xbar=True)
            for idx, bl_c in enumerate(bl_cfgs))

    baseline_accuracies = []
    for c, top1_baseline, top5_baseline in bl_res:
        c.update({
            "top1_baseline": top1_baseline,
            "top5_baseline": top5_baseline
        })
        baseline_accuracies.append(c)
    return baseline_accuracies


def _get_matching_baseline(cfg: dict, baseline_accuracies: List[dict]) -> dict:
    keys = ['nn_name', 'nn_data', 'nn_data_set', 'batch', 'num_runs']
    matching_dict = next(
        (e for e in baseline_accuracies
         if all(e[k] == cfg[k]
                for k in keys) and e['m_mode'][:3] == cfg['m_mode'][:3]))
    return matching_dict['top1_baseline'], matching_dict['top5_baseline']


def run_experiments(exp: ExpConfig, exp_name: str, dbg: bool = False):
    _check_pathes()
    cfgs = iterate_experiments(exp)

    result_path = f"{repo_path}/results/{exp_name}"
    cfgs, df = _check_prev_results(cfgs, result_path, exp_name)

    print(
        f"---Execute experiment '{exp_name}': {len(cfgs)} simulations pending---"
    )
    if len(cfgs) == 0:
        sys.exit(0)

    baseline_accuracies = _get_baseline_accuracy(cfgs)

    if dbg:
        res = []
        for c_idx, c in enumerate(cfgs):
            res.append(
                _run_single_experiment(c, c_idx, len(cfgs), _get_dataset(c)))
    else:
        res = Parallel(n_jobs=-2, backend='loky',
                       timeout=None)(delayed(_run_single_experiment)(
                           c, c_idx, len(cfgs), _get_dataset(c))
                                     for c_idx, c in enumerate(cfgs))

    for cfg, top1, top5 in res:
        top1_baseline, top5_baseline = _get_matching_baseline(
            cfg, baseline_accuracies)
        metrics = {
            'top1': top1,
            'top5': top5,
            'top1_baseline': top1_baseline,
            'top5_baseline': top5_baseline
        }
        cfg.update(metrics)
        df = pd.concat([df, pd.DataFrame([cfg])], ignore_index=True)

    df.to_csv(f"{result_path}/{exp_name}.csv", index=False)
