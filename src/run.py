from tvm.contrib import graph_executor
from itertools import product
from typing import Tuple
import tensorflow as tf
import numpy as np
import tempfile
import ctypes
import json
import glob
import sys
import tvm
import os

from experiment import ExpConfig
from model_parser import get_model_name

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))


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

    return num_classes, (train_images, train_labels), (test_images,
                                                       test_labels)


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


def run_experiments(exp: ExpConfig):
    cfgs = iterate_experiments(exp)
    for c in cfgs:
        n_classes, (train_images,
                    train_labels), (test_images, test_labels) = _get_dataset(c)

        # Execute dlopen to make symbols visible for the compiled executable
        python_version = '.'.join(sys.version.split('.', 2)[:2])
        lib_path = f"{repo_path}/.venv/lib/python{python_version}/site-packages"
        files = glob.glob(os.path.join(lib_path, 'acs_int.cpython*'))
        assert len(files) == 1
        acs_lib = ctypes.CDLL(f"{files[0]}", mode=ctypes.RTLD_GLOBAL)

        # Create config file for simulator
        acs_cfg_dir = f"{repo_path}/src/acs_configs"
        os.makedirs(acs_cfg_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=acs_cfg_dir,
                                         suffix=".json",
                                         delete=False) as tmp_file:
            tmp_name = tmp_file.name
            acs_config_data = _gen_acs_cfg_data(c, tmp_name)

        # Set acs config
        import acs_int
        acs_int.set_config(os.path.abspath(tmp_name))

        if c['nn_data'] == 'TRAIN':
            images = train_images
            labels = train_labels
        else:
            images = test_images
            labels = test_labels

        model_file_name = get_model_name(c)
        dev = tvm.device("llvm", 0)
        lib: tvm.runtime.Module = tvm.runtime.load_module(
            f"models/{model_file_name}")
        m = graph_executor.GraphModule(lib["default"](dev))

        top1_counter = 0
        top5_counter = 0

        for run in range(c['num_runs']):
            ifm = [
                images[int(c['batch'] * run):int(c['batch'] *
                                                 (run + 1)), :, :, :]
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
                1 if
                (labels[int(c['batch'] * run +
                            i)] in np.argsort(out[0][i, :])[::-1][:5]) else 0
                for i in range(c['batch'])
            ])

        top1_perc = (top1_counter / (c['num_runs'] * c['batch'])) * 100
        top5_perc = (top5_counter / (c['num_runs'] * c['batch'])) * 100

        print(f'Top 1% Accuracy: {top1_perc}')
        print(f'Top 5% Accuracy: {top5_perc}')

        del acs_int
        del acs_lib
