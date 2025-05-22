##############################################################################
# Copyright (C) 2025 Rebecca Pelke                                           #
# All Rights Reserved                                                        #
#                                                                            #
# This is work is licensed under the terms described in the LICENSE file     #
# found in the root directory of this source tree.                           #
##############################################################################
import re

from experiment import ExpConfig


def parse_model_string(file_name: str) -> dict:
    """Parse the model file name to extract the model data.
    Args:
        file_name (str): file name (without path and extension)
    Returns:
        dict: Parameter dict
    """
    pattern = r"(?P<type>^[a-z]+)_(?P<data>[a-z0-9]+)_(?P<model>[a-zA-Z0-9]+)_(?P<batch>b\d+)_mxn(?P<xbar>\d+x\d+)_inp(?P<ifm>\d+x\d+x\d+x\d+)"
    match = re.match(pattern, file_name)
    if match:
        raw_data = match.groupdict()

        ifm_shape_str = raw_data['ifm']
        ifm_shape_array = list(map(int, ifm_shape_str.split('x')))
        raw_data['ifm'] = ifm_shape_array

        xbar_dim_str = raw_data['xbar']
        xbar_dim_array = list(map(int, xbar_dim_str.split('x')))
        raw_data['xbar'] = xbar_dim_array

        raw_data['batch'] = int(raw_data['batch'][1:])
        return raw_data
    else:
        raise ValueError("Unknown format.")


def create_experiment(cfg: dict) -> ExpConfig:
    exp = ExpConfig(
        nn_names=cfg['nn_names'],
        ifm=cfg['ifm'],
        nn_data_set=cfg['nn_data_set'],
        nn_data=cfg['nn_data'],
        batch=cfg['batch'],
        num_runs=cfg['num_runs'],
        xbar_size=cfg['xbar_size'],
        digital_only=cfg['digital_only'],
        hrs_lrs=tuple(cfg['hrs_lrs']),
        adc_type=cfg['adc_type'],
        m_mode=cfg['m_mode'],
        hrs_noise=cfg['hrs_noise'],
        lrs_noise=cfg['lrs_noise'],
        verbose=cfg['verbose'],
        alpha=cfg.get('alpha'),
        resolution=cfg.get('resolution'),
        read_disturb=cfg.get('read_disturb'),
        V_read=cfg.get('V_read'),
        t_read=cfg.get('t_read'),
        read_disturb_update_freq=cfg.get('read_disturb_update_freq'),
        read_disturb_mitigation=cfg.get('read_disturb_mitigation'),
        read_disturb_mitigation_fp=cfg.get('read_disturb_mitigation_fp'))

    for key in cfg.keys():
        if not hasattr(exp, key):
            raise Exception(
                f"Config parameter {key} not supported in ExpConfig.")
    return exp


def get_model_name(cfg: str) -> str:
    if cfg['m_mode'].startswith('BNN'):
        mode = 'bnn'
    elif cfg['m_mode'].startswith('TNN'):
        mode = 'tnn'
    else:
        raise ValueError("Unknown mode.")
    model_name = f"{mode}_{cfg['nn_data_set']}_{cfg['nn_name']}_b{cfg['batch']}_mxn{cfg['xbar_size'][0]}x{cfg['xbar_size'][1]}_inp{cfg['batch']}x{cfg['ifm'][0]}x{cfg['ifm'][1]}x{cfg['ifm'][2]}.so"
    return model_name
