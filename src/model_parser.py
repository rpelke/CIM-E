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
        hrs_lrs=cfg.get('hrs_lrs'),
        gmin_gmax=cfg.get('gmin_gmax'),
        adc_type=cfg['adc_type'],
        m_mode=cfg['m_mode'],
        hrs_noise=cfg['hrs_noise'],
        lrs_noise=cfg['lrs_noise'],
        verbose=cfg['verbose'],
        resolution=cfg.get('resolution'),
        adc_profile=cfg.get('adc_profile'),
        adc_calib_mode=cfg.get('adc_calib_mode'),
        adc_calib_dict=cfg.get('adc_calib_dict'),
        read_disturb=cfg.get('read_disturb'),
        V_read=cfg.get('V_read'),
        t_read=cfg.get('t_read'),
        read_disturb_update_freq=cfg.get('read_disturb_update_freq'),
        read_disturb_mitigation_strategy=cfg.get(
            'read_disturb_mitigation_strategy'),
        read_disturb_mitigation_fp=cfg.get('read_disturb_mitigation_fp'),
        read_disturb_update_tolerance=cfg.get('read_disturb_update_tolerance'),
        parasitics=cfg.get('parasitics'),
        w_res=cfg.get('w_res'),
        c2c_var=cfg.get('c2c_var')
    )

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

    m_xbar = cfg['xbar_size'][0]
    n_xbar = cfg['xbar_size'][1]

    # Special mappings
    # if cfg['m_mode'] == "BNN_V":
    #     m_xbar = cfg['xbar_size'][0] * 2

    if cfg['m_mode'] in ["BNN_V", "BNN_VI", "TNN_I"]:
        n_xbar = cfg['xbar_size'][1] // 2

    model_name = f"{mode}_{cfg['nn_data_set']}_{cfg['nn_name']}_b{cfg['batch']}_mxn{m_xbar}x{n_xbar}_inp{cfg['batch']}x{cfg['ifm'][0]}x{cfg['ifm'][1]}x{cfg['ifm'][2]}.so"
    return model_name
