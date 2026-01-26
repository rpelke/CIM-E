##############################################################################
# Copyright (C) 2026 Rebecca Pelke, Arunkumar Vaidyanathan                   #
# All Rights Reserved                                                        #
#                                                                            #
# This is work is licensed under the terms described in the LICENSE file     #
# found in the root directory of this source tree.                           #
##############################################################################
import json
import argparse
import os
import pandas as pd
import numpy as np
import math
from json.decoder import JSONDecodeError


def statistical_limits(layer_profile: dict):
    mean = layer_profile['mean']
    std_dev = math.sqrt(layer_profile['var'])
    return [mean - 3*std_dev, mean + 3*std_dev]


def bounds_limits(layer_profile: dict, bin_size: float = 10.0, max_tolerence: float = 0.0):
    values = {k: v for k, v in layer_profile['hist'] if v > 0.0}
    tolerence = layer_profile['samples'] * max_tolerence
    ss = 0
    for k, v in sorted(values.items()):
        ss += v
        if ss >= tolerence:
            min_val = k
            break
    ss = 0
    for k, v in reversed(sorted(values.items())):
        ss += v
        if ss >= tolerence:
            max_val = k
            break

    return [max(min_val - bin_size, 0.0), max_val]


def optimize_adc_limits(layer_profile: dict[str, float], m_mode: str, bin_size: float = 10.0):
    non_differential_modes = ['BNN_III', 'BNN_IV', 'BNN_V', 'TNN_IV', 'TNN_V']
    if m_mode in non_differential_modes:
        limits = bounds_limits(layer_profile, bin_size)
    else:
        limits = statistical_limits(layer_profile)
    return limits


def run_calibration(df: pd.DataFrame, store_path: str, profiles: dict[int, dict]):
    """Perform calibration for the given profiling run."""
    calibrated_limits: dict[str, dict] = {}
    for nn_name in list(df['nn_name'].unique()):
        print(f"Running calibration for NN: {nn_name}")
        df_nn = df[(df['nn_name'] == nn_name)]
        layers = profiles[int(df_nn.at[0, "config_idx"])].keys()
        bin_size = df_nn.at[0, "adc_profile"]
        calibrated_limits[nn_name] = {}
        print(f"Layers: {layers}.")
        for xs in list(df_nn['xbar_size'].unique()):
            calibrated_limits[nn_name][xs] = {}
            for mm in list(df_nn['m_mode'].unique()):
                calibrated_limits[nn_name][xs][mm] = {}
                c_idx = int(
                    df_nn[(df_nn["m_mode"] == mm) & (df_nn['xbar_size'] == xs)].loc[:, "config_idx"].iloc[0])
                for l_name in layers:
                    l_data = profiles[c_idx][l_name]
                    limits = optimize_adc_limits(l_data, mm, bin_size)
                    calibrated_limits[nn_name][xs][mm][l_name] = limits

    return calibrated_limits


def main(args):
    """ADC calibration utility that uses the results of an ADC
    profiling run to compute per-layer calibration ranges.

    The output calibration data can be patched onto another config.
    """
    exp_name = args.config.split('/')[-1].split('.json')[0]
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    exp_result_path = repo_path + '/results/' + exp_name
    df = pd.read_csv(f"{exp_result_path}/{exp_name}.csv")
    store_path = args.to_patch.split('.json')[0] + "_patched.json"
    profiles: dict[int, dict] = {
        c: json.load(open(f"{exp_result_path}/adc_prof_{c}.json", 'r'))
        for c in df.loc[:, "config_idx"].to_numpy(dtype=np.int32)}

    calibrated_limits = run_calibration(df, store_path, profiles)

    with open(args.to_patch, 'r') as json_in_file:
        try:
            data = json.load(json_in_file)
        except JSONDecodeError:
            data = {}
    data["adc_calib_dict"] = calibrated_limits
    with open(store_path, 'w') as json_out_file:
        json.dump(data, json_out_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        help='Path to experiment config',
                        required=True)

    parser.add_argument('--to_patch',
                        type=str,
                        help='Path to output config to be patched',
                        required=True)

    args = parser.parse_args()
    main(args)
