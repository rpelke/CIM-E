##############################################################################
# Copyright (C) 2025 Rebecca Pelke                                           #
# All Rights Reserved                                                        #
#                                                                            #
# This is work is licensed under the terms described in the LICENSE file     #
# found in the root directory of this source tree.                           #
##############################################################################
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ExpConfig:
    nn_names: List[str]
    ifm: List[List[int]]
    nn_data_set: str
    nn_data: str
    batch: int
    num_runs: int
    xbar_sizes: List[Tuple[int, int]]
    digital_only: bool
    hrs_lrs: List[Tuple[float]]
    adc_type: str
    alpha: List[float]
    resolution: List[int]
    m_mode: List[str]
    hrs_noise: List[float]
    lrs_noise: List[float]
    verbose: bool

    def __check_paramters(self):
        if self.nn_data_set not in ["cifar10", "cifar100"]:
            raise ValueError("nn_data_set not supported.")
        for ifm_shape in self.ifm:
            for elem in ifm_shape:
                if elem <= 0:
                    raise ValueError("ifm values should be greater than 0")
        if self.nn_data not in ["TEST", "TRAIN"]:
            raise ValueError("nn_data should be either 'TEST' or 'TRAIN'")
        if self.batch <= 0:
            raise ValueError("batch should be greater than 0")
        if self.num_runs <= 0:
            raise ValueError("num_runs should be greater than 0")
        for (m, n) in self.xbar_sizes:
            if m <= 0 or n <= 0:
                raise ValueError("xbar_sizes should be greater than 0")
        for (hrs, lrs) in self.hrs_lrs:
            if hrs < 0.0 or lrs <= 0.0 or hrs >= lrs:
                raise ValueError("error in hrs_lrs should be greater than 0")
        if self.adc_type not in ["FP_ALPHA_ADC", "INF_ADC"]:
            raise ValueError("adc_type not valid.")
        for a in self.alpha:
            if a <= 0.0:
                raise ValueError("alpha should be greater than 0")
        for r in self.resolution:
            if r != -1 and r <= 0:
                raise ValueError("resolution should be greater than 0")
        for mode in self.m_mode:
            if mode not in [
                    "BNN_I", "BNN_II", "BNN_III", "BNN_IV", "BNN_V", "BNN_VI",
                    "TNN_I", "TNN_II", "TNN_III", "TNN_IV", "TNN_V"
            ]:
                raise ValueError(f"m_mode {mode} not valid.")
        for noise in self.hrs_noise:
            if noise < 0.0:
                raise ValueError("hrs_noise should be greater than 0")
        for noise in self.lrs_noise:
            if noise < 0.0:
                raise ValueError("lrs_noise should be greater than 0")

        # Checks for variability benchmarks
        if (self.resolution == [-1]) or (self.adc_type == "INF_ADC"):
            if len(self.alpha) != 1:
                print(
                    "alpha has no influence in this configuration. It is set to 1.0."
                )
                self.alpha = [1.0]
            if (self.adc_type != "INF_ADC"):
                raise ValueError(
                    "When using resolution=[-1], adc_type should be 'INF_ADC'")
            if (self.resolution != [-1]):
                raise ValueError(
                    "When using adc_type='INF_ADC', resolution should be set to [-1]"
                )

    def __post_init__(self):
        for field in self.__dataclass_fields__:
            if getattr(self, field) is None:
                raise ValueError(
                    f"Argument '{field}' is missing in ExpConfig. Please provide all the required arguments."
                )

        self.__check_paramters()
