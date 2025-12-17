##############################################################################
# Copyright (C) 2025 Rebecca Pelke                                           #
# All Rights Reserved                                                        #
#                                                                            #
# This is work is licensed under the terms described in the LICENSE file     #
# found in the root directory of this source tree.                           #
##############################################################################
from typing import List, Tuple, Optional, Union, get_type_hints
from dataclasses import dataclass
import numpy as np


@dataclass
class ExpConfig:
    nn_names: List[str]
    ifm: List[List[int]]
    nn_data_set: str
    nn_data: str
    batch: int
    num_runs: int
    xbar_size: List[Tuple[int, int]]
    digital_only: bool
    hrs_lrs: Optional[List[Tuple[float]]]
    gmin_gmax: Optional[List[Tuple[float]]]
    adc_type: str
    hrs_noise: List[float]
    lrs_noise: List[float]
    verbose: bool
    m_mode: List[str]
    resolution: Optional[List[int]] = None
    adc_profile: Optional[bool] = None
    read_disturb: Optional[bool] = None
    V_read: Optional[List[float]] = None
    t_read: Optional[List[float]] = None
    read_disturb_update_freq: Optional[int] = None
    read_disturb_mitigation_strategy: Optional[str] = None
    read_disturb_mitigation_fp: Optional[List[float]] = None
    read_disturb_update_tolerance: Optional[List[float]] = None
    parasitics: Optional[bool] = None
    w_res: Optional[List[float]] = None

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
        for (m, n) in self.xbar_size:
            if m <= 0 or n <= 0:
                raise ValueError("xbar_size should be greater than 0")

        if self.hrs_lrs is not None:
            for (hrs, lrs) in self.hrs_lrs:
                if hrs < 0.0 or lrs <= 0.0 or hrs >= lrs:
                    raise ValueError(
                        "error in hrs_lrs should be greater than 0")
        else:
            if self.gmin_gmax is None:
                raise ValueError(
                    "Either hrs_lrs or gmin_gmax should be provided.")
            else:
                if self.V_read is None:
                    raise ValueError(
                        "V_read should be provided when gmin_gmax is used.")
                for (gmin, gmax) in self.gmin_gmax:
                    if gmin < 0.0 or gmax <= 0.0 or gmin >= gmax:
                        raise ValueError(
                            "Error in gmin_gmax: should be greater than 0")

        # Check ADC parameters
        if self.adc_type not in ["FP_ALPHA_ADC", "INF_ADC"]:
            raise ValueError("adc_type not valid.")
        if self.adc_type != "INF_ADC":
            # TODO: Add calibration params
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

        # Check read disturb parameters
        if (self.read_disturb):
            if self.V_read is None or self.t_read is None:
                raise ValueError(
                    "V_read and t_read should be provided when read_disturb is True"
                )
            for v in self.V_read:
                if v >= 0.0:
                    raise ValueError(
                        "Read disturb model requires negative V_read values")
            for t in self.t_read:
                if t <= 0.0:
                    raise ValueError("t_read should be greater than 0")
            if self.read_disturb_update_freq is not None:
                for f in self.read_disturb_update_freq:
                    if f <= 0:
                        raise ValueError(
                            "read_disturb_update_freq should be greater than 0 (minimum: 1)"
                        )
            if self.read_disturb_mitigation_strategy is not None:
                if self.read_disturb_mitigation_strategy == "SOFTWARE":
                    if self.read_disturb_mitigation_fp is None:
                        raise ValueError(
                            "read_disturb_mitigation_fp should be provided for SOFTWARE strategy"
                        )
                    else:
                        for fp in self.read_disturb_mitigation_fp:
                            if fp < 1.0:
                                raise ValueError(
                                    "read_disturb_mitigation_fp must be at least 1.0."
                                )

                elif self.read_disturb_mitigation_strategy == "CELL_BASED":
                    if self.read_disturb_update_tolerance is None:
                        raise ValueError(
                            "read_disturb_update_tolerance should be provided for CELL_BASED strategy"
                        )
                elif self.read_disturb_mitigation_strategy != "OFF":
                    raise ValueError(
                        "read_disturb_mitigation_strategy should be either 'OFF', 'SOFTWARE', or 'CELL_BASED'"
                    )

        # Check parasitics parameters
        if (self.parasitics):
            if self.w_res is None:
                raise ValueError(
                    "w_res should be provided when parasitics is True.")
            for res in self.w_res:
                if res < 0.0:
                    raise ValueError("w_res should be non-negative.")
            if self.V_read is None:
                raise ValueError(
                    "V_read should be provided when parasitics is True.")
            for v in self.V_read:
                if v >= 0.0:
                    raise ValueError(
                        "Parasitics model requires negative V_read values")

    def __post_init__(self):
        type_hints = get_type_hints(self.__class__)
        for field_name, field_type in type_hints.items():
            value = getattr(self, field_name)

            is_optional = (getattr(field_type, '__origin__', None) is Union
                           and type(None) in field_type.__args__)

            if not is_optional and value is None:
                raise ValueError(
                    f"Argument '{field_name}' is missing in ExpConfig. Please provide all required arguments."
                )
        self.__check_paramters()


@dataclass
class SimulationStats:
    config: dict
    config_idx: int
    cycles_p: np.ndarray
    cycles_m: np.ndarray
    write_ops: int
    mvm_ops: int
    refresh_ops: int
    refresh_cell_ops: int
