import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import rc, ticker
import pandas as pd
import numpy as np
import math
import os
import ast
import pickle
import argparse
import json
from functools import reduce

from model_parser import *
from run import *
from RWTHColors import ColorManager

# Colors
cm = ColorManager()

colors = [
    cm.RWTHBlau(),      # blue
    cm.RWTHOrange(),    # orange
    cm.RWTHMaiGruen(),  # light green
    cm.RWTHGruen(),     # green
    cm.RWTHBordeaux(),  # bordeaux
    cm.RWTHTuerkis(),   # turquoise
    cm.RWTHMagenta(),   # magenta
    cm.RWTHPetrol(),    # petrol
    cm.RWTHViolett(),   # violet
]

grid_color = cm.RWTHSchwarz(25)

color_mode = {
    'BNN_I': colors[0],
    'BNN_II': colors[1],
    'BNN_III': colors[2],
    'BNN_IV': colors[3],
    'BNN_V': colors[4],
    'BNN_VI': colors[5],
    'TNN_I': colors[0],
    'TNN_II': colors[1],
    'TNN_III': colors[2],
    'TNN_IV': colors[3],
    'TNN_V': colors[4]
}

color_bits = {
    3: colors[0],
    4: colors[1],
    5: colors[2],
    6: colors[3],
    8: colors[4]
}

# Font
rc('text', usetex=True)
rc('text.latex', preamble="\\usepackage{libertine}")
title_fontsize = 12
tick_fontsize = 8
label_fontsize = 10
legend_fontsize = 8

# title_fontsize = 16
# tick_fontsize = 12
# label_fontsize = 14
# legend_fontsize = 10

# Figure sizes
fig_width = 12
fig_height = 3

# Other helpers
times_str = r"$\times$"
nn_labels = {"VGG7": "VGG-7", "LeNet": "LeNet-5"}


def adc_profile_plot(df: pd.DataFrame, store_path: str, s_cat: list,
                     d_cat: list, profiles: dict[int, dict]):
    bnn_mode_labels = ["BNN_I", "BNN_II",
                       "BNN_III", "BNN_IV", "BNN_V", "BNN_VI"]
    tnn_mode_labels = ["TNN_I", "TNN_II", "TNN_III", "TNN_IV", "TNN_V"]
    xbar_size = "[256, 256]"

    for nn_name in list(df['nn_name'].unique()):
        print(f"Generate plots for {nn_name}.")
        df_nn = df[(df['nn_name'] == nn_name) & (df['xbar_size'] == xbar_size)]

        if 'num_runs' in d_cat:
            max_num_runs = max(df_nn['num_runs'].unique())
            df_nn = df_nn[(df_nn['num_runs'] == max_num_runs)]

        # Count modes in experiment
        m_modes = list(df_nn['m_mode'].unique())
        bnn_modes = [bm for bm in bnn_mode_labels if bm in m_modes]
        tnn_modes = [tm for tm in tnn_mode_labels if tm in m_modes]
        mm_sets = {'bnn': bnn_modes, 'tnn': tnn_modes}

        layers = profiles[int(df_nn.loc[:, "config_idx"].iloc[0])].keys()
        print(f"Layers: {layers}.")

        for mm_set_name, mm_set in mm_sets.items():
            if len(mm_set) > 0:
                fig, axs = plt.subplots(1, len(mm_set), figsize=(
                    2*len(mm_set), 3), layout='constrained', sharey=True)
                axs = axs.flatten() if len(mm_set) > 1 else [axs]
                axs[0].set_ylabel("Frequency")
                for n, mm in enumerate(mm_set):
                    print(f"Plotting for {mm}.")
                    axs[n].set_title(mm.replace('NN_', ' '))
                    c_idx = int(
                        df_nn[(df_nn["m_mode"] == mm)].loc[:, "config_idx"].iloc[0])
                    for i, l_name in enumerate(layers):
                        l_data = profiles[c_idx][l_name]
                        color = colors[i]

                        bins = np.array([b for b, _ in l_data["hist"]])
                        cnts = np.array([c for _, c in l_data["hist"]])
                        density = cnts / cnts.max()
                        axs[n].step(bins, density, where='mid', linewidth=0.8,
                                    label=l_name, color=color, alpha=0.7)
                        axs[n].fill_between(bins, density, step='mid',
                                            color=color, alpha=0.05)

                    if bins.min() < 0:
                        axs[n].set_xlim([-500, 500])
                    else:
                        axs[n].set_xlim([0, 1000])

                    unique_handles_labels = {}
                    for handle, label in zip(axs[0].get_legend_handles_labels()[0],
                                             axs[0].get_legend_handles_labels()[1]):
                        if label not in unique_handles_labels.values():
                            unique_handles_labels[handle] = label

                    axs[n].tick_params(axis='both', labelsize=10)
                    axs[n].grid(axis='y', linestyle=':',
                                color=grid_color)
                    axs[n].set_xlabel(
                        r"ADC Input Current $(\mu A)$")

                fig.legend(
                    unique_handles_labels.keys(),
                    unique_handles_labels.values(),
                    loc='outside lower center',
                    ncol=3)

                fig.savefig(
                    f"{store_path}/adc_profile_{mm_set_name}_{nn_name}.pdf", dpi=300)
                fig.savefig(
                    f"{store_path}/adc_profile_{mm_set_name}_{nn_name}.png", dpi=300)


def adc_calibration_plot(df: pd.DataFrame, store_path: str, s_cat: list,
                         d_cat: list, plt_legend: bool = True
                         ):
    bnn_mode_labels = ["BNN_I", "BNN_II",
                       "BNN_III", "BNN_IV", "BNN_V", "BNN_VI"]
    tnn_mode_labels = ["TNN_I", "TNN_II", "TNN_III", "TNN_IV", "TNN_V"]

    adc_calib_mode_markers = {'MAX': 'x', 'CALIB': 'o'}
    adc_calib_mode_legend = {"Uncalib. ADC": 'x', "Calib. ADC": 'o'}

    for nn_name in list(df['nn_name'].unique()):
        print(f"Generate plots for {nn_name}.")
        df_nn = df[(df['nn_name'] == nn_name)]

        if 'num_runs' in d_cat:
            max_num_runs = max(df_nn['num_runs'].unique())
            df_nn = df_nn[(df_nn['num_runs'] == max_num_runs)]

        if 'resolution' in d_cat:
            print(f"Found experiment for ADC resolution (x-Axis).")
            resolutions = df_nn['resolution'].unique()

            xbar_sizes = df_nn['xbar_size'].unique()
            adc_calib_modes = df_nn['adc_calib_mode'].unique()

            # Count modes in experiment
            m_modes = list(df_nn['m_mode'].unique())
            bnn_modes = [bm for bm in bnn_mode_labels if bm in m_modes]
            tnn_modes = [tm for tm in tnn_mode_labels if tm in m_modes]
            mm_sets = {'BNN': bnn_modes, 'TNN': tnn_modes}

            for mm_set_name, mm_set in mm_sets.items():
                if len(mm_set) > 0:
                    fig, axs = plt.subplots(1, len(xbar_sizes), figsize=(
                        2 * len(xbar_sizes), fig_height), layout='tight', sharey=True)
                    axs = axs.flatten() if len(xbar_sizes) > 1 else [axs]
                    axs[0].set_ylabel("Top-1 Accuracy (\\%)",
                                      fontsize=label_fontsize)
                    for n, xs in enumerate(xbar_sizes):
                        print(f"Plotting for {mm_set_name}: {xs}")
                        axs[n].set_title(
                            f"{nn_labels[nn_name]} - {mm_set_name} - {xs[1:-1].replace(', ', times_str)}", fontsize=title_fontsize)
                        df_xs_mms = df_nn[(df_nn['xbar_size'] == xs) & (
                            df_nn['m_mode'].str.startswith(mm_set_name))]
                        base_top1 = df_xs_mms['top1_baseline'].unique()
                        assert len(base_top1) == 1
                        axs[n].axhline(
                            y=base_top1[0], color='black', linestyle='--')
                        for mm in mm_set:
                            for acm in adc_calib_modes:
                                df_xs_mm = df_xs_mms[(df_xs_mms['m_mode'] == mm) &
                                                     (df_xs_mms['adc_calib_mode'] == acm)
                                                     ].sort_values(by='resolution')
                                axs[n].plot(df_xs_mm['resolution'],
                                            df_xs_mm['top1'],
                                            marker=adc_calib_mode_markers[acm],
                                            label=f"{mm.replace('NN_', ' ')}",
                                            color=color_mode[mm])

                        axs[n].set_ylim(1, min(100, base_top1 + 1))
                        axs[n].set_xlim(axs[n].get_xlim()[::-1])
                        axs[n].set_xticks(resolutions)
                        axs[n].tick_params(axis='both', labelsize=10)
                        axs[n].set_xlabel(
                            r"ADC Resolution (bits)", fontsize=label_fontsize)
                        axs[n].grid(axis='y', linestyle=':', color=grid_color)

                        if plt_legend:
                            # Create structured legend
                            # Legend for markers (ADC calibration modes)
                            marker_legend = [
                                # markersize=10,
                                mlines.Line2D([], [], color='black',
                                              marker=m, linestyle='None', label=acm)
                                for acm, m in adc_calib_mode_legend.items()
                            ]
                        # Legend for colors (Mapping modes)
                            color_legend = [
                                # linewidth=2,
                                mlines.Line2D(
                                    [], [], color=c, marker='None', linestyle='-', label=mm.replace('NN_', ' '))
                                for mm, c in color_mode.items() if mm in mm_set
                            ]
                            axs[0].legend(handles=marker_legend + color_legend,
                                          loc='lower left', fontsize=legend_fontsize, ncol=1)
                        fig.savefig(
                            f"{store_path}/adc_calib_{mm_set_name}_{nn_name}.pdf", dpi=300)
                        fig.savefig(
                            f"{store_path}/adc_calib_{mm_set_name}_{nn_name}.png", dpi=300)


def scale_variability_plot(df: pd.DataFrame,
                           store_path: str,
                           s_cat: list,
                           d_cat: list,
                           state: str,
                           plt_legend_nr: int = -1) -> None:
    bnn_mode_labels = ["BNN_I", "BNN_II",
                       "BNN_III", "BNN_IV", "BNN_V", "BNN_VI"]
    tnn_mode_labels = ["TNN_I", "TNN_II", "TNN_III", "TNN_IV", "TNN_V"]

    this_label = f"{state}_noise"
    other_label = f"{'hrs' if state == 'lrs' else 'lrs'}_noise"

    for nn_name in list(df['nn_name'].unique()):
        print(f"Generate plots for {nn_name}.")
        df_nn = df[(df['nn_name'] == nn_name)]

        if 'num_runs' in d_cat:
            max_num_runs = max(df_nn['num_runs'].unique())
            df_nn = df_nn[(df_nn['num_runs'] == max_num_runs)]

        if this_label in d_cat:
            print(f"Found experiment for {this_label} (x-Axis).")
            df_nn = df_nn[(df_nn[other_label] == 0.0)]
            hrs_lrs = df_nn['hrs_lrs'].unique()
            noise = df_nn[this_label].unique()

            xbar_sizes = df_nn['xbar_size'].unique()

            # Count modes in experiment
            m_modes = list(df_nn['m_mode'].unique())
            bnn_modes = [bm for bm in bnn_mode_labels if bm in m_modes]
            tnn_modes = [tm for tm in tnn_mode_labels if tm in m_modes]
            mm_sets = {'BNN': bnn_modes, 'TNN': tnn_modes}

            for mm_set_name, mm_set in mm_sets.items():
                if len(mm_set) > 0:
                    for hrs_lrs_str in hrs_lrs:
                        hrs, lrs = ast.literal_eval(hrs_lrs_str)
                        df_hrs_lrs = df_nn[(df_nn['hrs_lrs'] == hrs_lrs_str)]
                        fig, axs = plt.subplots(1, len(xbar_sizes), figsize=(
                            3.7 * len(xbar_sizes), 3), layout='tight', sharey=True)
                        axs = axs.flatten() if len(
                            xbar_sizes) > 1 else [axs[0]]
                        axs[0].set_ylabel("Top-1 Accuracy (\\%)")

                        for n, xs in enumerate(xbar_sizes):
                            print(f"Plotting for {mm_set_name}: {xs}")
                            axs[n].set_title(
                                f"Crossbar Size: {xs[1:-1].replace(', ', times_str)}")
                            df_xs_mms = df_hrs_lrs[(df_hrs_lrs['xbar_size'] == xs) & (
                                df_hrs_lrs['m_mode'].str.startswith(mm_set_name))]
                            base_top1 = df_xs_mms['top1_baseline'].unique()
                            assert len(base_top1) == 1
                            axs[n].axhline(
                                y=base_top1[0], color='black', linestyle='--')

                            for mm in mm_set:
                                df_xs_mm = df_xs_mms[(df_xs_mms['m_mode'] == mm)
                                                     ].sort_values(by=this_label)
                                print(
                                    f"---------------------------------------------------------"
                                )
                                print(f"Best results for {nn_name} and {mm}:")
                                best_of = df_xs_mm[
                                    (df_xs_mm['top1'] >=
                                     max(df_xs_mm['top1_baseline'].unique()) -
                                     1)].sort_values(by='lrs_noise', ascending=True)
                                print(best_of[[
                                    'xbar_size', 'm_mode', 'hrs_noise', 'lrs_noise', 'top1'
                                ]])
                                axs[n].plot(df_xs_mm[this_label],
                                            df_xs_mm['top1'],
                                            marker='x',
                                            label=f"{mm.replace('NN_', ' ')}",
                                            color=color_mode[mm])

                            axs[n].set_xticks(noise)
                            axs[n].xaxis.set_major_formatter(
                                ticker.StrMethodFormatter("{x:.2g}"))
                            axs[n].tick_params(axis='both', labelsize=10)
                            axs[n].set_xlabel(
                                rf"{state.upper()} $\sigma (\mu A)$")
                            axs[n].grid(axis='y', linestyle=':',
                                        color=grid_color)

                        axs[0].legend(loc='lower left', fontsize=8, ncol=2)
                        fig.savefig(
                            f"{store_path}/{state}_scale_var_{mm_set_name}_{nn_name}_{hrs_lrs}.pdf", dpi=300)
                        fig.savefig(
                            f"{store_path}/{state}_scale_var_{mm_set_name}_{nn_name}_{hrs_lrs}.png", dpi=300)


def scale_variability_plot_with_c2c(df: pd.DataFrame,
                                    df_c2c: pd.DataFrame,
                                    store_path: str,
                                    s_cat: list,
                                    d_cat: list,
                                    state: str,
                                    plt_legend: bool = True) -> None:
    bnn_mode_labels = ["BNN_I", "BNN_II",
                       "BNN_III", "BNN_IV", "BNN_V", "BNN_VI"]
    tnn_mode_labels = ["TNN_I", "TNN_II", "TNN_III", "TNN_IV", "TNN_V"]

    this_label = f"{state}_noise"
    other_label = f"{'hrs' if state == 'lrs' else 'lrs'}_noise"

    var_markers = ['x', 'o']
    var_legend = {"D2D": 'x', "D2D + C2C": 'o'}

    for nn_name in list(df['nn_name'].unique()):
        print(f"Generate plots for {nn_name}.")
        df_nn = df[(df['nn_name'] == nn_name)]
        df_c2c_nn = df_c2c[(df_c2c['nn_name'] == nn_name)]

        if 'num_runs' in d_cat:
            max_num_runs = max(df_nn['num_runs'].unique())
            df_nn = df_nn[(df_nn['num_runs'] == max_num_runs)]
            df_c2c_nn = df_c2c_nn[(df_c2c_nn['num_runs'] == max_num_runs)]

        if this_label in d_cat:
            print(f"Found experiment for {this_label} (x-Axis).")
            df_nn = df_nn[(df_nn[other_label] == 0.0)]
            df_c2c_nn = df_c2c_nn[(df_c2c_nn[other_label] == 0.0)]
            hrs_lrs = df_nn['hrs_lrs'].unique()
            noise = df_nn[this_label].unique()

            xbar_sizes = df_nn['xbar_size'].unique()

            # Count modes in experiment
            m_modes = list(df_nn['m_mode'].unique())
            bnn_modes = [bm for bm in bnn_mode_labels if bm in m_modes]
            tnn_modes = [tm for tm in tnn_mode_labels if tm in m_modes]
            mm_sets = {'BNN': bnn_modes, 'TNN': tnn_modes}

            for mm_set_name, mm_set in mm_sets.items():
                if len(mm_set) > 0:
                    for hrs_lrs_str in hrs_lrs:
                        hrs, lrs = ast.literal_eval(hrs_lrs_str)
                        df_hrs_lrs = df_nn[(df_nn['hrs_lrs'] == hrs_lrs_str)]
                        df_c2c_hrs_lrs = df_c2c_nn[(
                            df_c2c_nn['hrs_lrs'] == hrs_lrs_str)]
                        fig, axs = plt.subplots(1, len(xbar_sizes), figsize=(
                            fig_width * len(xbar_sizes) / 4, 3), layout='tight', sharey=True)
                        axs = axs.flatten() if len(mm_set) > 1 else [axs]
                        axs[0].set_ylabel(
                            "Top-1 Accuracy (\\%)", fontsize=label_fontsize)

                        for n, xs in enumerate(xbar_sizes):
                            print(f"Plotting for {mm_set_name}: {xs}")
                            axs[n].set_title(
                                f"{mm_set_name} - Crossbar Size: {xs[1:-1].replace(', ', times_str)}", fontsize=title_fontsize)
                            df_xs_mms = df_hrs_lrs[(df_hrs_lrs['xbar_size'] == xs) & (
                                df_hrs_lrs['m_mode'].str.startswith(mm_set_name))]
                            df_c2c_xs_mms = df_c2c_hrs_lrs[(df_c2c_hrs_lrs['xbar_size'] == xs) & (
                                df_c2c_hrs_lrs['m_mode'].str.startswith(mm_set_name))]
                            base_top1 = df_xs_mms['top1_baseline'].unique()
                            assert len(base_top1) == 1
                            axs[n].axhline(
                                y=base_top1[0], color='black', linestyle='--')

                            for mm in mm_set:
                                df_xs_mm = df_xs_mms[(df_xs_mms['m_mode'] == mm)
                                                     ].sort_values(by=this_label)
                                df_c2c_xs_mm = df_c2c_xs_mms[(df_c2c_xs_mms['m_mode'] == mm)
                                                             ].sort_values(by=this_label)
                                print(
                                    f"---------------------------------------------------------"
                                )
                                print(f"Best results for {nn_name} and {mm}:")
                                best_of = df_xs_mm[
                                    (df_xs_mm['top1'] >=
                                     max(df_xs_mm['top1_baseline'].unique()) -
                                     1)].sort_values(by='lrs_noise', ascending=True)
                                print(best_of[[
                                    'xbar_size', 'm_mode', 'hrs_noise', 'lrs_noise', 'top1'
                                ]])
                                axs[n].plot(df_xs_mm[this_label],
                                            df_xs_mm['top1'],
                                            marker=var_markers[0],
                                            color=color_mode[mm])
                                axs[n].plot(df_c2c_xs_mm[this_label],
                                            df_c2c_xs_mm['top1'],
                                            marker=var_markers[1],
                                            color=color_mode[mm])

                            axs[n].set_xticks(noise)
                            axs[n].xaxis.set_major_formatter(
                                ticker.StrMethodFormatter("{x:.2g}"))
                            axs[n].tick_params(
                                axis='both', labelsize=tick_fontsize)
                            axs[n].set_xlabel(
                                rf"$\sigma_{{{state}}} \: (\mu A)$",
                                fontsize=label_fontsize)
                            axs[n].grid(axis='y', linestyle=':',
                                        color=grid_color)
                            axs[n].set_ylim(1, min(100, base_top1 + 1))

                        if plt_legend:
                            # Create structured legend
                            # Legend for markers (ADC calibration modes)
                            marker_legend = [
                                # markersize=10,
                                mlines.Line2D([], [], color='black',
                                              marker=m, linestyle='None', label=acm)
                                for acm, m in var_legend.items()
                            ]
                            # Legend for colors (Mapping modes)
                            color_legend = [
                                # linewidth=2,
                                mlines.Line2D(
                                    [], [], color=c, marker='None', linestyle='-', label=mm.replace('NN_', ' '))
                                for mm, c in color_mode.items() if mm_set_name in mm
                            ]
                            axs[0].legend(handles=marker_legend + color_legend,
                                          loc='lower left', fontsize=legend_fontsize, ncol=1)

                        fig.savefig(
                            f"{store_path}/{state}_scale_var_{mm_set_name}_{nn_name}_c2c.pdf", dpi=300)
                        fig.savefig(
                            f"{store_path}/{state}_scale_var_{mm_set_name}_{nn_name}_c2c.png", dpi=300)


def parasitics_plot(df: pd.DataFrame,
                    store_path: str,
                    s_cat: list,
                    d_cat: list) -> None:
    bnn_mode_labels = ["BNN_I", "BNN_II",
                       "BNN_III", "BNN_IV", "BNN_V", "BNN_VI"]
    tnn_mode_labels = ["TNN_I", "TNN_II", "TNN_III", "TNN_IV", "TNN_V"]

    for nn_name in list(df['nn_name'].unique()):
        print(f"Generate plots for {nn_name}.")
        df_nn = df[(df['nn_name'] == nn_name)]

        if 'num_runs' in d_cat:
            max_num_runs = max(df_nn['num_runs'].unique())
            df_nn = df_nn[(df_nn['num_runs'] == max_num_runs)]

        if 'w_res' in d_cat:
            print("Found experiment for parasitic resistance (x-Axis).")
            df_nn = df_nn[(df_nn['w_res'] != 0.1)]
            w_res = df_nn['w_res'].unique()

            xbar_sizes = df_nn['xbar_size'].unique()

        # Count modes in experiment
            m_modes = list(df_nn['m_mode'].unique())
            bnn_modes = [bm for bm in bnn_mode_labels if bm in m_modes]
            tnn_modes = [tm for tm in tnn_mode_labels if tm in m_modes]
            mm_sets = {'BNN': bnn_modes, 'TNN': tnn_modes}

            for mm_set_name, mm_set in mm_sets.items():
                if len(mm_set) > 0:
                    fig, axs = plt.subplots(1, len(xbar_sizes), figsize=(
                        fig_width * len(xbar_sizes) / 3, fig_height),
                        layout='tight',
                        sharey=True)
                    axs = axs.flatten() if len(mm_set) > 1 else [axs]
                    axs[0].set_ylabel("Top-1 Accuracy (\\%)",
                                      fontsize=label_fontsize)
                    for n, xs in enumerate(xbar_sizes):
                        print(f"Plotting for {mm_set_name}: {xs}")

                        axs[n].set_title(
                            f"{mm_set_name} - Crossbar Size: {xs[1:-1].replace(', ', times_str)}",
                            fontsize=title_fontsize)
                        df_xs_mms = df_nn[(df_nn['xbar_size'] == xs) & (
                            df_nn['m_mode'].str.startswith(mm_set_name))]
                        base_top1 = df_xs_mms['top1_baseline'].unique()
                        assert len(base_top1) == 1
                        axs[n].axhline(
                            y=base_top1[0], color='black', linestyle='--')
                        for mm in mm_set:
                            df_xs_mm = df_xs_mms[(df_xs_mms['m_mode'] == mm)
                                                 ].sort_values(by='w_res')
                            axs[n].plot(df_xs_mm['w_res'],
                                        df_xs_mm['top1'],
                                        marker='x',
                                        label=f"{mm.replace('NN_', ' ')}",
                                        color=color_mode[mm])

                        axs[n].set_xticks(w_res)
                        axs[n].xaxis.set_major_formatter(
                            ticker.StrMethodFormatter("{x:.2g}"))
                        axs[n].set_xlabel(
                            r"Parasitic Resistance ($\Omega$)", fontsize=label_fontsize)
                        axs[n].tick_params(
                            axis='both', labelsize=tick_fontsize)

                        axs[n].set_ylim(1, min(100, base_top1 + 1))
                        axs[n].grid(axis='y', linestyle=':', color=grid_color)

                    axs[0].legend(loc='lower left',
                                  fontsize=label_fontsize, ncol=2)
                    fig.savefig(
                        f"{store_path}/parasitics_{mm_set_name}_{nn_name}.pdf", dpi=300)
                    fig.savefig(
                        f"{store_path}/parasitics_{mm_set_name}_{nn_name}.png", dpi=300)


def parasitics_multi_plot(df: pd.DataFrame,
                          store_path: str,
                          s_cat: list,
                          d_cat: list) -> None:
    bnn_mode_labels = ["BNN_I", "BNN_II",
                       "BNN_III", "BNN_IV", "BNN_V", "BNN_VI"]
    tnn_mode_labels = ["TNN_I", "TNN_II", "TNN_III", "TNN_IV", "TNN_V"]

    nn_marker = {"VGG7": 'o', "LeNet": 'x'}
    nn_label = {"VGG7": 'VGG-7', "LeNet": 'LeNet-5'}
    nn_baseline_linestyle = {"VGG7": '--', "LeNet": ':'}

    if 'num_runs' in d_cat:
        max_num_runs = max(df_nn['num_runs'].unique())
        df = df[(df['num_runs'] == max_num_runs)]

    if 'w_res' in d_cat:
        print("Found experiment for parasitic resistance (x-Axis).")
        df = df[(df['w_res'] != 0.1)]
        w_res = df['w_res'].unique()

        xbar_sizes = df['xbar_size'].unique()

        # Count modes in experiment
        m_modes = list(df['m_mode'].unique())
        bnn_modes = [bm for bm in bnn_mode_labels if bm in m_modes]
        tnn_modes = [tm for tm in tnn_mode_labels if tm in m_modes]
        mm_sets = {'BNN': bnn_modes, 'TNN': tnn_modes}
        nn_names = df['nn_name'].unique()

        for mm_set_name, mm_set in mm_sets.items():
            if len(mm_set) > 0:
                fig, axs = plt.subplots(1, len(xbar_sizes), figsize=(
                    fig_width * len(xbar_sizes) / 3, fig_height),
                    layout='tight',
                    sharey=True)
                axs = axs.flatten() if len(mm_set) > 1 else [axs]
                axs[0].set_ylabel("Top-1 Accuracy (\\%)",
                                  fontsize=label_fontsize)
                for n, xs in enumerate(xbar_sizes):
                    print(f"Plotting for {mm_set_name}: {xs}")

                    axs[n].set_title(
                        f"{mm_set_name} - Crossbar Size: {xs[1:-1].replace(', ', times_str)}",
                        fontsize=title_fontsize)
                    df_xs_mms = df[(df['xbar_size'] == xs) & (
                        df['m_mode'].str.startswith(mm_set_name))]
                    for nn in nn_names:
                        df_xs_mms_nn = df_xs_mms[df_xs_mms['nn_name'] == nn]
                        base_top1 = df_xs_mms_nn['top1_baseline'].unique()
                        assert len(base_top1) == 1
                        axs[n].axhline(
                            y=base_top1[0], color='black', linestyle=nn_baseline_linestyle[nn])
                        for mm in mm_set:
                            df_xs_mm = df_xs_mms_nn[(df_xs_mms_nn['m_mode'] == mm)
                                                    ].sort_values(by='w_res')
                            axs[n].plot(df_xs_mm['w_res'],
                                        df_xs_mm['top1'],
                                        marker=nn_marker[nn],
                                        label=f"{mm.replace('NN_', ' ')}",
                                        color=color_mode[mm])

                    axs[n].set_xticks(w_res)
                    axs[n].xaxis.set_major_formatter(
                        ticker.StrMethodFormatter("{x:.2g}"))
                    axs[n].set_xlabel(
                        r"Parasitic Resistance ($\Omega$)", fontsize=label_fontsize)
                    axs[n].tick_params(
                        axis='both', labelsize=tick_fontsize)

                    axs[n].set_ylim(1, 101)
                    axs[n].grid(axis='y', linestyle=':', color=grid_color)

                # Create structured legend
                # Legend for markers (ADC calibration modes)
                marker_legend = [
                    # markersize=10,
                    mlines.Line2D([], [], color='black',
                                  marker=m, linestyle='None', label=nn_label[nn])
                    for nn, m in nn_marker.items()
                ]
                # Legend for colors (Mapping modes)
                color_legend = [
                    # linewidth=2,
                    mlines.Line2D(
                        [], [], color=c, marker='None', linestyle='-', label=mm.replace('NN_', ' '))
                    for mm, c in color_mode.items() if mm_set_name in mm
                ]
                axs[0].legend(handles=marker_legend + color_legend,
                              loc='lower left', fontsize=legend_fontsize, ncol=1)
                fig.savefig(
                    f"{store_path}/parasitics_{mm_set_name}.pdf", dpi=300)
                fig.savefig(
                    f"{store_path}/parasitics_{mm_set_name}.png", dpi=300)


def get_exp_products(config: str):
    exp_name = config.split('/')[-1].split('.json')[0]
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    exp_result_path = repo_path + '/results/' + exp_name
    df = pd.read_csv(f"{exp_result_path}/{exp_name}.csv")
    return exp_name, repo_path, exp_result_path, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        help='Path to experiment config',
                        required=True)

    parser.add_argument('--secondary_config',
                        type=str,
                        default="",
                        help='Path to secondary experiment config',
                        required=False)

    args = parser.parse_args()

    with open(args.config, 'r') as json_file:
        cfg = json.load(json_file)

    exp_name, repo_path, exp_result_path, df = get_exp_products(args.config)

    categories = df.columns
    cat_static = []  # Categories (columns) that all experiments have in common
    cat_dynamic = []  # Categories that change for at least one experiment

    for c in categories:
        if len(set(df[c])) > 1:
            if type(df[c].iloc[0]) in [float, np.float64, np.float32]:
                if all(math.isnan(x) for x in df[c]):
                    cat_static.append(c)
                    continue
            cat_dynamic.append(c)
        else:
            cat_static.append(c)

    print(
        f"The benchmark has the following (static) properties:\n{cat_static}")
    print(f"The benchmarks varies the following properties:\n{cat_dynamic}")

    store_path = f"{exp_result_path}"

    if exp_name.startswith('adc_profile'):
        profiles: dict[int, dict] = {
            c: json.load(open(f"{exp_result_path}/adc_prof_{c}.json", 'r'))
            for c in df.loc[:, "config_idx"].to_numpy(dtype=np.int32)}
        adc_profile_plot(df=df,
                         store_path=store_path,
                         s_cat=cat_static,
                         d_cat=cat_dynamic,
                         profiles=profiles)

    elif exp_name.startswith('adc_calibration'):
        adc_calibration_plot(df=df,
                             store_path=store_path,
                             s_cat=cat_static,
                             d_cat=cat_dynamic,
                             plt_legend=(cfg['nn_names'][0] == 'VGG7'))

    elif exp_name.startswith('parasitics'):
        if args.secondary_config:
            _, _, _, df_other = get_exp_products(args.secondary_config)
            df = pd.concat([df, df_other], ignore_index=True)
            parasitics_multi_plot(df=df,
                                  store_path=store_path,
                                  s_cat=cat_static,
                                  d_cat=cat_dynamic)
        else:
            parasitics_plot(df=df,
                            store_path=store_path,
                            s_cat=cat_static,
                            d_cat=cat_dynamic)

    elif exp_name == 'lrs_var_scaling':
        if not args.secondary_config:
            scale_variability_plot(df=df,
                                   store_path=store_path,
                                   s_cat=cat_static,
                                   d_cat=cat_dynamic,
                                   state='lrs')
        else:
            _, _, _, df_c2c = get_exp_products(args.secondary_config)
            scale_variability_plot_with_c2c(df=df,
                                            df_c2c=df_c2c,
                                            store_path=store_path,
                                            s_cat=cat_static,
                                            d_cat=cat_dynamic,
                                            state='lrs')

    elif exp_name == 'hrs_var_scaling':
        if not args.secondary_config:
            scale_variability_plot(df=df,
                                   store_path=store_path,
                                   s_cat=cat_static,
                                   d_cat=cat_dynamic,
                                   state='hrs')
        else:
            _, _, _, df_c2c = get_exp_products(args.secondary_config)
            scale_variability_plot_with_c2c(df=df,
                                            df_c2c=df_c2c,
                                            store_path=store_path,
                                            s_cat=cat_static,
                                            d_cat=cat_dynamic,
                                            state='hrs',
                                            plt_legend=False)
    else:
        raise Exception(f"Plot for experiment {exp_name} not implemented.")
