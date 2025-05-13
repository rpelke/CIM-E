import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import os
import ast
import argparse
import json

from model_parser import *
from run import *

colors = [
    "#00549F",  # blue
    "#A11035",  # bordeaux
    "#57AB27",  # green
    "#F6A800",  # orange
    "#0098A1",  # turquoise
    "#BDCD00",  # light green
    "E30066"  # magenta
]

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


def adc_alpha_plot(df: pd.DataFrame, store_path: str, s_cat: list,
                   d_cat: list) -> None:
    for nn_name in list(df['nn_name'].unique()):

        print(f"Generate plots for {nn_name}.")
        df_nn = df[(df['nn_name'] == nn_name)]

        if 'num_runs' in d_cat:
            max_num_runs = max(df_nn['num_runs'].unique())
            df_nn = df_nn[(df_nn['num_runs'] == max_num_runs)]

        if 'alpha' in d_cat:
            print("Found experiment for alpha (x-Axis).")
            bits = list(df_nn['resolution'].unique())
            c_mode = list(df_nn['m_mode'].unique())

            for c_m in c_mode:
                plt.figure(figsize=(4, 3))
                plt.rcParams['font.family'] = 'serif'
                plt.xlabel('ADC Clipping ' + r'$\alpha$', fontsize=14)
                plt.ylabel('Top-1 Accuracy (%)', fontsize=14)

                print(
                    f"---------------------------------------------------------"
                )
                print(f"Best results for {nn_name} and {c_m}:")
                best_of = df_nn[(df_nn['m_mode'] == c_m) & (
                    df_nn['top1'] >= max(df_nn['top1_baseline'].unique()) -
                    1)].sort_values(by='resolution', ascending=True)
                print(best_of[[
                    'xbar_size', 'm_mode', 'resolution', 'alpha', 'top1'
                ]])

                base_top1 = df_nn[(
                    df_nn['m_mode'] == c_m)]['top1_baseline'].unique()
                assert len(base_top1) == 1
                plt.axhline(y=base_top1[0], color='black', linestyle='--')

                for b in bits:
                    df_nn_cm_b = df_nn[(df_nn['m_mode'] == c_m) & (
                        df_nn['resolution'] == b)].sort_values(by='alpha')
                    plt.plot(df_nn_cm_b['alpha'],
                             df_nn_cm_b['top1'],
                             marker='x',
                             label=f"{b} Bit",
                             color=color_bits[b])

                plt.legend(loc='lower right', fontsize=12, ncol=1)

                if nn_name.startswith('cifar100'):
                    plt.ylim(1, min(100, base_top1 + 2))
                elif nn_name.startswith('cifar10'):
                    plt.ylim(10, min(100, base_top1 + 2))
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.tight_layout()
                plt.savefig(f"{store_path}/{nn_name}_{c_m}_adc.png")
                plt.savefig(f"{store_path}/{nn_name}_{c_m}_adc.pdf")


def variability_lrs_plot(df: pd.DataFrame,
                         store_path: str,
                         s_cat: list,
                         d_cat: list,
                         plt_legend_nr: int = -1) -> None:
    for nn_name in list(df['nn_name'].unique()):
        print(f"Generate plots for {nn_name}.")
        df_nn = df[(df['nn_name'] == nn_name)]

        if 'num_runs' in d_cat:
            max_num_runs = max(df_nn['num_runs'].unique())
            df_nn = df_nn[(df_nn['num_runs'] == max_num_runs)]

        if 'lrs_noise' in d_cat:
            print("Found experiment for lrs_noise (x-Axis).")
            df_nn = df_nn[(df_nn['hrs_noise'] == 0.0)]
            hrs_lrs = df_nn['hrs_lrs'].unique()

            for hrs_lrs_str in hrs_lrs:
                hrs, lrs = ast.literal_eval(hrs_lrs_str)
                df_hrs_lrs = df_nn[(df_nn['hrs_lrs'] == hrs_lrs_str)]

                plt.figure(figsize=(3.7, 3))
                plt.rcParams['font.family'] = 'serif'
                plt.xlabel('LRS ' + r'$\sigma$' + ' ' + r'$(\mu A)$',
                           fontsize=14)
                plt.ylabel('Top-1 Accuracy (%)', fontsize=14)

                base_top1 = df_nn['top1_baseline'].unique()
                assert len(base_top1) == 1
                plt.axhline(y=base_top1[0], color='black', linestyle='--')

                c_modes = list(df_hrs_lrs['m_mode'].unique())
                for c_m in c_modes:
                    df_hrs_lrs_cm = df_hrs_lrs[(
                        df_hrs_lrs['m_mode'] == c_m)].sort_values(
                            by='lrs_noise')

                    print(
                        f"---------------------------------------------------------"
                    )
                    print(f"Best results for {nn_name} and {c_m}:")
                    best_of = df_hrs_lrs_cm[
                        (df_hrs_lrs_cm['m_mode'] == c_m)
                        & (df_hrs_lrs_cm['top1'] >=
                           max(df_hrs_lrs_cm['top1_baseline'].unique()) -
                           1)].sort_values(by='lrs_noise', ascending=True)
                    print(best_of[[
                        'xbar_size', 'm_mode', 'hrs_noise', 'lrs_noise', 'top1'
                    ]])

                    plt.plot(df_hrs_lrs_cm['lrs_noise'],
                             df_hrs_lrs_cm['top1'],
                             marker='x',
                             label=f"{c_m.replace('_', ' ')}",
                             color=color_mode[c_m])

                plt.legend(loc='lower left', fontsize=12, ncol=2)

                if nn_name.startswith('cifar100'):
                    plt.ylim(1, min(100, base_top1 + 1))
                elif nn_name.startswith('cifar10'):
                    plt.ylim(10, min(100, base_top1 + 1))
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.tight_layout()
                plt.savefig(f"{store_path}/hrs{hrs}_lrs{lrs}_lrs_noise.png")
                plt.savefig(f"{store_path}/hrs{hrs}_lrs{lrs}_lrs_noise.pdf")


def variability_hrs_plot(df: pd.DataFrame,
                         store_path: str,
                         s_cat: list,
                         d_cat: list,
                         plt_legend_nr: int = -1) -> None:
    for nn_name in list(df['nn_name'].unique()):
        print(f"Generate plots for {nn_name}.")
        df_nn = df[(df['nn_name'] == nn_name)]

        if 'num_runs' in d_cat:
            max_num_runs = max(df_nn['num_runs'].unique())
            df_nn = df_nn[(df_nn['num_runs'] == max_num_runs)]

        if 'hrs_noise' in d_cat:
            print("Found experiment for hrs_noise (x-Axis).")
            df_nn = df_nn[(df_nn['lrs_noise'] == 0.0)]
            hrs_lrs = df_nn['hrs_lrs'].unique()

            for hrs_lrs_str in hrs_lrs:
                hrs, lrs = ast.literal_eval(hrs_lrs_str)
                df_hrs_lrs = df_nn[(df_nn['hrs_lrs'] == hrs_lrs_str)]

                plt.figure(figsize=(3.7, 3))
                plt.rcParams['font.family'] = 'serif'
                plt.xlabel('HRS ' + r'$\sigma$' + ' ' + r'$(\mu A)$',
                           fontsize=14)
                plt.ylabel('Top-1 Accuracy (%)', fontsize=14)

                base_top1 = df_nn['top1_baseline'].unique()
                assert len(base_top1) == 1
                plt.axhline(y=base_top1[0], color='black', linestyle='--')

                c_modes = list(df_hrs_lrs['m_mode'].unique())
                for c_m in c_modes:
                    df_hrs_lrs_cm = df_hrs_lrs[(
                        df_hrs_lrs['m_mode'] == c_m)].sort_values(
                            by='hrs_noise')

                    print(
                        f"---------------------------------------------------------"
                    )
                    print(f"Best results for {nn_name} and {c_m}:")
                    best_of = df_hrs_lrs_cm[
                        (df_hrs_lrs_cm['m_mode'] == c_m)
                        & (df_hrs_lrs_cm['top1'] >=
                           max(df_hrs_lrs_cm['top1_baseline'].unique()) -
                           1)].sort_values(by='hrs_noise', ascending=True)
                    print(best_of[[
                        'xbar_size', 'm_mode', 'hrs_noise', 'lrs_noise', 'top1'
                    ]])

                    plt.plot(df_hrs_lrs_cm['hrs_noise'],
                             df_hrs_lrs_cm['top1'],
                             marker='x',
                             label=f"{c_m.replace('_', ' ')}",
                             color=color_mode[c_m])

                plt.legend(loc='lower left', fontsize=12, ncol=2)

                if nn_name.startswith('cifar100'):
                    plt.ylim(1, min(100, base_top1 + 1))
                elif nn_name.startswith('cifar10'):
                    plt.ylim(10, min(100, base_top1 + 1))
                plt.xticks(fontsize=14)
                plt.yticks(fontsize=14)
                plt.tight_layout()
                plt.savefig(f"{store_path}/hrs{hrs}_lrs{lrs}_hrs_noise.png")
                plt.savefig(f"{store_path}/hrs{hrs}_lrs{lrs}_hrs_noise.pdf")


def rd_sim_time_plot(df: pd.DataFrame,
                     store_path: str,
                     s_cat: list,
                     d_cat: list,
                     plt_legend_nr: int = -1) -> None:
    for nn_name in list(df['nn_name'].unique()):
        print(f"Generate plots for {nn_name}.")
        df_nn = df[(df['nn_name'] == nn_name)]
        if 'read_disturb_update_freq' in d_cat:
            v_read = df_nn['V_read'].unique()
            for v in v_read:
                df_nn_v = df_nn[(df_nn['V_read'] == v)]
                t_read = df_nn_v['t_read'].unique()
                for t in t_read:
                    df_nn_t = df_nn_v[(df_nn_v['t_read'] == t)]
                    freq = df_nn_t['read_disturb_update_freq'].unique()
                    print(
                        f"-----Accuracy: t_read: {t}, v_read: {v}, nn: {nn_name}-----"
                    )

                    # Plot accuracy
                    plt.figure(figsize=(3.5, 3))
                    plt.rcParams['font.family'] = 'serif'
                    plt.xlabel(
                        f"Batch @batch_size={df_nn_t['batch'].unique()[0]}",
                        fontsize=14)
                    plt.ylabel('Top-1 Accuracy (%)', fontsize=14)

                    top1_dict = {}
                    for f_idx, f in enumerate(freq):
                        df_nn_f = df_nn_t[(
                            df_nn_t['read_disturb_update_freq'] == f)]
                        assert len(df_nn_f) == 1

                        num_runs = df_nn_f['num_runs'].unique()[0]
                        top1 = df_nn_f['top1_batch'].unique()[0]
                        top1 = ast.literal_eval(top1)
                        top1_dict[f] = top1

                        plt.plot(range(1, num_runs + 1),
                                 top1,
                                 marker='x',
                                 label=f"{f}",
                                 color=colors[f_idx])
                        print(
                            f"Update freq: {f}: Top-1%: {round(100 * sum(top1) / len(top1), 2)}"
                        )

                    plt.legend(loc='lower left', fontsize=12, ncol=1)
                    plt.xticks(fontsize=14)
                    plt.yticks(fontsize=14)
                    plt.tight_layout()
                    plt.savefig(f"{store_path}/vread{v}_tread{t}_accuracy.png")
                    plt.savefig(f"{store_path}/vread{v}_tread{t}_accuracy.pdf")
                    csv_df = pd.DataFrame({
                        'batchnum': range(1, num_runs + 1),
                        **top1_dict
                    })
                    csv_df.to_csv(
                        f"{store_path}/vread{v}_tread{t}_accuracy.csv",
                        index=False)
                    plt.close()

                    # Print accuracy error per batch
                    print(
                        "-----Accuracy error per batch for t_read: {t}, v_read: {v}, nn: {nn_name}-----"
                    )
                    acc_baseline = top1_dict[1]
                    for f, acc in top1_dict.items():
                        if f != 1:
                            error = 100 * (np.array(acc_baseline) -
                                           np.array(acc))
                            print(f"Update rate {f}:")
                            print(
                                f"Max. absolute error: {round(max(abs(error)), 2)}%"
                            )
                            print(
                                f"Mean absolute error: {round(np.mean(abs(error)), 2)}%"
                            )
                            print([f"{round(e, 2)}%" for e in error])

                    # Plot simulation time
                    sim_times_median = []
                    labels = []
                    for f_idx, f in enumerate(freq):
                        df_nn_f = df_nn_t[(
                            df_nn_t['read_disturb_update_freq'] == f)]
                        assert len(df_nn_f) == 1
                        labels.append(f"{f}")
                        sim_time = df_nn_f['sim_time_batch_ns'].iloc[0]
                        sim_time = ast.literal_eval(sim_time)
                        sim_times_median.append(np.median(sim_time))

                    plt.figure(figsize=(4.2, 3))
                    plt.rcParams['font.family'] = 'serif'
                    plt.xlabel('Update frequency', fontsize=14)
                    plt.ylabel('Simulation time (ns)', fontsize=14)
                    plt.bar(x=range(1,
                                    len(freq) + 1),
                            height=sim_times_median,
                            label=labels,
                            color=colors[:len(freq)])
                    plt.xticks(ticks=range(1,
                                           len(freq) + 1),
                               rotation=45,
                               labels=freq,
                               ha='center',
                               fontsize=14)
                    plt.yticks(fontsize=14)
                    plt.legend(loc='upper left', fontsize=12, ncol=2)
                    plt.tight_layout()
                    plt.savefig(f"{store_path}/vread{v}_tread{t}_sim_time.png")
                    plt.savefig(f"{store_path}/vread{v}_tread{t}_sim_time.pdf")
                    csv_df = pd.DataFrame({
                        'Freq': freq,
                        'SimTime': sim_times_median
                    })
                    csv_df.to_csv(
                        f"{store_path}/vread{v}_tread{t}_sim_time.csv",
                        index=False)
                    plt.close()

                    # Plot speedup
                    speedup = []
                    df_nn_f_1 = df_nn_t[(
                        df_nn_t['read_disturb_update_freq'] == 1)]

                    if len(df_nn_f_1) == 1:
                        sim_time_batch_ns_1 = df_nn_f_1[
                            'sim_time_batch_ns'].iloc[0]
                        sim_time_batch_ns_1 = ast.literal_eval(
                            sim_time_batch_ns_1)
                        sim_time_batch_ns_1 = np.median(sim_time_batch_ns_1)

                        labels = []
                        for f_idx, f in enumerate(freq):
                            df_nn_f = df_nn_t[(
                                df_nn_t['read_disturb_update_freq'] == f)]
                            assert len(df_nn_f) == 1
                            labels.append(f"{f}")
                            sim_time = df_nn_f['sim_time_batch_ns'].iloc[0]
                            sim_time = ast.literal_eval(sim_time)
                            sim_times_median = np.median(sim_time)
                            speedup.append(sim_time_batch_ns_1 /
                                           sim_times_median)

                        plt.figure(figsize=(4.2, 3))
                        plt.rcParams['font.family'] = 'serif'
                        plt.xlabel('Update frequency', fontsize=14)
                        plt.ylabel('Speedup', fontsize=14)
                        plt.bar(x=range(1,
                                        len(freq) + 1),
                                height=speedup,
                                label=labels,
                                color=colors[:len(freq)])
                        plt.xticks(ticks=range(1,
                                               len(freq) + 1),
                                   labels=freq,
                                   rotation=45,
                                   ha='center',
                                   fontsize=14)
                        plt.yticks(fontsize=14)
                        plt.legend(loc='upper right', fontsize=12, ncol=2)
                        plt.tight_layout()
                        plt.savefig(
                            f"{store_path}/vread{v}_tread{t}_speedup.png")
                        plt.savefig(
                            f"{store_path}/vread{v}_tread{t}_speedup.pdf")
                        csv_df = pd.DataFrame({
                            'Freq': freq,
                            'Speedup': speedup
                        })
                        csv_df.to_csv(
                            f"{store_path}/vread{v}_tread{t}_speedup.csv",
                            index=False)
                        plt.close()


def rd_overhead_plot(df: pd.DataFrame,
                     df_baseline: pd.DataFrame,
                     store_path: str,
                     s_cat: list,
                     d_cat: list,
                     plt_legend_nr: int = -1) -> None:
    baseline = [dict(row) for idx, row in df_baseline.iterrows()]
    for bl in baseline:
        bl_sim_time = ast.literal_eval(bl['sim_time_batch_ns'])
        keys = [
            'nn_data_set', 'nn_data', 'batch', 'num_runs', 'digital_only',
            'adc_type', 'verbose', 'nn_name', 'xbar_size', 'hrs_lrs', 'm_mode',
            'hrs_noise', 'lrs_noise', 'ifm'
        ]
        df_entries = df[np.logical_and.reduce(
            [df[key] == bl[key] for key in keys])]

        for v in df_entries['V_read'].unique():
            df_v = df_entries[(df_entries['V_read'] == v)]
            for t in df_entries['t_read'].unique():
                df_t = df_v[(df_v['t_read'] == t)]
                assert len(
                    df_t['read_disturb_update_freq'].unique()) == len(df_t)
                print(
                    f"Read disturb simulation overhead for V_read: {v}, t_read: {t}"
                )
                for f in df_t['read_disturb_update_freq'].unique():
                    df_f = df_t[(df_t['read_disturb_update_freq'] == f)]
                    sim_time = ast.literal_eval(
                        df_f['sim_time_batch_ns'].iloc[0])
                    overhead_perc = 100 * (
                        np.median(sim_time) / np.median(bl_sim_time) - 1)
                    print(f"Update freq: {f}: {round(overhead_perc, 2)}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        help='Path to experiment config',
                        required=True)

    args = parser.parse_args()

    with open(args.config, 'r') as json_file:
        cfg = json.load(json_file)

    exp_name = args.config.split('/')[-1].split('.json')[0]
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))

    if exp_name == 'read_disturb_simulation_overhead':
        exp_result_path = repo_path + '/results/read_disturb_simulation_time'
        baseline_path = repo_path + '/results/' + exp_name
        try:
            df = pd.read_csv(
                f"{exp_result_path}/read_disturb_simulation_time.csv")
        except:
            print(
                f"Error: {exp_name} experiment needs both 'read_disturb_simulation_time.csv' and '{exp_name}.csv'"
            )
            raise
        df_baseline = pd.read_csv(f"{baseline_path}/{exp_name}.csv")
    else:
        exp_result_path = repo_path + '/results/' + exp_name
        df = pd.read_csv(f"{exp_result_path}/{exp_name}.csv")

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

    if exp_name in ['adc', 'adc_vgg7']:
        adc_alpha_plot(df=df,
                       store_path=store_path,
                       s_cat=cat_static,
                       d_cat=cat_dynamic)
    elif exp_name == 'lrs_var':
        variability_lrs_plot(df=df,
                             store_path=store_path,
                             s_cat=cat_static,
                             d_cat=cat_dynamic,
                             plt_legend_nr=2)
    elif exp_name == 'hrs_var':
        variability_hrs_plot(df=df,
                             store_path=store_path,
                             s_cat=cat_static,
                             d_cat=cat_dynamic,
                             plt_legend_nr=0)
    elif exp_name == 'read_disturb_simulation_time':
        rd_sim_time_plot(df=df,
                         store_path=store_path,
                         s_cat=cat_static,
                         d_cat=cat_dynamic)
    elif exp_name == 'read_disturb_simulation_overhead':
        rd_overhead_plot(df=df,
                         df_baseline=df_baseline,
                         store_path=store_path,
                         s_cat=cat_static,
                         d_cat=cat_dynamic)
    else:
        raise Exception(f"Plot for experiment {exp_name} not implemented.")
