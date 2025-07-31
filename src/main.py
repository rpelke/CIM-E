##############################################################################
# Copyright (C) 2025 Rebecca Pelke                                           #
# All Rights Reserved                                                        #
#                                                                            #
# This is work is licensed under the terms described in the LICENSE file     #
# found in the root directory of this source tree.                           #
##############################################################################
import sysconfig
import argparse
import json

from model_parser import *
from run import *

repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
python_version = '.'.join(sys.version.split('.', 2)[:2])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        help='Path to experiment config',
                        required=True)
    parser.add_argument(
        '--n_jobs',
        type=int,
        help=
        'The maximum number of concurrently running jobs (joblib parameter)',
        required=False,
        default=4)
    parser.add_argument('--debug', action='store_true', help='Use debug mode')
    parser.add_argument('--use_same_inputs',
                        action='store_true',
                        help='Use the same IFMs if num_runs > 1')
    parser.add_argument('--save_sim_stats',
                        action='store_true',
                        help='Save all simulation statistics to a file')
    parser.add_argument('--acs_lib_path',
                        type=str,
                        help='Path to the site-packages directory',
                        default=sysconfig.get_paths()["purelib"])
    parser.add_argument(
        '--emu_lib_path',
        type=str,
        help='Path to the libacs_cb_emu.so library',
        default=f"{repo_path}/build/release/lib/libacs_cb_emu.so")
    parser.add_argument('--acs_cfg_dir',
                        type=str,
                        help='Path to the ACS config directory',
                        default=f"{repo_path}/src/acs_configs")
    args = parser.parse_args()

    print(f"Run experiment with config: {args.config}")
    print(f"Number of parallel jobs: {args.n_jobs}")
    print(f"Debug mode: {args.debug}")
    print(f"Use same inputs: {args.use_same_inputs}")
    print(f"Save simulation statistics: {args.save_sim_stats}")
    print(f"ACS library path: {args.acs_lib_path}")
    print(f"EMU library path: {args.emu_lib_path}")
    print(f"ACS config directory: {args.acs_cfg_dir}")

    with open(args.config, 'r') as json_file:
        cfg = json.load(json_file)

    exp_name = args.config.split('/')[-1].split('.json')[0]

    exp = create_experiment(cfg)
    run_experiments(exp, exp_name, args.n_jobs, args.debug,
                    args.use_same_inputs, args.save_sim_stats,
                    args.acs_lib_path, args.emu_lib_path, args.acs_cfg_dir)
