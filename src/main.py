import argparse
import json

from model_parser import *
from run import *

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
    
    exp = create_experiment(cfg)
    run_experiments(exp, exp_name)
