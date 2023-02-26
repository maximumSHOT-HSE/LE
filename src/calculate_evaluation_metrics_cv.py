import argparse
import json
import pandas as pd
from tqdm import tqdm
from src.utils import calculate_f_beta_score
from collections import defaultdict
import numpy as np
import os


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-dir-path", type=str, required=True)
    return parser


def parse_args():
    return get_parser().parse_args()


def main(args):
    all_metrics_values = defaultdict(list)
    for f in sorted(os.listdir(args.cv_dir_path)):
        if "fold" not in f:
            continue;
        metrics_json_path = os.path.join(args.cv_dir_path, f, "metrics.json")
        with open(metrics_json_path, "r") as fin:
            metrics_values = json.load(fin)
        print(f)
        print(json.dumps(metrics_values, indent=2))
        for k, v in metrics_values.items():
            all_metrics_values[k].append(v)
            
    all_metrics = {k: np.mean(v) for k, v in all_metrics_values.items()}
    print("all")
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
