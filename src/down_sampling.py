import argparse
import json
from datasets import Dataset, DatasetDict
from collections import Counter
import random
from math import ceil
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dd-path", type=str, required=True, help="Path to the HF dd to be downsampled")
    parser.add_argument("--ratio", type=float, required=True, help="The maximum ratio of major label to the minor label")
    parser.add_argument("--save-path", type=str, required=True, help="Path where downsampled HF dd will be saved")
    parser.add_argument("--seed", type=int, default=100)
    return parser


def parse_args():
    return get_parser().parse_args()


def downsample_dataset(ds, ratio):
    n = len(ds)
    perm = list(range(n))
    random.shuffle(perm)
    max_freq = ceil(min(Counter(ds["label"]).values()) * ratio)
    freq = Counter()
    rem = []
    for i in tqdm(perm):
        l = ds[i]["label"]
        freq[l] += 1
        if freq[l] <= max_freq:
            rem.append(i)
    return Dataset.from_dict(ds[rem])


def main(args):
    random.seed(args.seed)
    dd = DatasetDict.load_from_disk(args.dd_path)
    for k in dd:
        dd[k] = downsample_dataset(dd[k], args.ratio)
    save_path = args.dd_path if args.save_path is None else args.save_path
    print(f"saving to {save_path}")
    print(dd)
    dd.save_to_disk(args.save_path)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
