import argparse
import json
import pandas as pd
from collections import defaultdict


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates-csv-path", type=str, required=True)
    parser.add_argument("--head-size", type=int, required=True)
    parser.add_argument("--save-csv-path", type=str, required=True)
    return parser


def parse_args():
    return get_parser().parse_args()


def take_first_n_elements(content_ids: str, head_size: int):
    return " ".join(content_ids.split(" ")[:head_size])


def main(args):
    df = pd.read_csv(args.candidates_csv_path)
    df["head_content_ids"] = df.apply(lambda x: take_first_n_elements(x["content_ids"], args.head_size), axis=1)
    df = df.drop(columns=["content_ids"])
    df = df.rename(columns={"head_content_ids": "content_ids"})
    print(df)
    df.to_csv(args.save_csv_path, index=False)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
