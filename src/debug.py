import argparse
import json
import os
import pandas as pd


def get_parser():
    parser = argparse.ArgumentParser()
    return parser


def parse_args():
    return get_parser().parse_args()


def main(args):
    data_root = "/home/mksurkov/LE/data"
    content_df = pd.read_csv(os.path.join(data_root, "content.short.csv")).fillna("").head(5)
    print(content_df)
    print()
    for row in content_df.iloc:
        print(row)
        print()
    print("=" * 100)
    topics_df = pd.read_csv(os.path.join(data_root, "topics.csv")).fillna("").head(5)
    print(topics_df)
    print()
    for row in topics_df.iloc:
        print(row)
        print()
    


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
