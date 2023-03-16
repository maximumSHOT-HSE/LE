import argparse
import json
from datasets import DatasetDict
import pandas as pd


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dd-with-proba-path", type=str, required=True)
    parser.add_argument("--dd-with-embeddings-path", type=str, required=True)
    parser.add_argument("--candidates-csv-path", type=str, required=True)
    parser.add_argument("--correlations-csv-path", type=str, required=True)
    parser.add_argument("--save-csv-path", type=str, required=True)
    return parser


def parse_args():
    return get_parser().parse_args()


def main(args):
    dd_with_proba = DatasetDict.load_from_disk(args.dd_with_proba_path)
    dd_with_emb = DatasetDict.load_from_disk(args.dd_with_embeddings_path)
    candidates_df = pd.read_csv(args.candidates_csv_path)
    correlations_df = pd.read_csv(args.correlations_csv_path)
    
    print(dd_with_proba)
    print(dd_with_emb)
    print(candidates_df)
    print(correlations_df)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
