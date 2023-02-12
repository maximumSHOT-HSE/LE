import argparse
import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-path", type=str, required=True, help="Path to the csv file with features (with <id> and <full_text> columns)")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to the tokenizer")
    parser.add_argument("--save-ds-path", type=str, required=True, help="Path to the file where constructed HF dataset will be saved")
    return parser


def parse_args():
    return get_parser().parse_args()


def tokenize_text(x, tokenizer):
    try:
        t = tokenizer(
            x["full_text"],
            padding="max_length",
            truncation=True
        )
        return t
    except:
        print(x)
        exit(0)


def main(args):
    df = pd.read_csv(args.features_path)
    print(df[df["id"].isnull()])
    df = df[["id", "full_text"]]
    ds = Dataset.from_pandas(df)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = ds.map(lambda x: tokenize_text(x, tokenizer))
    ds.save_pretrained(args.save_ds_path)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
