import argparse
import json
import pandas as pd
from tqdm import tqdm
from datasets import Dataset


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content-path", type=str, required=True, help="Path to the csv file with content related data")
    parser.add_argument("--save-path", type=str, required=True, help="Path to the HF dataset where built content features will be saved")
    return parser


def parse_args():
    return get_parser().parse_args()


def build_content_full_text(x):
    return (
        x["title"] + "[SEP]" \
        + x["language"] + "[SEP]" \
        + x["kind"] + "[SEP]" \
        + x["description"] + "[SEP]" \
        # + x["text"]
    ).replace("\n", "*").replace("\t", "*")


def main(args):
    content_df = pd.read_csv(args.content_path).fillna("")
    tqdm.pandas()
    content_df["full_text"] = content_df.progress_apply(build_content_full_text, axis=1)
    Dataset.from_pandas(content_df).save_to_disk(args.save_path)
    

if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
