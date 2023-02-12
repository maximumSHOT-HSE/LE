import argparse
import json
import pandas as pd
from datasets import DatasetDict
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dd-path", type=str, required=True, help="Path to the HF dd")
    parser.add_argument("--topics-csv-path", type=str, required=True, help="Path to the csv file with topics related data")
    parser.add_argument("--content-csv-path", type=str, required=True, help="Path to the csv file with content related data")
    parser.add_argument("--save-path", type=str, required=False, help="Path to the place where filtered HF dd will be saved")
    return parser


def parse_args():
    return get_parser().parse_args()


def main(args):
    dd = DatasetDict.load_from_disk(args.dd_path)
    topics_df = pd.read_csv(args.topics_csv_path)
    content_df = pd.read_csv(args.content_csv_path)
    
    topic_id_to_lang = {row["id"]: row["language"] for row in tqdm(topics_df.iloc, total=len(topics_df))}
    content_id_to_lang = {row["id"]: row["language"] for row in tqdm(content_df.iloc, total=len(content_df))}
    
    dd = dd.filter(lambda x: topic_id_to_lang[x["topic_id"]] == content_id_to_lang[x["content_id"]])
    save_path = args.dd_path if args.save_path is None else args.save_path
    print(f"saving to = {save_path}")
    print(dd)
    dd.save_to_disk(save_path)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
