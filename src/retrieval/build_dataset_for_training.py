import argparse
import json
from datasets import Dataset, DatasetDict
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dd-path", type=str, required=True)
    parser.add_argument("--correlations-csv-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    return parser


def parse_args():
    return get_parser().parse_args()


def main(args):
    dd = DatasetDict.load_from_disk(args.dd_path)
    correlations_df = pd.read_csv(args.correlations_csv_path)
    
    train_topics_ids = {row["id"] for row in tqdm(dd["train_topics"])}
    id_to_full_text = {}
    for ds_key in dd:
        print(ds_key)
        for row in tqdm(dd[ds_key]):
            id_to_full_text[row["id"]] = row["full_text"]
        
    data = {"train": defaultdict(list), "validation": defaultdict(list)}
    for row in tqdm(correlations_df.iloc, total=len(correlations_df)):
        topic_id = row["topic_id"]
        data_type = "train" if topic_id in train_topics_ids else "validation"
        for content_id in row["content_ids"].split(" "):
            data[data_type]["topic_id"].append(topic_id)
            data[data_type]["content_id"].append(content_id)
            data[data_type]["topic_text"].append(id_to_full_text[topic_id])
            data[data_type]["content_text"].append(id_to_full_text[content_id])
    
    for k in data.keys():
        data[k] = Dataset.from_dict(data[k])
    built_dd = DatasetDict(data)
    print(built_dd)
    built_dd.save_to_disk(args.save_path)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
