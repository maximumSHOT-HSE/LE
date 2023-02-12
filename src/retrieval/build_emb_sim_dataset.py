import argparse
import json
from datasets import Dataset, DatasetDict
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dd-path", type=str, required=True, help="Path to the HF dd with tokens")
    parser.add_argument("--correlations-csv-path", type=str, required=True, help="Path to the csv file with positive samples")
    parser.add_argument("--save-path", type=str, required=True, 
                        help="Path to the HF dd built for embedding learning problem will be saved")
    return parser


def parse_args():
    return get_parser().parse_args()


def construct_emb_sim_dataset(topics_ds, content_id_to_full_text, topic_id_to_correlated_content_ids):
    data = defaultdict(list)
    
    def add_sample(topic_id, content_id, topic_full_text):
        data["topic_id"].append(topic_id)
        data["content_id"].append(content_id)
        data["topic_full_text"].append(topic_full_text)
        data["content_full_text"].append(content_id_to_full_text[content_id])

    for row in tqdm(topics_ds):
        topic_id = row["id"]
        correlated_content_ids = topic_id_to_correlated_content_ids[topic_id]
        topic_full_text = row["full_text"]
        for content_id in correlated_content_ids:
            add_sample(topic_id, content_id, topic_full_text)
    
    return Dataset.from_dict(data)


def main(args):
    dd = DatasetDict.load_from_disk(args.dd_path)
    correlations_df = pd.read_csv(args.correlations_csv_path)
    
    topic_id_to_correlated_content_ids = {
        row["topic_id"]: set(row["content_ids"].split(" "))
        for row in tqdm(correlations_df.iloc, total=len(correlations_df))
    }
    
    content_id_to_full_text = {
        row["id"]: row["full_text"]
        for row in tqdm(dd["content"], total=len(dd["content"]))
    }
    
    train_ds = construct_emb_sim_dataset(
        dd["train_topics"], 
        content_id_to_full_text, 
        topic_id_to_correlated_content_ids
    )
    
    validation_ds = construct_emb_sim_dataset(
        dd["validation_topics"],
        content_id_to_full_text,
        topic_id_to_correlated_content_ids
    )
    
    DatasetDict({"train": train_ds, "validation": validation_ds}).save_to_disk(args.save_path)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
