import argparse
import json
from datasets import Dataset, DatasetDict
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dd-path", type=str, required=True)
    parser.add_argument("--candidates-csv-path", type=str, required=True, 
                        help="Path to the csv files with retrieved candidates")
    parser.add_argument("--save-path", type=str, required=True, 
                        help="Path to the HF dd built for binary calssification problem will be saved")
    return parser


def parse_args():
    return get_parser().parse_args()


def construct_BC_dataset(topics_ds, content_id_to_full_text, topic_id_to_candidates_content_ids):
    data = defaultdict(list)
    
    def add_sample(topic_id, content_id, topic_full_text):
        data["topic_id"].append(topic_id)
        data["content_id"].append(content_id)
        data["topic_full_text"].append(topic_full_text)
        data["content_full_text"].append(content_id_to_full_text[content_id])

    for row in tqdm(topics_ds):
        topic_id = row["id"]
        if topic_id not in topic_id_to_candidates_content_ids:
            continue
        candidates_content_ids = topic_id_to_candidates_content_ids[topic_id]
        topic_full_text = row["full_text"]
        for content_id in candidates_content_ids:
            add_sample(topic_id, content_id, topic_full_text)
    
    return Dataset.from_dict(data)


def main(args):
    dd = DatasetDict.load_from_disk(args.dd_path)
    candidates_df = pd.read_csv(args.candidates_csv_path)
    
    topic_id_to_candidates_content_ids = {
        row["topic_id"]: set(row["content_ids"].split(" ")) 
        for row in tqdm(candidates_df.iloc, total=len(candidates_df))
    }
    
    content_id_to_full_text = {}
    for key in ["train_content", "validation_content"]:
        for row in tqdm(dd[key], total=len(dd[key]))    :
            content_id_to_full_text[row["id"]] = row["full_text"]
    
    test_ds = construct_BC_dataset(
        dd["validation_topics"], 
        content_id_to_full_text, 
        topic_id_to_candidates_content_ids
    )
    
    test_ds.save_to_disk(args.save_path)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
