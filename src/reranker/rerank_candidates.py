import argparse
import json
from datasets import Dataset
import pandas as pd
import os
from collections import defaultdict
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds-with-proba-path", type=str, required=True)
    parser.add_argument("--thresholds-list", nargs="+", type=float, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    return parser


def parse_args():
    return get_parser().parse_args()


def build_reranked_candidates(ds, threshold):
    topic_id_to_candidates = defaultdict(list)    
    for row in tqdm(ds):
        topic_id = row["topic_id"]
        content_id = row["content_id"]
        proba = row["proba"]
        if proba > threshold:
            topic_id_to_candidates[topic_id].append((content_id, proba))
    candidates_data = defaultdict(list)
    for topic_id, content_ids_list in topic_id_to_candidates.items():
        content_ids_list.sort(key=lambda x: x[1], reverse=True)
        content_ids = " ".join(map(lambda x: x[0], content_ids_list))
        candidates_data["topic_id"].append(topic_id)
        candidates_data["content_ids"].append(content_ids)
    return pd.DataFrame.from_dict(candidates_data)


def main(args):
    ds = Dataset.load_from_disk(args.ds_with_proba_path)
    for threshold in args.thresholds_list:
        df = build_reranked_candidates(ds, threshold)
        df.to_csv(os.path.join(args.save_dir, f"candidates.threshold_{threshold}.csv"), index=False)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
