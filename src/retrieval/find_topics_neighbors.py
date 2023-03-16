import argparse
import json
from datasets import DatasetDict
import pandas as pd
from sklearn.neighbors import KDTree
import numpy as np
from sklearn.metrics import recall_score, precision_score, fbeta_score
from src.utils import calculate_f_beta_score
from tqdm import tqdm
from collections import defaultdict


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dd-path", type=str, required=True)
    parser.add_argument("--n-neighbors", type=int, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    return parser


def parse_args():
    return get_parser().parse_args()


def main(args):
    dd = DatasetDict.load_from_disk(args.dd_path)

    channel_id_to_topic_ids = defaultdict(set)
    topic_id_to_emb = {}

    for k in ["train_topics", "validation_topics"]:
        for row in tqdm(dd[k]):
            topic_id = row["id"]
            channel_id = row["channel"]
            channel_id_to_topic_ids[channel_id].add(topic_id)
            topic_id_to_emb[topic_id] = np.array(row["embedding"]).reshape(-1)

    data = defaultdict(list)
    
    for row in dd["validation_topics"]:
        topic_id = row["id"]
        channel_id = row["channel"]
        neighbors = list(sorted(
            channel_id_to_topic_ids[channel_id], 
            key=lambda neighbor_topic_id: np.linalg.norm(topic_id_to_emb[topic_id] - topic_id_to_emb[neighbor_topic_id])
        ))
        topic_ids = " ".join(neighbors[:args.n_neighbors])
        data["topic_id"].append(topic_id)
        data["topic_ids"].append(topic_ids)
    
    pd.DataFrame.from_dict(data).to_csv(args.save_path, index=False)
    

if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
