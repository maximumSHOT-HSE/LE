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

    topics_X = np.array(dd["validation_topics"]["embedding"]).squeeze()
    content_X = np.array(dd["train_content"]["embedding"] + dd["validation_content"]["embedding"]).squeeze()
    
    n_train_content_ids = len(dd["train_content"])
    n_validation_content_ids = len(dd["validation_content"])
    
    print(f"embeddings loaded", flush=True)
    print(f"topics = {topics_X.shape} | conent = {content_X.shape}", flush=True)
    
    kd_tree = KDTree(content_X)
    print(f"kd tree fitted", flush=True)
    
    candidates = kd_tree.query(topics_X, k=args.n_neighbors, return_distance=False)
    print(f"candidates shape = {candidates.shape}", flush=True)
    
    data = defaultdict(list)
    
    for i in tqdm(range(len(topics_X))):
        topic_id = dd["validation_topics"][i]["id"]
        content_ids = " ".join(
            list(map(
                lambda j: dd["train_content"][int(j)]["id"] if j < n_train_content_ids else \
                    dd["validation_content"][int(j) - n_train_content_ids]["id"],
                candidates[i]
            ))
        )
        data["topic_id"].append(topic_id)
        data["content_ids"].append(content_ids)
    
    pd.DataFrame.from_dict(data).to_csv(args.save_path, index=False)
    

if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
