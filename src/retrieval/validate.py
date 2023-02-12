import argparse
import json
from datasets import DatasetDict
import pandas as pd
from sklearn.neighbors import KDTree
import numpy as np
from sklearn.metrics import recall_score, precision_score, fbeta_score
from src.utils import calculate_f_beta_score
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dd-path", type=str, required=True)
    parser.add_argument("--correlations-csv-path", type=str, required=True)
    parser.add_argument("--n-neighbors", type=int, required=True)
    return parser


def parse_args():
    return get_parser().parse_args()


def main(args):
    dd = DatasetDict.load_from_disk(args.dd_path)
    df = pd.read_csv(args.correlations_csv_path)

    topics_X = np.array(dd["topics"]["embedding"])
    content_X = np.array(dd["content"]["embedding"])
    
    print(f"embeddings loaded", flush=True)
    
    # content_X = []
    # for i, x in enumerate(dd["content"]):
    #     if i >= 500:
    #         break
    #     content_X.append(x["embedding"])
    # content_X = np.stack(content_X)    
    # topics_X = []
    # for i, x in enumerate(dd["topics"]):
    #     if i >= 400:
    #         break
    #     topics_X.append(x["embedding"])
    # topics_X = np.stack(topics_X)
    
    kd_tree = KDTree(content_X)
    print(f"kd tree fitted", flush=True)
    candidates = kd_tree.query(topics_X, k=args.n_neighbors, return_distance=False)
    print(f"candidates shape = {candidates.shape}", flush=True)
    
    topic_id_to_content_ids = {}
    correlations_df = pd.read_csv(args.correlations_csv_path)
    for row in tqdm(correlations_df.iloc, total=len(correlations_df)):
        topic_id = row["topic_id"]
        topic_id_to_content_ids[topic_id] = set(row["content_ids"].split(" "))

    metrics_ar = []    

    for i in tqdm(range(len(topics_X))):
        topic_id = dd["topics"][i]["id"]
        y_true = topic_id_to_content_ids[topic_id]
        y_pred = set(map(lambda j: dd["content"][int(j)]["id"], candidates[i]))
        
        intersection_size = len(y_true & y_pred)
        recall = intersection_size / len(y_true)
        precision = intersection_size / len(y_pred)
        
        metrics_ar.append((precision, recall))
        
    print(f"mean recall = {np.mean([r for p, r in metrics_ar])}")
    print(f"mean precision = {np.mean([p for p, r in metrics_ar])}")
    print(f"mean f1 score = {np.mean([calculate_f_beta_score(p, r, beta=1) for p, r in metrics_ar])}")
    print(f"mean f2 score = {np.mean([calculate_f_beta_score(p, r, beta=2) for p, r in metrics_ar])}")
    


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
