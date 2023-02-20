import argparse
import json
import pandas as pd
from tqdm import tqdm
from src.utils import calculate_f_beta_score
from collections import defaultdict
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates-csv-path", type=str, required=True)
    parser.add_argument("--correlations-csv-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    return parser


def parse_args():
    return get_parser().parse_args()


def main(args):
    candidates_df = pd.read_csv(args.candidates_csv_path)
    correlations_df = pd.read_csv(args.correlations_csv_path)
    
    topic_id_to_true_content_ids = {
        row["topic_id"]: set(row["content_ids"].split(" "))
        for row in tqdm(correlations_df.iloc, total=len(correlations_df))
    }
    
    metric_name_to_values = defaultdict(list)
    
    for row in tqdm(candidates_df.iloc, total=len(candidates_df)):
        topic_id = row["topic_id"]
        predicted_content_ids = set(row["content_ids"].split(" "))
        true_content_ids = topic_id_to_true_content_ids[topic_id]
        
        intersection_size = len(predicted_content_ids & true_content_ids)
        precision = intersection_size / len(predicted_content_ids)
        recall = intersection_size / len(true_content_ids)
        
        f1_score = calculate_f_beta_score(precision, recall, beta=1)
        f2_score = calculate_f_beta_score(precision, recall, beta=2)
        
        metric_name_to_values["predicion"].append(precision)
        metric_name_to_values["recall"].append(recall)
        metric_name_to_values["f1-score"].append(f1_score)
        metric_name_to_values["f2-score"].append(f2_score)
        
    metrics = {
        f"mean_{metric_name}": np.mean(values)
        for metric_name, values in metric_name_to_values.items()
    }
    
    print(json.dumps(metrics, indent=2))
    with open(args.save_path, "w") as fout:
        json.dump(metrics, fout, indent=2)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
