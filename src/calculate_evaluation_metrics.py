import argparse
import json
import pandas as pd
from tqdm import tqdm
from src.utils import calculate_f_beta_score
from collections import defaultdict, Counter
import numpy as np
from datasets import Dataset, DatasetDict
import random
from copy import deepcopy


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dd", type=str, required=True)
    parser.add_argument("--dd-with-emb-path", type=str, required=True)
    parser.add_argument("--candidates-csv-path", type=str, required=True)
    parser.add_argument("--correlations-csv-path", type=str, required=True)
    return parser


def parse_args():
    return get_parser().parse_args()


def main(args):
    dd_emb = DatasetDict.load_from_disk(args.dd_with_emb_path)
    id_to_emb = {}
    for k in dd_emb.keys():
        for row in dd_emb[k]:
            id_to_emb[row["id"]] = np.array(row["embedding"]).reshape(-1)
    print(dd_emb)
    
    dd = DatasetDict.load_from_disk(args.dd)
    print(dd)
    
    id_to_full_text = {}
    for k in dd.keys():
        for row in dd[k]:
            id_to_full_text[row["id"]] = row["full_text"]
    
    topic_ids_for_evaluation = set(dd["validation_topics"]["id"])
    
    print(f"number of different topics = {len(topic_ids_for_evaluation)}")
    
    candidates_df = pd.read_csv(args.candidates_csv_path)
    correlations_df = pd.read_csv(args.correlations_csv_path)
    
    topic_id_to_true_content_ids = {
        row["topic_id"]: set(row["content_ids"].split(" "))
        for row in tqdm(correlations_df.iloc, total=len(correlations_df))
    }
    
    topic_id_to_predicted_content_ids = {
        row["topic_id"]: set(row["content_ids"].split(" "))
        for row in tqdm(candidates_df.iloc, total=len(candidates_df))
    }
    
    def calculate_metrics(dist_threshold):
        metric_name_to_values = defaultdict(list)
        for topic_id in tqdm(topic_ids_for_evaluation):
            true_content_ids = deepcopy(topic_id_to_true_content_ids.get(topic_id, set()))
            predicted_content_ids = deepcopy(topic_id_to_predicted_content_ids.get(topic_id, set()))
            
            predicted_content_ids = set(filter(
                lambda content_id: np.linalg.norm(
                    id_to_emb[topic_id] - id_to_emb[content_id]
                ) < dist_threshold,
                predicted_content_ids
            ))

            intersection_size = len(predicted_content_ids & true_content_ids)
            
            precision = intersection_size / (len(predicted_content_ids) + 1e-9)
            recall = intersection_size / (len(true_content_ids) + 1e-9)
            
            f1_score = calculate_f_beta_score(precision, recall, beta=1)
            f2_score = calculate_f_beta_score(precision, recall, beta=2)
            
            metric_name_to_values["precision"].append(precision)
            metric_name_to_values["recall"].append(recall)
            metric_name_to_values["f1-score"].append(f1_score)
            metric_name_to_values["f2-score"].append(f2_score)
        
        metrics = {
            f"mean_{metric_name}": np.mean(values)
            for metric_name, values in metric_name_to_values.items()
        }
        metrics["dist_threshold"] = dist_threshold
        return metrics
        
    for dist_threshold in np.linspace(0, 20, 11):
        metrics = calculate_metrics(dist_threshold)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
