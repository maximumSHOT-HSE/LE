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
import random


SEED = 100


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dd", type=str, required=True)
    parser.add_argument("--candidates-csv-path", type=str, required=True)
    parser.add_argument("--neighbors-csv-path", type=str, required=True)
    parser.add_argument("--retriever-candidates-csv-path", type=str, required=False)
    parser.add_argument("--correlations-csv-path", type=str, required=True)
    parser.add_argument("--max-new-candidates", type=int, required=True)
    parser.add_argument("--ignore-neighbors", type=int, default=1)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--lb", type=int, default=1, required=False)
    parser.add_argument("--lb-k", type=int, default=6, required=False)
    parser.add_argument("--add", type=int, default=1, required=False)
    return parser


def parse_args():
    return get_parser().parse_args()


def main(args):
    dd = DatasetDict.load_from_disk(args.dd)
    print(dd)
    
    id_to_full_text = {}
    for k in dd.keys():
        for row in dd[k]:
            id_to_full_text[row["id"]] = row["full_text"]
    
    id_to_lang = {}
    for k in dd.keys():
        for row in tqdm(dd[k]):
            id_to_lang[row["id"]] = row["language"]
    
    topic_ids_for_evaluation = set(dd["validation_topics"]["id"])
    
    print(f"number of different topics = {len(topic_ids_for_evaluation)}")
    
    candidates_df = pd.read_csv(args.candidates_csv_path)
    retriever_candidates_df = pd.read_csv(args.retriever_candidates_csv_path)
    
    topic_id_to_predicted_content_ids = {
        row["topic_id"]: set(row["content_ids"].split(" "))
        for row in tqdm(candidates_df.iloc, total=len(candidates_df))
    }
    
    topic_id_to_retriever_content_ids = {
        row["topic_id"]: list(row["content_ids"].split(" "))
        for row in tqdm(retriever_candidates_df.iloc, total=len(retriever_candidates_df))
    }
    
    correlations_df = pd.read_csv(args.correlations_csv_path)
    topic_id_to_true_content_ids = {
        row["topic_id"]: set(row["content_ids"].split(" "))
        for row in tqdm(correlations_df.iloc, total=len(correlations_df))
    }
    
    # channel_id_to_all_candidates = defaultdict(set)
    # topic_id_to_channel_id = {}
    # random.seed(SEED)
    # for k in ["train_topics", "validation_topics"]:
    #     n = len(dd[k])
    #     perm = list(range(n))
    #     random.shuffle(perm)
    #     for i in perm:
    #         row = dd[k][i]
    #         topic_id = row["id"]
    #         channel_id = row["channel"]
    #         if len(channel_id_to_all_candidates[channel_id]) < args.max_new_candidates:
    #             channel_id_to_all_candidates[channel_id] |= topic_id_to_predicted_content_ids.get(topic_id, set())
    #         topic_id_to_channel_id[topic_id] = channel_id
    
    neighbors_df = pd.read_csv(args.neighbors_csv_path)
    topic_id_to_neighbors = {
        row["topic_id"]: set(row["topic_ids"].split(" ")) 
        for row in tqdm(neighbors_df.iloc, total=len(neighbors_df))
    }
    
    topic_id_to_final_pred = {}
    for topic_id in tqdm(topic_ids_for_evaluation):
        predicted_content_ids = deepcopy(topic_id_to_predicted_content_ids.get(topic_id, set()))
        retriever_content_ids = deepcopy(topic_id_to_retriever_content_ids.get(topic_id, list()))
        
        if len(predicted_content_ids) < args.lb:
            predicted_content_ids |= set(retriever_content_ids[0:args.lb_k])
        
        predicted_content_ids |= set(retriever_content_ids[:args.add])
        topic_id_to_final_pred[topic_id] = predicted_content_ids
    
    pred_data = defaultdict(list)
    
    for topic_id in tqdm(topic_ids_for_evaluation):
        predicted_content_ids = deepcopy(topic_id_to_final_pred.get(topic_id, set()))
        true_content_ids = deepcopy(topic_id_to_true_content_ids.get(topic_id, set()))
        
        # channel_id = topic_id_to_channel_id[topic_id]
        # predicted_content_ids |= (channel_id_to_all_candidates[channel_id] & true_content_ids)
        
        if not args.ignore_neighbors:
            for neighbor_topic_id in topic_id_to_neighbors[topic_id]:
                predicted_content_ids |= topic_id_to_final_pred.get(neighbor_topic_id, set())
        
        pred_data["topic_id"].append(topic_id)
        pred_data["content_ids"].append(" ".join(predicted_content_ids))
        
    pd.DataFrame.from_dict(pred_data).to_csv(args.save_path, index=False)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
