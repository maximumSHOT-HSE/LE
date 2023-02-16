import argparse
import json
from datasets import Dataset, DatasetDict
import os
from tqdm import tqdm
import pandas as pd
from collections import Counter


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-dir-path", type=str, required=True)
    parser.add_argument("--correlations-csv-path", type=str, required=True)
    return parser


def parse_args():
    return get_parser().parse_args()


def main(args):
    topic_id_to_fold = {}
    content_id_to_fold = {}

    # validate that folds do not intersect by topic ids and content ids
    for fold_path in os.listdir(args.cv_dir_path):
        assert fold_path.startswith("fold_")
        fold_id = int(fold_path[len("fold_"):])
        fold_path = os.path.join(args.cv_dir_path, fold_path)
        fold_dd = DatasetDict.load_from_disk(fold_path)
        for row in tqdm(fold_dd["validation_topics"]):
            topic_id = row["id"]
            assert topic_id not in topic_id_to_fold
            topic_id_to_fold[topic_id] = fold_id
        for row in tqdm(fold_dd["validation_content"]):
            content_id = row["id"]
            assert content_id not in content_id_to_fold
            content_id_to_fold[content_id] = fold_id
    
    # validate that content id connects topics from the same fold
    correlations_df = pd.read_csv(args.correlations_csv_path)
    content_id_to_correlated_topics_folds = {}
    fold_id_to_number_of_edges = Counter()
    total_n_edges = 0
    for row in tqdm(correlations_df.iloc, total=len(correlations_df)):
        topic_id = row["topic_id"]
        topic_fold = topic_id_to_fold[topic_id]
        for content_id in row["content_ids"].split(" "):
            if content_id not in content_id_to_correlated_topics_folds:
                content_id_to_correlated_topics_folds[content_id] = topic_fold
            assert content_id_to_correlated_topics_folds[content_id] == topic_fold
            fold_id_to_number_of_edges[topic_fold] += 1
            total_n_edges += 1

    # show the balance of folds
    fold_topics_counter = Counter(topic_id_to_fold.values())
    fold_content_counter = Counter(content_id_to_fold.values())
    total_n_topics = len(topic_id_to_fold)
    total_n_content = len(content_id_to_fold)
    
    print(f"total number of topics = {total_n_topics}")
    print(f"total number of content = {total_n_content}")
    
    for fold in sorted(fold_topics_counter.keys()):
        n_topics_in_fold = fold_topics_counter[fold]
        n_content_in_fold = fold_content_counter[fold]
        n_edges_in_fold = fold_id_to_number_of_edges[fold]
        
        print(f"fold = {fold}")
        print(f"\ttotal topics = {n_topics_in_fold}")
        print(f"\trelative topics = {round(100 * n_topics_in_fold / total_n_topics, 3)}%")
        print()
        print(f"\ttotal content = {n_content_in_fold}")
        print(f"\trelative content = {round(100 * n_content_in_fold / total_n_content, 3)}%")
        print()
        print(f"\ttotal edges = {n_edges_in_fold}")
        print(f"\trelative edges = {round(100 * n_edges_in_fold / total_n_edges, 3)}%")
        print(flush=True)

    print("OK")


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
