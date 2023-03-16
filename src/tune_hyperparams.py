import argparse
import json
import os
import pandas as pd
from datasets import Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser()
    return parser


def parse_args():
    return get_parser().parse_args()


def main(args):
    ds = Dataset.load_from_disk("data/CV.full.seed_100.n_folds_5.tokenized_sentence-transformers-paraphrase-xlm-r-multilingual-v1.with_embeddings.prepared_for_reranking_inference.tokenized_sentence-transformers-paraphrase-multilingual-MiniLM-L12-v2.with_proba/fold_0/")
    retriever_candidates_df = pd.read_csv("data/CV.full.seed_100.n_folds_5.tokenized_sentence-transformers-paraphrase-xlm-r-multilingual-v1.with_embeddings/fold_0/candidates.50.csv")
    reranker_candidates_df = pd.read_csv("data/CV.full.seed_100.n_folds_5.tokenized_sentence-transformers-paraphrase-xlm-r-multilingual-v1.with_embeddings.prepared_for_reranking_inference.tokenized_sentence-transformers-paraphrase-multilingual-MiniLM-L12-v2.with_proba/fold_0/candidates.threshold_0.01.csv")
    correlations_df = pd.read_csv("data/correlations.csv")
    
    print(ds)
    print(retriever_candidates_df)
    print(reranker_candidates_df)
    print(correlations_df)
    
    # topic_ids = set(ds["topic_id"])
    topic_ids = set(correlations_df["topic_id"])
    print(len(topic_ids))
    
    topic_id_to_true_content_ids = {
        row["topic_id"]: set(row["content_ids"].split(" "))
        for row in tqdm(correlations_df.iloc, total=len(correlations_df))
    }
    
    # plt.figure(figsize=(10, 10))
    true_content_ids_sz = list(map(lambda tid: len(topic_id_to_true_content_ids.get(tid, set())), topic_ids))
    true_content_ids_sz = list(filter(lambda x: x < 60, true_content_ids_sz))
    print(np.min(true_content_ids_sz), np.max(true_content_ids_sz), np.mean(true_content_ids_sz), np.std(true_content_ids_sz))
    true_content_ids_sz.sort()
    print(true_content_ids_sz[-200:])
    plt.hist(true_content_ids_sz)
    plt.xlabel(list(range(0, 60, 4)))
    plt.savefig("tmp.png")


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
