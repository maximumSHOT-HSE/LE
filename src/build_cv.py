import argparse
import json
import random
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import pandas as pd
from collections import deque, Counter
from typing import List
import numpy as np
import os


def get_parser():
    parser = argparse.ArgumentParser(
        description=
        """
            Let us build a bipartite graph 
                topics and contents are vertices
                if topic and content are correlated then we will create an edge
                between them
            
            Find all connectivity components
            Analyse their sizes and try to split them into folds in the following way:
                random shuffle components, then split them into n_folds parts iteratively
                adding elements to the minimum size fold
        """
    )
    parser.add_argument("--topics-ds-path", type=str, required=True)
    parser.add_argument("--content-ds-path", type=str, required=True)
    parser.add_argument("--correlations-csv-path", type=str, required=True, help="Path to the file with correlations info")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--save-dir", type=str, required=True, 
                        help="Path to the directory where built folds datasets wil be saved")
    parser.add_argument("--n-folds", type=int, required=True, help="The number of folds in CV")
    return parser


def parse_args():
    return get_parser().parse_args()


def get_components(graph: List[List[int]]) -> List[int]:
    n_components = 0
    n_vertices = len(graph)
    component_id = [-1 for _ in range(n_vertices)]
    for s in tqdm(range(n_vertices)):
        if component_id[s] != -1:
            continue
        component_id[s] = n_components
        q = deque()
        q.appendleft(s)
        while len(q) > 0:
            v = q.popleft()
            for to in graph[v]:
                if component_id[to] == -1:
                    component_id[to] = n_components
                    q.append(to)
        n_components += 1
    return component_id


def split_array_into_folds(a: List, n_folds: int) -> List[List[int]]:
    permutation = list(range(len(a)))
    random.shuffle(permutation)
    folds = [[] for _ in range(n_folds)]
    sum_in_fold = [0 for _ in range(n_folds)]
    for i in permutation:
        x = a[i]
        min_sum_fold = 0
        for j in range(n_folds):
            if sum_in_fold[min_sum_fold] > sum_in_fold[j]:
                min_sum_fold = j
        folds[min_sum_fold].append(i)
        sum_in_fold[min_sum_fold] += x
    return folds
  


def main(args):
    random.seed(args.seed)

    correlations_df = pd.read_csv(args.correlations_csv_path)
    
    id_to_vertex_number = {}
    graph = []
    ids = []
    
    def get_vertex_number(id: str) -> int:
        if id not in id_to_vertex_number:
            vertex_number = len(id_to_vertex_number)
            id_to_vertex_number[id] = vertex_number
            graph.append([])
            ids.append(id)
        return id_to_vertex_number[id]
    
    def add_edge(id1, id2):
        v1 = get_vertex_number(id1)
        v2 = get_vertex_number(id2)
        graph[v1].append(v2)
        graph[v2].append(v1)
    
    for row in tqdm(correlations_df.iloc, total=len(correlations_df)):
        topic_id = row["topic_id"]
        content_ids = row["content_ids"].split(" ")
        for content_id in content_ids:
            add_edge(topic_id, content_id)
        
    component_id = get_components(graph)
    component_id_to_vertices = [[] for _ in component_id]
    for v, cid in enumerate(component_id):
        component_id_to_vertices[cid].append(v)
    
    component_ids_ar = []
    component_sizes_ar = []
    for cid, sz in Counter(component_id).items():
        component_ids_ar.append(cid)
        component_sizes_ar.append(sz)
    
    folds = split_array_into_folds(component_sizes_ar, args.n_folds)
    
    topics_ds = Dataset.load_from_disk(args.topics_ds_path)
    content_ds = Dataset.load_from_disk(args.content_ds_path)
    
    for fold_id, fold in tqdm(enumerate(folds)):
        fold = list(map(lambda j: component_ids_ar[j], fold))
        fold_ids = set()
        for cid in fold:
            for v in component_id_to_vertices[cid]:
                id = ids[v]
                fold_ids.add(id)
        
        fold_topics_ds = topics_ds.filter(lambda x: x["id"] in fold_ids)
        fold_content_ds = content_ds.filter(lambda x: x["id"] in fold_ids)
        
        fold_dd = DatasetDict({"topics": fold_topics_ds, "content": fold_content_ds})
        fold_path = os.path.join(f"{args.save_dir}.seed_{args.seed}.n_folds_{args.n_folds}", f"fold_{fold_id}")
        print(f"save to = {fold_path}")
        fold_dd.save_to_disk(fold_path)
        


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
