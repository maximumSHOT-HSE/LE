import argparse
import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from src.retrieval.sentence_embedding_model import AutoModelForSentenceEmbedding
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dd-path", type=str, required=True, help="Path to the HF dd")
    parser.add_argument("--retriever", type=str, required=True, help="Path to the dir with encoders")
    parser.add_argument("--save-path", type=str, required=False, help="Path where HF dd with embeddings will be saved")
    return parser


def parse_args():
    return get_parser().parse_args()


def apply_encoder(model, ds: Dataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"apply encoder on device = {device}", flush=True)
    model = model.eval().to(device)
    embeddings = []
    with torch.no_grad():
        for sample in tqdm(ds):
            input_ids = torch.Tensor(sample["input_ids"]).long().unsqueeze(dim=0).to(device)
            attention_mask = torch.Tensor(sample["attention_mask"]).long().unsqueeze(dim=0).to(device)
            emb = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).pooler_output
            emb = emb.reshape(-1).cpu().tolist()
            embeddings.append(emb)
    return ds.map(lambda x, i: {"embedding": embeddings[i]}, with_indices=True)
            

def main(args):
    dd = DatasetDict.load_from_disk(args.dd_path)
    print(dd)
    
    # encoder = AutoModel.from_pretrained(args.retriever)
    # model = AutoModelForSentenceEmbedding(encoder)
    model = AutoModel.from_pretrained(args.retriever)
    
    for k in dd:
        dd[k] = apply_encoder(model, dd[k])

    save_path = args.dd_path if args.save_path is None else args.save_path
    dd.save_to_disk(save_path)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
