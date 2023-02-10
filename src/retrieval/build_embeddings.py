import argparse
import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
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


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def normalize(x, eps: float = 1e-9):
    l2_norm = torch.linalg.vector_norm(x, ord=2)
    return x / max(l2_norm, eps)


def apply_encoder(model, ds: Dataset):
    model = model.eval()
    embeddings = []
    with torch.no_grad():
        for sample in tqdm(ds):
            input_ids = torch.Tensor(sample["input_ids"]).long()
            token_type_ids = torch.Tensor(sample["token_type_ids"]).long()
            attention_mask = torch.Tensor(sample["attention_mask"]).long()
            model_out = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask
            )
            emb = mean_pooling(model_out, attention_mask).reshape(-1)
            emb = normalize(emb).numpy()
            embeddings.append(emb)
    return np.stack(embeddings)
            

def main(args):
    dd = DatasetDict.load_from_disk(args.dd_path)
    print(dd)
    
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(args.retriever, "tokenizer"))
    print(tokenizer)
    
    dd = dd.map(
        lambda x: tokenizer(
            x["full_text"], 
            padding="max_length", 
            truncation=True, 
            return_tensors="np", 
            max_length=128
        )
    )
    
    topic_encoder = AutoModel.from_pretrained(os.path.join(args.retriever, "topic_enc"))
    content_encoder = AutoModel.from_pretrained(os.path.join(args.retriever, "content_enc"))
    
    topic_embeddings = apply_encoder(topic_encoder, dd["topics"])
    content_embeddings = apply_encoder(content_encoder, dd["content"])

    dd["topics"] = dd["topics"].map(lambda x, i: {"embedding": topic_embeddings[i]}, with_indices=True)
    dd["content"] = dd["content"].map(lambda x, i: {"embedding": content_embeddings[i]}, with_indices=True)
    
    save_path = args.dd_path if args.save_path is None else args.save_path
    
    dd.save_to_disk(save_path)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
