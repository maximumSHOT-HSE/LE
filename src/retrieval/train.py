import argparse
import json
from src.retrieval.sentence_embedding_model import AutoModelForSentenceEmbedding
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import DatasetDict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import os


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-config", type=str, required=True)
    return parser


def parse_args():
    return get_parser().parse_args()


class TorchDataset(Dataset):
    
    def __init__(self, ds):
        super().__init__()
        self.data = []
        for i, row in enumerate(ds):
            self.data.append(row)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i]


def main(args):
    with open(args.training_config, "r") as f:
        train_config = json.load(f)

    print(train_config, flush=True)

    encoder = AutoModel.from_pretrained(train_config["model_path"])
    tokenizer = AutoTokenizer.from_pretrained(train_config["tokenizer_path"])
    model = AutoModelForSentenceEmbedding(encoder, tokenizer)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"train on device = {device}", flush=True)
    model = model.to(device)
    
    optimizer = AdamW(params=model.parameters(), lr=train_config["lr"], correct_bias=True)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=train_config["num_warmup_steps"],
        num_training_steps=train_config["num_training_steps"]
    )
    
    CEL = nn.CrossEntropyLoss()
    max_grad_norm = train_config["max_grad_norm"]
    
    model.train()
    
    dd = DatasetDict.load_from_disk(train_config["dd_path"])
    dls = {
        "train": DataLoader(TorchDataset(dd["train"]), batch_size=train_config["batch_size"], shuffle=True),
        # "validation": DataLoader(dd["validation"], batch_size=train_config["batch_size"], shuffle=False)
    }

    losses = []
    
    batch_it = iter(dls["train"])
    for step in tqdm(range(train_config["num_training_steps"])):
        try:
            batch = next(batch_it)
        except StopIteration as e:
            batch_it = iter(dls["train"])
            batch = next(batch_it)
            print(f"reinit batch iterator on step = {step}", flush=True)
        topic_input_ids = torch.stack(batch["topic_input_ids"]).transpose(1, 0).long().to(device)
        topic_attention_mask = torch.stack(batch["topic_attention_mask"]).transpose(1, 0).long().to(device)
        
        content_input_ids = torch.stack(batch["content_input_ids"]).transpose(1, 0).long().to(device)
        content_attention_mask = torch.stack(batch["content_attention_mask"]).transpose(1, 0).long().to(device)
        
        topic_emb = model(
            input_ids=topic_input_ids,
            attention_mask=topic_attention_mask
        )
        content_emb = model(
            input_ids=content_input_ids,
            attention_mask=content_attention_mask
        )
        
        scores = torch.mm(topic_emb, content_emb.transpose(0, 1)) * train_config["scale"]
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=device)
        loss = (CEL(scores, labels) + CEL(scores.transpose(0, 1), labels)) * 0.5
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        lr_scheduler.step()
        
        if (step + 1) % train_config["save_steps"] == 0:
            save_path = os.path.join(train_config["checkpoints"], "encoder.ckpt")
            model.model.save_pretrained(save_path)
        
        print(f"loss = {loss}", flush=True)
        losses.append(loss)
        
    with open(os.path.join(train_config["checkpoints"], "losses.txt"), "w") as fout:
        fout.write(str(losses))


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
