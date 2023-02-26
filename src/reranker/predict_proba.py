import argparse
import json
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import os
from sklearn.utils.extmath import softmax
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reranker-path", type=str, required=True)
    parser.add_argument("--ds-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    return parser


def parse_args():
    return get_parser().parse_args()


def main(args):
    model = AutoModelForSequenceClassification.from_pretrained(args.reranker_path)
    ds = Dataset.load_from_disk(args.ds_path)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=os.path.join(args.save_path, "outs"),
            logging_dir=os.path.join(args.save_path, "logs"),
            per_device_eval_batch_size=4
        )
    )
    proba = np.array(softmax(trainer.predict(test_dataset=ds).predictions))[:, 1]
    ds = ds.map(lambda x, i: {"proba": proba[i]}, with_indices=True)
    ds.save_to_disk(args.save_path)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
