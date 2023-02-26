import argparse
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import Trainer, TrainingArguments
from transformers.data import DataCollatorForTokenClassification
from transformers.trainer_utils import EvaluationStrategy, EvalPrediction
from datasets import Dataset, DatasetDict
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.special import softmax
from torch.utils.checkpoint import checkpoint
from src.utils import calculate_f_beta_score


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--dd-path", type=str, required=True)
    parser.add_argument("--dir-path", type=str, required=True)
    return parser


def parse_args():
    return get_parser().parse_args()


def load_training_config(dir_path):
    training_args_config_path = os.path.join(dir_path, 'config.json')
    if not os.path.exists(training_args_config_path) or not os.path.isfile(training_args_config_path):
        raise FileNotFoundError(f'Can not find config with file training args by path = '
                                f'{os.path.abspath(training_args_config_path)}')
    with open(training_args_config_path, 'r') as f:
        training_args_config = json.load(f)
        return training_args_config


def load_training_args(training_args_config):
    training_args = TrainingArguments(
        output_dir=os.path.join(args.dir_path, 'outs'),
        logging_dir=os.path.join(args.dir_path, 'logs')
    )
    for k, v in training_args_config.items():
        attr = training_args.__getattribute__(k)
        if attr is not None:
            k_type = type(attr)
            v = k_type(v)
        training_args.__setattr__(k, v)
    print('TRAINING ARGS')
    print(json.dumps(training_args.to_dict(), indent=2))
    return training_args


def compute_metrics(predictions: EvalPrediction, base_recall: float):
    y_true = predictions.label_ids
    y_pred = predictions.predictions.argmax(axis=1)  # threshold == 0.5
    
    precision = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
    recall = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
    recall_with_base = recall * base_recall
    
    f1_score = calculate_f_beta_score(precision, recall, beta=1)
    f2_score = calculate_f_beta_score(precision, recall, beta=2)
        
    f1_score_with_base = calculate_f_beta_score(precision, recall_with_base, beta=1)
    f2_score_with_base = calculate_f_beta_score(precision, recall_with_base, beta=2)

    return {
        "accuracy": accuracy_score(y_true=y_true, y_pred=y_pred),
        
        "precision": precision,
        "recall": recall,
        "f1-score": f1_score,
        "f2-score": f2_score,
        
        "base-recall": base_recall,
        "recall-with-base": recall_with_base,
        "f1-score-with-base": f1_score_with_base,
        "f2-score-with-base": f2_score_with_base
    }


def main(args):
    training_config = load_training_config(args.dir_path)
    train_args = load_training_args(training_config["training_args"])
    
    model_config = AutoConfig.from_pretrained(args.model, num_labels=training_config.get("num_labels", 2))
    model = AutoModelForSequenceClassification.from_pretrained(args.model, config=model_config, ignore_mismatched_sizes=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(tokenizer)
    print("tokenizer size = ", len(tokenizer))
    model.resize_token_embeddings(len(tokenizer))
    print(model)
    print(model_config)

    dd = DatasetDict.load_from_disk(args.dd_path)
    print(dd)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dd["train"],
        eval_dataset=dd["validation"],
        tokenizer=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, training_config.get("base_recall", 1.0))
    )
    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
