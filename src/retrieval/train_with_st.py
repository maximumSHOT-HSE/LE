import argparse
import json
from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer, models, InputExample, losses
from tqdm import tqdm
from torch.utils.data import DataLoader
import os


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dd-path", type=str, required=True, help="Path to the HF dd prepared for retrieval training")
    parser.add_argument("--exp-dir-path", type=str, required=True, help="Path to the experiment directory")
    parser.add_argument("--st-name", type=str, required=True, help="The name of Sentence Transformer to be trained")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--checkpoint_save_steps", type=int, default=5000)
    parser.add_argument("--full", type=int, default=0)
    return parser


def parse_args():
    return get_parser().parse_args()


def main(args):
    dd = DatasetDict.load_from_disk(args.dd_path)
    print(dd)
    
    train_examples = []
    keys = ["train"]
    if args.full:
        keys.append("validation")
    for key in keys:
        print(f"add data from {key} part", flush=True)
        for row in tqdm(dd[key]):
            topic_text = row["topic_text"]
            content_text = row["content_text"]
            train_examples.append(InputExample(texts=[topic_text, content_text]))

    model = SentenceTransformer(args.st_name)
    model.max_seq_length = 128
    print(model)
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    num_epochs = args.num_epochs
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1) #10% of train data

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        save_best_model=True,
        output_path=os.path.join(args.exp_dir_path, "st_best_checkpoint"),
        warmup_steps=warmup_steps,
        
        checkpoint_path=os.path.join(args.exp_dir_path, "st_all_checkpoints"),
        checkpoint_save_steps=args.checkpoint_save_steps,
        checkpoint_save_total_limit=3
    )


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
