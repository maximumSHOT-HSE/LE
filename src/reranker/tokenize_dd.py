import argparse
import json
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dd-path", type=str, required=True, help="Path to the HF dd")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to the dir with tokenizer")
    parser.add_argument("--save-path", type=str, required=False, help="Path where HF dd with tokens will be saved")
    return parser


def parse_args():
    return get_parser().parse_args()


def main(args):
    try:
        data = DatasetDict.load_from_disk(args.dd_path)
    except:
        data = Dataset.load_from_disk(args.dd_path)
    print(data)
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(tokenizer)
    
    data = data.map(
        lambda x: tokenizer(
            x["topic_full_text"], 
            x["content_full_text"],
            padding="max_length", 
            truncation=True,
            max_length=256
        )
    )

    save_path = args.dd_path if args.save_path is None else args.save_path
    data.save_to_disk(save_path)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
