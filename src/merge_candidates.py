import argparse
import json
import pandas as pd
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-to", type=str, required=True)
    parser.add_argument("--csv-from", type=str, required=True)
    parser.add_argument("--csv-save", type=str, required=True)
    return parser


def parse_args():
    return get_parser().parse_args()


def main(args):
    df_to = pd.read_csv(args.csv_to)
    df_from = pd.read_csv(args.csv_from)
    
    data_from = {
        row["topic_id"]: set(row["content_ids"].split(" ")) 
        for row in tqdm(df_from.iloc, total=len(df_from))
    }
    
    df_to["merged_content_ids"] = df_to.apply(
        lambda x: " ".join(set(x["content_ids"].split(" ")) | data_from.get(x["topic_id"], set())),
        axis=1
    )
    
    df_to = df_to[["topic_id", "merged_content_ids"]]
    df_to = df_to.rename(columns={"merged_content_ids": "content_ids"})
    df_to.to_csv(args.csv_save, index=False)
    

if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
