import argparse
import json
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from datasets import Dataset


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics-path", type=str, required=True, help="Path to the csv file with topics related data")
    parser.add_argument("--save-path", type=str, required=True, help="Path to the HF dataset where built topics features will be saved")
    return parser


def parse_args():
    return get_parser().parse_args()


def build_topic_full_text(x):
    return (
        x["title"] + "[SEP]" +\
        x["title_path"] + "[SEP]" +\
        x["language"] + "[SEP]" +\
        x["category"] + "[SEP]" +\
        x["description"] + "[SEP]" +\
        x["parent_description"] + "[SEP]" +\
        x["children_description"]
    ).replace("\n", "*").replace("\t", "*")


def main(args):
    topics_df = pd.read_csv(args.topics_path).fillna("")
    print(topics_df.columns)
    topic_id_to_data = {row["id"]: row for row in tqdm(topics_df.iloc, total=len(topics_df))}
    topic_ids = list(topic_id_to_data.keys())
    topic_ids.sort(key=lambda i: topic_id_to_data[i]["level"])
    topic_id_to_title_path = {}
    topic_id_to_parent_description = defaultdict(str)
    topic_id_to_children_description = defaultdict(str)
    for topic_id in tqdm(topic_ids):
        data = topic_id_to_data[topic_id]
        if data["level"] > 0:
            parent_topic_id = data["parent"]
            title_path = topic_id_to_title_path[parent_topic_id] + " >> " + data["title"]
            topic_id_to_parent_description[topic_id] = topic_id_to_data[parent_topic_id]["description"]
            topic_id_to_children_description[parent_topic_id] += "+++ " + data["description"]
        else:
            title_path = data["title"]
        topic_id_to_title_path[topic_id] = title_path
    tqdm.pandas()
    topics_df["title_path"] = topics_df.progress_apply(lambda x: topic_id_to_title_path[x["id"]], axis=1)
    topics_df["parent_description"] = topics_df.progress_apply(lambda x: topic_id_to_parent_description[x["id"]], axis=1)
    topics_df["children_description"] = topics_df.progress_apply(lambda x: topic_id_to_children_description[x["id"]], axis=1)
    topics_df["full_text"] = topics_df.progress_apply(build_topic_full_text, axis=1)
    Dataset.from_pandas(topics_df).filter(lambda x: x["has_content"] == True).save_to_disk(args.save_path)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
