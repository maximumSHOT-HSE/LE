import argparse
import json
import os
import shutil
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever-tokenizer-path", type=str, required=True)
    parser.add_argument("--retriever-path", type=str, required=True)
    parser.add_argument("--reranker-tokenizer-path", type=str, required=True)
    parser.add_argument("--reranker-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    return parser


def parse_args():
    return get_parser().parse_args()


def find_best_checkpoint(dir_path: str, metric_name: str = "eval_f2-score-with-base") -> str:
    if not os.path.exists(dir_path):
        return None
    max_metric_value = None
    best_cp_name = None
    for f in os.listdir(dir_path):
        if not f.startswith("checkpoint-"):
            continue
        ckpt_id = int(f[len("checkpoint-"):])
        if best_cp_name is None or max_metric_value < ckpt_id:
            max_metric_value = ckpt_id
            best_cp_name = os.path.join(dir_path, f)
        # with open(os.path.join(dir_path, f, "trainer_state.json"), "r") as fn:
        #     trainer_state_d = json.load(fn)
        #     metric_value = trainer_state_d["log_history"][-1][metric_name]
        #     if best_cp_name is None or max_metric_value < metric_value:
        #         max_metric_value = metric_value
        #         best_cp_name = os.path.join(dir_path, f)
    return best_cp_name


def main(args):
    shutil.copytree(args.retriever_tokenizer_path, os.path.join(args.save_path, "retriever_tokenizer"), dirs_exist_ok=False)
    
    shutil.copytree(
        os.path.join(args.retriever_path),
        os.path.join(args.save_path, "retriever", "st_checkpoints"),
        dirs_exist_ok=False
    )

    shutil.copytree(args.reranker_tokenizer_path, os.path.join(args.save_path, "reranker_tokenizer"), dirs_exist_ok=False)
    
    for f in os.listdir(args.reranker_path):
        if f.startswith("fold_"):
            best_cp_name = find_best_checkpoint(os.path.join(args.reranker_path, f, "outs"))
            if best_cp_name is None:
                continue
            shutil.copytree(best_cp_name, os.path.join(args.save_path, "reranker", f, "checkpoint"), dirs_exist_ok=False)
            for ff in os.listdir(os.path.join(args.reranker_path, f)):
                if ff.startswith("outs"):
                    continue
                ff = os.path.join(args.reranker_path, f, ff)
                if os.path.isfile(ff):
                    shutil.copyfile(ff, os.path.join(args.save_path, "reranker", f, os.path.basename(ff)))
    
    for f in Path(args.save_path).rglob("*optimizer.pt"):
        os.remove(f)
    
    submission_config = {"build_args": vars(args)}
    with open(os.path.join(args.save_path, "submission_config.json"), "w") as fout:
        json.dump(submission_config, fout, indent=2)


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
