import argparse
import json


def get_parser():
    parser = argparse.ArgumentParser()
    return parser


def parse_args():
    return get_parser().parse_args()


def main(args):
    pass


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2), flush=True)
    main(args)
